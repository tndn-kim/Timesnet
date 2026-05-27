"""
Timesnet_CIC/Timesnet.py
========================
CIC-UNSW-NB15 침입 탐지 분류용 TimesNet (Self-Contained, 독립 실행 가능)

▶ 데이터셋 스펙
    - 피처 수   : 76개 (CIC-UNSW-NB15, 전처리 완료)
    - 클래스 수 : 4개  (레이블 축소 적용)
    - 입력 형태 : [B, 76] → 내부에서 [B, 1, 76] 으로 reshape

▶ 아키텍처
    DataEmbedding_inverted → ClassificationEncoder → ClassificationHead
    (layers/ 외부 의존성 없이 완전 독립)

▶ run_timesnet() 반환값
    {
        "model"  : 학습된 Model 인스턴스,
        "metrics": 평가 지표 딕셔너리 (FPR/FNR 포함),
        "config" : 사용된 TimesNetConfig,
        "history": { "train_loss", "train_acc", "val_loss", "val_acc" }
    }
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              f1_score, classification_report)
from sklearn.model_selection import train_test_split


# ══════════════════════════════════════════════════════════════════
# 1. 레이어 구성 요소 (Self-Contained)
# ══════════════════════════════════════════════════════════════════

class PositionalEmbedding(nn.Module):
    """Sinusoidal 위치 인코딩."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return self.pe[:, : x.shape[1], :]


class DataEmbedding_inverted(nn.Module):
    """
    Inverted (variate-as-token) 임베딩.

    입력: [B, T, N]  (T=1, N=76)
    출력: [B, N, d_model]
    """

    def __init__(self, c_in: int, d_model: int,
                 embed_type: str = "fixed",
                 freq: str = "h",
                 dropout: float = 0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark=None):
        # x: [B, T, N]  →  permute  →  [B, N, T]  →  linear  →  [B, N, d_model]
        out = self.value_embedding(x.permute(0, 2, 1))
        if x_mark is not None:
            out = out + self.value_embedding(x_mark.permute(0, 2, 1))
        return self.dropout(out)


class FullAttention(nn.Module):
    """Scaled Dot-Product Attention."""

    def __init__(self, mask_flag: bool = False, factor: int = 5,
                 scale=None, attention_dropout: float = 0.1,
                 output_attention: bool = False):
        super().__init__()
        self.scale            = scale
        self.mask_flag        = mask_flag
        self.output_attention = output_attention
        self.dropout          = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values,
                attn_mask=None, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D  = values.shape
        scale = self.scale or (1.0 / math.sqrt(E))

        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * scale
        if self.mask_flag and attn_mask is not None:
            scores.masked_fill_(attn_mask, float("-inf"))

        attn = self.dropout(torch.softmax(scores, dim=-1))
        out  = torch.einsum("bhls,bshd->blhd", attn, values)
        out  = out.contiguous().view(B, L, -1)

        return (out, attn) if self.output_attention else (out, None)


class AttentionLayer(nn.Module):
    """Multi-Head Attention wrapper."""

    def __init__(self, attention, d_model: int, n_heads: int,
                 d_keys=None, d_values=None):
        super().__init__()
        d_keys   = d_keys   or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_proj = nn.Linear(d_model, d_keys   * n_heads)
        self.key_proj   = nn.Linear(d_model, d_keys   * n_heads)
        self.value_proj = nn.Linear(d_model, d_values * n_heads)
        self.out_proj   = nn.Linear(d_values * n_heads, d_model)
        self.n_heads    = n_heads

    def forward(self, queries, keys, values,
                attn_mask=None, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        Q = self.query_proj(queries).view(B, L, H, -1)
        K = self.key_proj(keys)    .view(B, S, H, -1)
        V = self.value_proj(values).view(B, S, H, -1)

        out, attn = self.inner_attention(Q, K, V,
                                         attn_mask=attn_mask,
                                         tau=tau, delta=delta)
        return self.out_proj(out), attn


# ══════════════════════════════════════════════════════════════════
# 2. 분류 전용 Encoder / Head
# ══════════════════════════════════════════════════════════════════

class ClassificationEncoderLayer(nn.Module):
    """
    Self-Attention 기반 Encoder Layer.
    76개 variate 토큰 간 상호 관계 모델링.

    입력/출력: [B, N=76, d_model]
    """

    def __init__(self, attention, d_model: int, d_ff: int = None,
                 dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention  = attention
        self.conv1      = nn.Conv1d(d_model, d_ff,    kernel_size=1)
        self.conv2      = nn.Conv1d(d_ff,    d_model, kernel_size=1)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x, attn_mask=None):
        # Self-Attention + Residual
        x = x + self.dropout(self.attention(x, x, x, attn_mask=attn_mask)[0])
        x = self.norm1(x)
        # FFN + Residual
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class ClassificationEncoder(nn.Module):
    """ClassificationEncoderLayer 스택."""

    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm   = norm_layer

    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ClassificationHead(nn.Module):
    """
    Global Average Pooling → LayerNorm → MLP → num_classes.

    입력: [B, N=76, d_model]
    출력: [B, num_classes=4]
    """

    def __init__(self, d_model: int, d_ff: int, num_classes: int,
                 dropout: float = 0.1):
        super().__init__()
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.norm    = nn.LayerNorm(d_model)
        self.fc1     = nn.Linear(d_model, d_ff)
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2     = nn.Linear(d_ff, num_classes)

    def forward(self, x):
        # x: [B, N, d_model]
        x = x.transpose(1, 2)        # [B, d_model, N]
        x = self.pool(x).squeeze(-1) # [B, d_model]
        x = self.norm(x)
        x = self.dropout(self.act(self.fc1(x)))
        return self.fc2(x)            # [B, num_classes]


# ══════════════════════════════════════════════════════════════════
# 3. 통합 Model
# ══════════════════════════════════════════════════════════════════

class Model(nn.Module):
    """
    CIC-UNSW-NB15 분류용 TimesNet.

    입력  : [B, 76]
    흐름  : reshape [B,1,76] → Inverted Embedding [B,76,d] → Encoder → Head
    출력  : [B, 4]  (logits)
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name   = getattr(configs, "task_name", "classification")
        self.use_norm    = getattr(configs, "use_norm",   True)
        self.feature_dim = getattr(configs, "feature_dim", 76)
        self.num_classes = getattr(configs, "num_classes",  4)

        d_model    = configs.d_model
        d_ff       = configs.d_ff
        n_heads    = configs.n_heads
        e_layers   = configs.e_layers
        dropout    = configs.dropout
        factor     = getattr(configs, "factor",     1)
        activation = getattr(configs, "activation", "gelu")
        seq_len    = getattr(configs, "seq_len",    1)

        # Inverted Embedding : [B, 1, 76] → [B, 76, d_model]
        self.embedding = DataEmbedding_inverted(
            c_in       = seq_len,
            d_model    = d_model,
            embed_type = getattr(configs, "embed", "fixed"),
            freq       = getattr(configs, "freq",  "h"),
            dropout    = dropout,
        )

        # Self-Attention Encoder
        self.encoder = ClassificationEncoder(
            [
                ClassificationEncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag         = False,
                            factor            = factor,
                            attention_dropout = dropout,
                            output_attention  = False,
                        ),
                        d_model, n_heads,
                    ),
                    d_model    = d_model,
                    d_ff       = d_ff,
                    dropout    = dropout,
                    activation = activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer = nn.LayerNorm(d_model),
        )

        # 분류 헤드
        self.head = ClassificationHead(
            d_model     = d_model,
            d_ff        = d_ff,
            num_classes = self.num_classes,
            dropout     = dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, feature_dim=76]  또는  [B, 1, feature_dim]
        Returns:
            logits : [B, num_classes=4]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)           # [B, 1, 76]

        if self.use_norm:
            means = x.mean(dim=1, keepdim=True).detach()
            x     = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev

        x_emb   = self.embedding(x, None)   # [B, 76, d_model]
        enc_out = self.encoder(x_emb)        # [B, 76, d_model]
        return self.head(enc_out)            # [B, num_classes]


# ══════════════════════════════════════════════════════════════════
# 4. 설정 클래스
# ══════════════════════════════════════════════════════════════════

class TimesNetConfig:
    """
    CIC-UNSW-NB15 분류를 위한 TimesNet 기본 설정.

    사용 예시:
        cfg = TimesNetConfig(d_model=256, e_layers=4)
        model = Model(cfg)
    """
    task_name   : str   = "classification"
    feature_dim : int   = 76
    num_classes : int   = 4
    seq_len     : int   = 1
    enc_in      : int   = 76
    d_model     : int   = 128
    d_ff        : int   = 256
    n_heads     : int   = 8
    e_layers    : int   = 3
    dropout     : float = 0.1
    use_norm    : bool  = True
    activation  : str   = "gelu"
    embed       : str   = "fixed"
    freq        : str   = "h"
    factor      : int   = 1

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ══════════════════════════════════════════════════════════════════
# 5. 지표 계산
# ══════════════════════════════════════════════════════════════════

def compute_metrics(model_name: str,
                    y_true: np.ndarray,
                    y_pred: np.ndarray,
                    class_names: list = None) -> dict:
    """
    Confusion Matrix 기반 지표 계산.

    반환 키:
        model, accuracy, f1_macro, f1_weighted,
        micro_FPR, micro_FNR, macro_FPR, macro_FNR,
        conf_matrix, class_FPR, class_FNR,
        class_precision, class_recall, class_f1
    """
    n_cls    = len(np.unique(np.concatenate([y_true, y_pred])))
    conf     = confusion_matrix(y_true, y_pred, labels=list(range(n_cls)))
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_cls)]

    class_FPR, class_FNR = [], []
    class_prec, class_rec, class_f1_sc = [], [], []

    for i in range(n_cls):
        TP = int(conf[i, i])
        FN = int(conf[i, :].sum() - TP)
        FP = int(conf[:, i].sum() - TP)
        TN = int(conf.sum() - (TP + FP + FN))

        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
        pre = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1  = (2 * pre * rec / (pre + rec)) if (pre + rec) > 0 else 0.0

        class_FPR.append(fpr)
        class_FNR.append(fnr)
        class_prec.append(pre)
        class_rec.append(rec)
        class_f1_sc.append(f1)

    FP_t = sum(conf[:, i].sum() - conf[i, i] for i in range(n_cls))
    FN_t = sum(conf[i, :].sum() - conf[i, i] for i in range(n_cls))
    TP_t = int(np.diag(conf).sum())
    TN_t = int(conf.sum()) - (int(FP_t) + int(FN_t) + TP_t)

    micro_FPR = FP_t / (FP_t + TN_t) if (FP_t + TN_t) > 0 else 0.0
    micro_FNR = FN_t / (FN_t + TP_t) if (FN_t + TP_t) > 0 else 0.0
    macro_FPR = float(np.mean(class_FPR))
    macro_FNR = float(np.mean(class_FNR))
    accuracy  = accuracy_score(y_true, y_pred)
    f1_macro  = f1_score(y_true, y_pred, average="macro",     zero_division=0)
    f1_weight = f1_score(y_true, y_pred, average="weighted",  zero_division=0)

    sep = "=" * 54
    print(f"\n{sep}")
    print(f"[{model_name}] 평가 결과")
    print(sep)
    print(f"  Accuracy       : {accuracy:.4f}")
    print(f"  F1  (macro)    : {f1_macro:.4f}")
    print(f"  F1  (weighted) : {f1_weight:.4f}")
    print(f"  오탐율 FPR (Micro) : {micro_FPR:.4f}")
    print(f"  미탐율 FNR (Micro) : {micro_FNR:.4f}")
    print(f"  오탐율 FPR (Macro) : {macro_FPR:.4f}")
    print(f"  미탐율 FNR (Macro) : {macro_FNR:.4f}")
    print(f"\n  Confusion Matrix:\n{conf}")
    print(f"\n  클래스별 FPR(오탐) / FNR(미탐):")
    for i, name in enumerate(class_names):
        print(f"    {name:20s} FPR={class_FPR[i]:.4f}  "
              f"FNR={class_FNR[i]:.4f}  "
              f"Prec={class_prec[i]:.4f}  Rec={class_rec[i]:.4f}")
    print(sep)

    return {
        "model"          : model_name,
        "accuracy"       : accuracy,
        "f1_macro"       : f1_macro,
        "f1_weighted"    : f1_weight,
        "micro_FPR"      : micro_FPR,
        "micro_FNR"      : micro_FNR,
        "macro_FPR"      : macro_FPR,
        "macro_FNR"      : macro_FNR,
        "conf_matrix"    : conf,
        "class_FPR"      : class_FPR,
        "class_FNR"      : class_FNR,
        "class_precision": class_prec,
        "class_recall"   : class_rec,
        "class_f1"       : class_f1_sc,
        "class_names"    : class_names,
    }


# ══════════════════════════════════════════════════════════════════
# 6. 학습 / 평가 함수
# ══════════════════════════════════════════════════════════════════

def run_timesnet(X: np.ndarray,
                 y: np.ndarray,
                 config: dict = None,
                 class_names: list = None) -> dict:
    """
    CIC-UNSW-NB15 TimesNet 분류 학습 및 평가.

    Args:
        X           : [N, 76]  float32 배열
        y           : [N]      int 레이블 (0~3)
        config      : 학습 설정 딕셔너리
        class_names : 클래스 이름 리스트 (None → ["Class 0", ...])

    config 키:
        test_size, random_state, d_model, d_ff, n_heads, e_layers,
        dropout, use_norm, activation, factor,
        lr, n_epochs, batch_size, weight_decay, patience

    Returns:
        {
            "model"  : 학습된 Model 인스턴스,
            "metrics": compute_metrics() 반환 딕셔너리,
            "config" : 사용된 TimesNetConfig,
            "history": {
                "train_loss": list,
                "train_acc" : list,
                "val_loss"  : list,
                "val_acc"   : list,
            }
        }
    """
    if config is None:
        config = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[TimesNet-CIC] 장치: {device}")

    # ── 하이퍼파라미터 ────────────────────────────
    test_size    = config.get("test_size",    0.2)
    random_state = config.get("random_state", 42)
    lr           = config.get("lr",           1e-3)
    n_epochs     = config.get("n_epochs",     30)
    batch_size   = config.get("batch_size",   256)
    weight_decay = config.get("weight_decay", 1e-4)
    patience     = config.get("patience",     7)
    num_classes  = int(y.max()) + 1

    # ── 데이터 분할 ────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = test_size,
        random_state = random_state,
        stratify     = y,
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"  클래스 수: {num_classes}")

    # ── DataLoader ─────────────────────────────────
    to_t = lambda arr, dt: torch.tensor(arr, dtype=dt)
    train_loader = DataLoader(
        TensorDataset(to_t(X_train, torch.float32),
                      to_t(y_train, torch.long)),
        batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        TensorDataset(to_t(X_test, torch.float32),
                      to_t(y_test, torch.long)),
        batch_size=batch_size, shuffle=False)

    # ── 모델 생성 ──────────────────────────────────
    cfg = TimesNetConfig(
        task_name   = "classification",
        feature_dim = X_train.shape[1],
        num_classes = num_classes,
        seq_len     = 1,
        enc_in      = X_train.shape[1],
        d_model     = config.get("d_model",    128),
        d_ff        = config.get("d_ff",       256),
        n_heads     = config.get("n_heads",    8),
        e_layers    = config.get("e_layers",   3),
        dropout     = config.get("dropout",    0.1),
        use_norm    = config.get("use_norm",   True),
        activation  = config.get("activation", "gelu"),
        factor      = config.get("factor",     1),
    )
    model = Model(cfg).to(device)

    param_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  파라미터 수: {param_cnt:,}")

    # ── 클래스 가중치 (불균형 대응) ───────────────
    class_counts = np.bincount(y_train, minlength=num_classes).astype(float)
    class_weight = torch.tensor(
        1.0 / (class_counts / class_counts.sum() + 1e-6),
        dtype=torch.float32).to(device)
    class_weight = class_weight / class_weight.sum() * num_classes

    # ── 학습 설정 ──────────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01)

    best_val_loss = float("inf")
    best_state    = None
    patience_cnt  = 0

    history = {"train_loss": [], "train_acc": [],
               "val_loss":   [], "val_acc":   []}

    # ── 학습 루프 ──────────────────────────────────
    print("\n[TimesNet-CIC] 학습 시작")
    print(f"  {'Epoch':>6} {'TrainLoss':>10} {'TrainAcc':>9} "
          f"{'ValLoss':>9} {'ValAcc':>9} {'LR':>10}")
    print("  " + "-" * 60)

    for epoch in range(1, n_epochs + 1):
        # Train
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            loss   = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss    += loss.item()
            t_correct += (logits.argmax(1) == y_b).sum().item()
            t_total   += len(y_b)
        scheduler.step()

        train_loss = t_loss / len(train_loader)
        train_acc  = t_correct / t_total

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for X_b, y_b in test_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits    = model(X_b)
                v_loss   += criterion(logits, y_b).item()
                v_correct += (logits.argmax(1) == y_b).sum().item()
                v_total   += len(y_b)

        val_loss = v_loss / len(test_loader)
        val_acc  = v_correct / v_total
        cur_lr   = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_loss)
        history["train_acc"] .append(train_acc)
        history["val_loss"]  .append(val_loss)
        history["val_acc"]   .append(val_acc)

        print(f"  [{epoch:>3}/{n_epochs}] "
              f"{train_loss:>10.4f} {train_acc:>9.4f} "
              f"{val_loss:>9.4f} {val_acc:>9.4f} {cur_lr:>10.2e}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone()
                             for k, v in model.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  [STOP] EarlyStopping at epoch {epoch} (patience={patience})")
                break

    # 최적 가중치 복원
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    # ── 최종 평가 ──────────────────────────────────
    model.eval()
    y_pred_all, y_true_all = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            preds = model(X_b.to(device)).argmax(1).cpu().numpy()
            y_pred_all.extend(preds)
            y_true_all.extend(y_b.numpy())

    metrics = compute_metrics(
        "TimesNet (CIC-UNSW-NB15)",
        np.array(y_true_all),
        np.array(y_pred_all),
        class_names=class_names,
    )

    return {
        "model"  : model,
        "metrics": metrics,
        "config" : cfg,
        "history": history,
    }
