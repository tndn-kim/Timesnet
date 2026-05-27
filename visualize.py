"""
Timesnet_CIC/visualize.py
=========================
CIC-UNSW-NB15 TimesNet 시각화 모듈.

생성 그래프:
  1. training_curves.png   : 학습/검증 Loss·Accuracy 곡선
  2. confusion_matrix.png  : Confusion Matrix 히트맵 (count + %)
  3. fpr_fnr_analysis.png  : 클래스별 오탐율(FPR) / 미탐율(FNR) 분석
  4. metrics_summary.png   : 전체 성능 지표 대시보드
  5. class_distribution.png: 클래스 분포 (학습/테스트)

사용법:
    from visualize import visualize_all
    visualize_all(result, y, class_names=["Normal","DoS","Probe","R2L"],
                  save_dir="./output/viz")
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


# ── 공통 팔레트 ─────────────────────────────────────
COLORS = {
    "train"  : "#4A90D9",   # 파랑
    "val"    : "#E87040",   # 주황
    "fpr"    : "#E74C3C",   # 빨강 (오탐)
    "fnr"    : "#3498DB",   # 파랑 (미탐)
    "good"   : "#2ECC71",   # 초록
    "neutral": "#95A5A6",   # 회색
}

# 클래스별 팔레트
CLASS_PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _set_style():
    plt.rcParams.update({
        "font.family"       : "DejaVu Sans",
        "axes.spines.top"   : False,
        "axes.spines.right" : False,
        "axes.grid"         : True,
        "grid.alpha"        : 0.3,
        "grid.linestyle"    : "--",
        "figure.dpi"        : 120,
        "savefig.dpi"       : 150,
        "savefig.bbox_inches": "tight",
    })


# ══════════════════════════════════════════════════════════════════
# 1. 학습 곡선 (Loss & Accuracy)
# ══════════════════════════════════════════════════════════════════

def plot_training_curves(history: dict, save_dir: str) -> str:
    """
    학습/검증 Loss 및 Accuracy 곡선 저장.

    Args:
        history  : run_timesnet() 반환의 "history" 키
        save_dir : 저장 경로

    Returns:
        저장된 파일 경로
    """
    _set_style()
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("TimesNet (CIC-UNSW-NB15) — 학습 곡선",
                 fontsize=14, fontweight="bold", y=1.02)

    # ── Loss ──────────────────────────────────────
    ax_loss.plot(epochs, history["train_loss"], color=COLORS["train"],
                 linewidth=2, label="Train Loss", marker="o",
                 markersize=3, markevery=max(1, len(epochs)//20))
    ax_loss.plot(epochs, history["val_loss"],   color=COLORS["val"],
                 linewidth=2, label="Val Loss",   marker="s",
                 markersize=3, markevery=max(1, len(epochs)//20))

    best_ep = int(np.argmin(history["val_loss"])) + 1
    best_vl = min(history["val_loss"])
    ax_loss.axvline(best_ep, color="grey", linestyle=":", linewidth=1.2,
                    label=f"Best epoch={best_ep}")
    ax_loss.scatter([best_ep], [best_vl],
                    color=COLORS["val"], s=80, zorder=5)
    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("Loss",  fontsize=11)
    ax_loss.set_title("Loss 곡선", fontsize=12, fontweight="bold")
    ax_loss.legend(fontsize=9)

    # ── Accuracy ───────────────────────────────────
    ax_acc.plot(epochs, history["train_acc"], color=COLORS["train"],
                linewidth=2, label="Train Acc", marker="o",
                markersize=3, markevery=max(1, len(epochs)//20))
    ax_acc.plot(epochs, history["val_acc"],   color=COLORS["val"],
                linewidth=2, label="Val Acc",   marker="s",
                markersize=3, markevery=max(1, len(epochs)//20))

    best_ep_a = int(np.argmax(history["val_acc"])) + 1
    best_va   = max(history["val_acc"])
    ax_acc.axvline(best_ep_a, color="grey", linestyle=":", linewidth=1.2,
                   label=f"Best epoch={best_ep_a}")
    ax_acc.scatter([best_ep_a], [best_va],
                   color=COLORS["val"], s=80, zorder=5)
    ax_acc.annotate(f"{best_va:.4f}",
                    xy=(best_ep_a, best_va),
                    xytext=(best_ep_a + max(1, len(epochs)//15), best_va - 0.02),
                    fontsize=9, color=COLORS["val"], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=COLORS["val"]))
    ax_acc.set_xlabel("Epoch", fontsize=11)
    ax_acc.set_ylabel("Accuracy", fontsize=11)
    ax_acc.set_title("Accuracy 곡선", fontsize=12, fontweight="bold")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.legend(fontsize=9)

    plt.tight_layout()
    fpath = os.path.join(_ensure_dir(save_dir), "training_curves.png")
    plt.savefig(fpath)
    plt.close()
    print(f"  [OK] 저장: {fpath}")
    return fpath


# ══════════════════════════════════════════════════════════════════
# 2. Confusion Matrix 히트맵
# ══════════════════════════════════════════════════════════════════

def plot_confusion_matrix(metrics: dict, save_dir: str) -> str:
    """
    정규화된 Confusion Matrix 히트맵 (count + %).

    Args:
        metrics  : compute_metrics() 반환 딕셔너리
        save_dir : 저장 경로
    """
    _set_style()
    conf        = metrics["conf_matrix"]
    class_names = metrics.get("class_names",
                              [f"Class {i}" for i in range(conf.shape[0])])
    n           = conf.shape[0]
    conf_norm   = conf.astype(float) / conf.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(max(6, 2.2 * n), max(5, 2 * n)))

    # 커스텀 컬러맵 (흰색 → 짙은 파랑)
    cmap = LinearSegmentedColormap.from_list(
        "cic_blue", ["#FFFFFF", "#1A5276"], N=256)

    im = ax.imshow(conf_norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="비율")

    for i in range(n):
        for j in range(n):
            cnt  = conf[i, j]
            rat  = conf_norm[i, j]
            c    = "white" if rat > 0.55 else "black"
            ax.text(j, i, f"{cnt:,}\n({rat:.1%})",
                    ha="center", va="center",
                    fontsize=9.5, color=c, fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10, rotation=0)
    ax.set_xlabel("예측 레이블 (Predicted)", fontsize=11)
    ax.set_ylabel("실제 레이블 (True)",      fontsize=11)

    acc = metrics["accuracy"]
    f1  = metrics["f1_macro"]
    ax.set_title(
        f"Confusion Matrix — Acc {acc:.4f}  F1(macro) {f1:.4f}",
        fontsize=13, fontweight="bold", pad=14)

    plt.tight_layout()
    fpath = os.path.join(_ensure_dir(save_dir), "confusion_matrix.png")
    plt.savefig(fpath)
    plt.close()
    print(f"  [OK] 저장: {fpath}")
    return fpath


# ══════════════════════════════════════════════════════════════════
# 3. FPR / FNR 분석
# ══════════════════════════════════════════════════════════════════

def plot_fpr_fnr(metrics: dict, save_dir: str) -> str:
    """
    클래스별 FPR(오탐율) / FNR(미탐율) 막대 차트 + 전체 요약.

    레이아웃:
        행 1 : 클래스별 FPR 막대  |  클래스별 FNR 막대
        행 2 : 오탐/미탐 Micro·Macro 요약 (가로 막대)
    """
    _set_style()
    class_names = metrics.get("class_names",
                              [f"Class {i}" for i in range(len(metrics["class_FPR"]))])
    class_FPR = metrics["class_FPR"]
    class_FNR = metrics["class_FNR"]
    n_cls     = len(class_names)
    x         = np.arange(n_cls)

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                             hspace=0.45, wspace=0.35)

    # ── (0,0) 클래스별 FPR ───────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    bars = ax0.bar(x, class_FPR,
                   color=[CLASS_PALETTE[i % 4] for i in range(n_cls)],
                   edgecolor="white", alpha=0.88, width=0.55)
    for bar, v in zip(bars, class_FPR):
        ax0.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=9,
                 fontweight="bold", color="dimgray")
    ax0.axhline(metrics["micro_FPR"], color=COLORS["fpr"], linewidth=1.5,
                linestyle="--", label=f"Micro FPR={metrics['micro_FPR']:.4f}")
    ax0.axhline(metrics["macro_FPR"], color="darkorange", linewidth=1.5,
                linestyle="-.", label=f"Macro FPR={metrics['macro_FPR']:.4f}")
    ax0.set_xticks(x)
    ax0.set_xticklabels(class_names, fontsize=9)
    ax0.set_ylabel("FPR (오탐율)", fontsize=10)
    ax0.set_title("클래스별 오탐율 (FPR)", fontsize=11, fontweight="bold")
    ax0.set_ylim(0, min(1.1, max(class_FPR + [metrics["micro_FPR"]]) * 1.3 + 0.05))
    ax0.legend(fontsize=8)

    # ── (0,1) 클래스별 FNR ───────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    bars = ax1.bar(x, class_FNR,
                   color=[CLASS_PALETTE[i % 4] for i in range(n_cls)],
                   edgecolor="white", alpha=0.88, width=0.55)
    for bar, v in zip(bars, class_FNR):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=9,
                 fontweight="bold", color="dimgray")
    ax1.axhline(metrics["micro_FNR"], color=COLORS["fnr"], linewidth=1.5,
                linestyle="--", label=f"Micro FNR={metrics['micro_FNR']:.4f}")
    ax1.axhline(metrics["macro_FNR"], color="steelblue", linewidth=1.5,
                linestyle="-.", label=f"Macro FNR={metrics['macro_FNR']:.4f}")
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, fontsize=9)
    ax1.set_ylabel("FNR (미탐율)", fontsize=10)
    ax1.set_title("클래스별 미탐율 (FNR)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, min(1.1, max(class_FNR + [metrics["micro_FNR"]]) * 1.3 + 0.05))
    ax1.legend(fontsize=8)

    # ── (1, :) Micro/Macro 요약 (가로 막대) ──────────
    ax2 = fig.add_subplot(gs[1, :])
    labels_all = (
        ["Micro FPR\n(오탐율)", "Micro FNR\n(미탐율)",
         "Macro FPR\n(오탐율)", "Macro FNR\n(미탐율)"]
        + [f"FPR\n{n}" for n in class_names]
        + [f"FNR\n{n}" for n in class_names]
    )
    values_all = (
        [metrics["micro_FPR"], metrics["micro_FNR"],
         metrics["macro_FPR"], metrics["macro_FNR"]]
        + class_FPR + class_FNR
    )
    colors_all = (
        [COLORS["fpr"], COLORS["fnr"], "darkorange", "steelblue"]
        + [COLORS["fpr"]] * n_cls
        + [COLORS["fnr"]] * n_cls
    )
    y_pos = np.arange(len(labels_all))
    hbars = ax2.barh(y_pos, values_all, color=colors_all,
                     edgecolor="white", alpha=0.85, height=0.6)
    for bar, v in zip(hbars, values_all):
        ax2.text(v + 0.003, bar.get_y() + bar.get_height() / 2,
                 f"{v:.4f}", va="center", fontsize=8.5, fontweight="bold")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels_all, fontsize=8.5)
    ax2.set_xlabel("비율 (낮을수록 좋음)", fontsize=10)
    ax2.set_title("오탐율·미탐율 전체 요약", fontsize=11, fontweight="bold")
    ax2.set_xlim(0, min(1.2, max(values_all) * 1.25 + 0.05))
    ax2.invert_yaxis()

    # 범례 패치
    patch_fpr = mpatches.Patch(color=COLORS["fpr"],    label="FPR (오탐율)")
    patch_fnr = mpatches.Patch(color=COLORS["fnr"],    label="FNR (미탐율)")
    patch_mac = mpatches.Patch(color="darkorange",     label="Macro FPR")
    patch_mns = mpatches.Patch(color="steelblue",      label="Macro FNR")
    ax2.legend(handles=[patch_fpr, patch_fnr, patch_mac, patch_mns],
               loc="lower right", fontsize=8.5)

    fig.suptitle("TimesNet (CIC-UNSW-NB15) — 오탐율 / 미탐율 분석",
                 fontsize=14, fontweight="bold", y=1.01)

    fpath = os.path.join(_ensure_dir(save_dir), "fpr_fnr_analysis.png")
    plt.savefig(fpath)
    plt.close()
    print(f"  [OK] 저장: {fpath}")
    return fpath


# ══════════════════════════════════════════════════════════════════
# 4. 성능 지표 대시보드
# ══════════════════════════════════════════════════════════════════

def plot_metrics_summary(metrics: dict, save_dir: str) -> str:
    """
    전체 성능 지표 막대 차트 + 클래스별 Precision/Recall/F1 테이블.

    레이아웃:
        위 : 전체 지표 막대 (Accuracy, F1-macro, F1-weighted, FPR, FNR)
        아래 : 클래스별 Precision / Recall / F1 그룹 막대
    """
    _set_style()
    class_names = metrics.get("class_names",
                              [f"Class {i}" for i in range(len(metrics["class_FPR"]))])

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(13, 11))
    fig.suptitle("TimesNet (CIC-UNSW-NB15) — 성능 지표 요약",
                 fontsize=14, fontweight="bold")

    # ── 상단: 전체 지표 막대 ─────────────────────────
    overall_labels = ["Accuracy", "F1\n(Macro)", "F1\n(Weighted)",
                      "FPR\n(Micro·오탐↓)", "FNR\n(Micro·미탐↓)",
                      "FPR\n(Macro·오탐↓)", "FNR\n(Macro·미탐↓)"]
    overall_vals   = [
        metrics["accuracy"],
        metrics["f1_macro"],
        metrics["f1_weighted"],
        metrics["micro_FPR"],
        metrics["micro_FNR"],
        metrics["macro_FPR"],
        metrics["macro_FNR"],
    ]
    bar_colors = [
        COLORS["good"],   # Accuracy
        COLORS["good"],   # F1 macro
        COLORS["good"],   # F1 weighted
        COLORS["fpr"],    # FPR micro
        COLORS["fnr"],    # FNR micro
        "darkorange",     # FPR macro
        "steelblue",      # FNR macro
    ]
    x_top = np.arange(len(overall_labels))
    bars  = ax_top.bar(x_top, overall_vals, color=bar_colors,
                       edgecolor="white", alpha=0.88, width=0.55)
    for bar, v in zip(bars, overall_vals):
        y_offset = 0.012
        ax_top.text(bar.get_x() + bar.get_width() / 2,
                    v + y_offset,
                    f"{v:.4f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="dimgray")
    ax_top.set_xticks(x_top)
    ax_top.set_xticklabels(overall_labels, fontsize=10)
    ax_top.set_ylim(0, 1.15)
    ax_top.set_ylabel("Score / Rate", fontsize=11)
    ax_top.set_title("전체 성능 지표  (↑ 높을수록 좋음 / FPR·FNR ↓ 낮을수록 좋음)",
                     fontsize=11, fontweight="bold", pad=10)
    ax_top.axhline(1.0, color="grey", linewidth=0.6, linestyle=":")

    # 화살표 범례
    ax_top.annotate("↑ 높을수록 좋음", xy=(0.01, 0.92), xycoords="axes fraction",
                    fontsize=8.5, color=COLORS["good"])
    ax_top.annotate("↓ 낮을수록 좋음", xy=(0.52, 0.92), xycoords="axes fraction",
                    fontsize=8.5, color=COLORS["fpr"])

    # ── 하단: 클래스별 P/R/F1 ────────────────────────
    n_cls = len(class_names)
    prec  = metrics.get("class_precision", [0] * n_cls)
    rec   = metrics.get("class_recall",    [0] * n_cls)
    f1_c  = metrics.get("class_f1",        [0] * n_cls)

    x_bot = np.arange(n_cls)
    w     = 0.26
    b1 = ax_bot.bar(x_bot - w,   prec, w, label="Precision",
                    color="#2980B9", edgecolor="white", alpha=0.85)
    b2 = ax_bot.bar(x_bot,       rec,  w, label="Recall",
                    color="#27AE60", edgecolor="white", alpha=0.85)
    b3 = ax_bot.bar(x_bot + w,   f1_c, w, label="F1-Score",
                    color="#8E44AD", edgecolor="white", alpha=0.85)

    for group_bars in [b1, b2, b3]:
        for bar in group_bars:
            h = bar.get_height()
            ax_bot.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.3f}", ha="center", va="bottom",
                        fontsize=8, color="dimgray")

    ax_bot.set_xticks(x_bot)
    ax_bot.set_xticklabels(class_names, fontsize=10)
    ax_bot.set_ylim(0, 1.15)
    ax_bot.set_ylabel("Score", fontsize=11)
    ax_bot.set_title("클래스별 Precision / Recall / F1-Score",
                     fontsize=11, fontweight="bold", pad=10)
    ax_bot.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    fpath = os.path.join(_ensure_dir(save_dir), "metrics_summary.png")
    plt.savefig(fpath)
    plt.close()
    print(f"  [OK] 저장: {fpath}")
    return fpath


# ══════════════════════════════════════════════════════════════════
# 5. 클래스 분포
# ══════════════════════════════════════════════════════════════════

def plot_class_distribution(y_train: np.ndarray,
                             y_test: np.ndarray,
                             class_names: list,
                             save_dir: str) -> str:
    """
    학습/테스트 클래스 분포 시각화 (막대 + 파이).

    Args:
        y_train     : 학습 레이블
        y_test      : 테스트 레이블
        class_names : 클래스 이름
        save_dir    : 저장 경로
    """
    _set_style()
    n_cls = len(class_names)

    train_counts = np.bincount(y_train, minlength=n_cls)
    test_counts  = np.bincount(y_test,  minlength=n_cls)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("CIC-UNSW-NB15 클래스 분포",
                 fontsize=14, fontweight="bold")

    x   = np.arange(n_cls)
    w   = 0.38
    clrs = CLASS_PALETTE[:n_cls]

    # ── 막대: 학습 vs 테스트 count ──────────────────
    ax = axes[0]
    ax.bar(x - w/2, train_counts, w, label="Train",
           color=COLORS["train"], alpha=0.85, edgecolor="white")
    ax.bar(x + w/2, test_counts,  w, label="Test",
           color=COLORS["val"],   alpha=0.85, edgecolor="white")
    for xi, (tc, vc) in enumerate(zip(train_counts, test_counts)):
        ax.text(xi - w/2, tc + max(train_counts)*0.01,
                f"{tc:,}", ha="center", va="bottom", fontsize=8.5)
        ax.text(xi + w/2, vc + max(train_counts)*0.01,
                f"{vc:,}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=9)
    ax.set_ylabel("샘플 수", fontsize=10)
    ax.set_title("샘플 수 비교 (Train / Test)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    # ── 파이: 학습 분포 ──────────────────────────────
    ax = axes[1]
    pct = train_counts / train_counts.sum() * 100
    wedges, texts, autotexts = ax.pie(
        train_counts, labels=None,
        colors=clrs,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.80,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax.legend(wedges, [f"{n} ({c:,})" for n, c in zip(class_names, train_counts)],
              loc="lower center", bbox_to_anchor=(0.5, -0.18),
              fontsize=8.5, ncol=2)
    ax.set_title("Train 클래스 분포", fontsize=11, fontweight="bold")

    # ── 파이: 테스트 분포 ───────────────────────────
    ax = axes[2]
    wedges, texts, autotexts = ax.pie(
        test_counts, labels=None,
        colors=clrs,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.80,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax.legend(wedges, [f"{n} ({c:,})" for n, c in zip(class_names, test_counts)],
              loc="lower center", bbox_to_anchor=(0.5, -0.18),
              fontsize=8.5, ncol=2)
    ax.set_title("Test 클래스 분포", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fpath = os.path.join(_ensure_dir(save_dir), "class_distribution.png")
    plt.savefig(fpath)
    plt.close()
    print(f"  [OK] 저장: {fpath}")
    return fpath


# ══════════════════════════════════════════════════════════════════
# 6. 통합 실행
# ══════════════════════════════════════════════════════════════════

def visualize_all(result: dict,
                  y: np.ndarray,
                  class_names: list = None,
                  save_dir: str = "./output/viz") -> dict:
    """
    run_timesnet() 결과를 받아 모든 그래프를 생성·저장.

    Args:
        result      : run_timesnet() 반환 딕셔너리
        y           : 전체 레이블 배열 (분포 시각화용)
        class_names : 클래스 이름 리스트
                      None → ["Class 0", ..., "Class n-1"]
        save_dir    : PNG 저장 디렉터리

    Returns:
        저장된 파일 경로 딕셔너리
    """
    metrics  = result["metrics"]
    history  = result["history"]
    n_cls    = len(metrics["class_FPR"])

    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_cls)]
    metrics["class_names"] = class_names   # 이름 주입

    print("\n[시각화] 그래프 생성 시작")
    paths = {}

    # 1. 학습 곡선
    paths["training_curves"] = plot_training_curves(history, save_dir)

    # 2. Confusion Matrix
    paths["confusion_matrix"] = plot_confusion_matrix(metrics, save_dir)

    # 3. FPR / FNR 분석
    paths["fpr_fnr"] = plot_fpr_fnr(metrics, save_dir)

    # 4. 전체 지표 요약
    paths["metrics_summary"] = plot_metrics_summary(metrics, save_dir)

    # 5. 클래스 분포 (y 배열에서 train/test split 재현)
    from sklearn.model_selection import train_test_split
    # TimesNetConfig 에는 test_size 가 없으므로 0.2 를 기본값으로 사용
    ts = getattr(result["config"], "test_size", 0.2)
    y_train_d, y_test_d = train_test_split(
        y, test_size=ts, random_state=42, stratify=y)
    paths["class_distribution"] = plot_class_distribution(
        y_train_d, y_test_d, class_names, save_dir)

    print(f"\n[완료] 시각화 완료 → {save_dir}/")
    print(f"   training_curves.png    : 학습/검증 Loss·Acc 곡선")
    print(f"   confusion_matrix.png   : Confusion Matrix 히트맵")
    print(f"   fpr_fnr_analysis.png   : 오탐율·미탐율 클래스별 분석")
    print(f"   metrics_summary.png    : 성능 지표 대시보드")
    print(f"   class_distribution.png : 클래스 분포")

    return paths
