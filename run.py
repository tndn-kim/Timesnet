"""
Timesnet_CIC/run.py
===================
CIC-UNSW-NB15 TimesNet 단독 실행 진입점.

실행 방법:
    cd C:\\ai_exam\\3DGAN
    python Timesnet_CIC/run.py

또는 임포트 방식:
    from Timesnet_CIC.run import main
    main(DATA_PATH="...", LABEL_PATH="...", ...)

데이터 전제:
    - features.csv : 76개 컬럼 (숫자형, 정규화 전)
    - labels.csv   : 'label' 컬럼 (0~3 정수)
    또는 하나의 통합 CSV 에서 피처/레이블 분리 가능
"""

import os
import sys
import copy
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 경로 추가 (같은 폴더 내 모듈 임포트용)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from Timesnet import run_timesnet
from visualize import visualize_all


# ══════════════════════════════════════════════════════════════════
# 설정 (여기를 수정하여 실험)
# ══════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    # ── 데이터 경로 ──────────────────────────────────────────────
    # 방법 A : 피처 / 레이블 CSV 분리
    "data_path"     : "./data/features.csv",
    "label_path"    : "./data/labels.csv",
    "label_col"     : "label",
    # 방법 B : 통합 CSV (피처+레이블 한 파일)
    #   "data_path"  : "./data/cicunswnb15.csv"
    #   "label_col"  : "label"     # 레이블 컬럼명
    #   (label_path 를 None 으로 두면 통합 CSV 모드)
    # "label_path"  : None,

    # ── 클래스 이름 (None → Class 0~3 자동) ─────────────────────
    # CIC-UNSW-NB15 4-class 예시 (실제 매핑에 맞게 수정)
    "class_names"   : None,
    # "class_names" : ["Normal", "DoS", "Probe", "R2L"],

    # ── 전처리 ────────────────────────────────────────────────────
    "normalize"     : True,          # MinMaxScaler [-1, 1]
    "drop_na"       : True,          # 결측치 행 제거

    # ── 학습 ──────────────────────────────────────────────────────
    "test_size"     : 0.2,
    "random_state"  : 42,
    "n_epochs"      : 30,
    "batch_size"    : 256,
    "lr"            : 1e-3,
    "weight_decay"  : 1e-4,
    "patience"      : 7,

    # ── 모델 구조 ─────────────────────────────────────────────────
    "d_model"       : 128,
    "d_ff"          : 256,
    "n_heads"       : 8,
    "e_layers"      : 3,
    "dropout"       : 0.1,
    "use_norm"      : True,
    "activation"    : "gelu",
    "factor"        : 1,

    # ── 출력 ──────────────────────────────────────────────────────
    "save_dir"      : "./output/timesnet_cic",   # 시각화 저장 경로
}


# ══════════════════════════════════════════════════════════════════
# 데이터 로더
# ══════════════════════════════════════════════════════════════════

def _load_data(config: dict):
    """
    CSV 파일에서 피처 행렬 X, 레이블 벡터 y 를 로드.

    반환:
        X : np.ndarray  [N, 76]  float32
        y : np.ndarray  [N]      int64
    """
    data_path  = config["data_path"]
    label_path = config.get("label_path")
    label_col  = config.get("label_col", "label")
    drop_na    = config.get("drop_na", True)

    print(f"\n[데이터 로드]")

    if label_path is not None and os.path.exists(label_path):
        # 방법 A: 분리된 CSV
        X_df = pd.read_csv(data_path)
        y_s  = pd.read_csv(label_path)[label_col]
        if drop_na:
            mask = X_df.notna().all(axis=1) & y_s.notna()
            X_df = X_df[mask].reset_index(drop=True)
            y_s  = y_s[mask].reset_index(drop=True)
        print(f"  피처 CSV  : {data_path}  → {X_df.shape}")
        print(f"  레이블 CSV: {label_path} → {y_s.shape}")
        X = X_df.values.astype(np.float32)
        y = y_s.values.astype(np.int64)
    else:
        # 방법 B: 통합 CSV
        df = pd.read_csv(data_path)
        if drop_na:
            df = df.dropna().reset_index(drop=True)
        y_col = df[label_col].values.astype(np.int64)
        X_df  = df.drop(columns=[label_col])
        print(f"  통합 CSV  : {data_path}  → {df.shape}")
        print(f"  레이블 컬럼: {label_col}")
        X = X_df.values.astype(np.float32)
        y = y_col

    print(f"  피처 차원 : {X.shape[1]}")
    print(f"  샘플 수   : {len(X):,}")
    print(f"  클래스 분포:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"    Label {cls}: {cnt:,}  ({cnt/len(y)*100:.1f}%)")

    return X, y


# ══════════════════════════════════════════════════════════════════
# 메인 실행 함수
# ══════════════════════════════════════════════════════════════════

def main(config: dict = None, **kwargs) -> dict:
    """
    전체 파이프라인 실행: 데이터 로드 → 전처리 → 학습 → 평가 → 시각화.

    Args:
        config : 설정 딕셔너리 (None → DEFAULT_CONFIG 사용)
        **kwargs : config 개별 키 오버라이드

    Returns:
        {
            "result"    : run_timesnet() 반환값,
            "viz_paths" : 생성된 그래프 경로 딕셔너리,
        }
    """
    if config is None:
        config = copy.deepcopy(DEFAULT_CONFIG)
    for k, v in kwargs.items():
        config[k] = v

    print("=" * 60)
    print("TimesNet — CIC-UNSW-NB15 분류 실행")
    print("=" * 60)

    # ── 1. 데이터 로드 ─────────────────────────────────────────
    X, y = _load_data(config)

    # ── 2. 전처리 ──────────────────────────────────────────────
    if config.get("normalize", True):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X = scaler.fit_transform(X).astype(np.float32)
        print(f"\n[전처리] MinMaxScaler 정규화 완료 → [-1, 1]")

    # ── 3. 학습 / 평가 ─────────────────────────────────────────
    class_names = config.get("class_names", None)
    result = run_timesnet(X, y, config=config, class_names=class_names)

    # ── 4. 시각화 ──────────────────────────────────────────────
    save_dir = config.get("save_dir", "./output/timesnet_cic")
    viz_paths = visualize_all(
        result      = result,
        y           = y,
        class_names = class_names,
        save_dir    = save_dir,
    )

    # ── 5. 최종 요약 출력 ──────────────────────────────────────
    m = result["metrics"]
    print("\n" + "=" * 60)
    print("최종 성능 요약")
    print("=" * 60)
    print(f"  Accuracy         : {m['accuracy']:.4f}")
    print(f"  F1-Score (Macro) : {m['f1_macro']:.4f}")
    print(f"  F1-Score (Wtd)   : {m['f1_weighted']:.4f}")
    print(f"  오탐율 FPR (Micro): {m['micro_FPR']:.4f}")
    print(f"  미탐율 FNR (Micro): {m['micro_FNR']:.4f}")
    print(f"  오탐율 FPR (Macro): {m['macro_FPR']:.4f}")
    print(f"  미탐율 FNR (Macro): {m['macro_FNR']:.4f}")
    print(f"\n  그래프 저장 위치: {save_dir}/")
    for name, path in viz_paths.items():
        print(f"    {name:<25}: {os.path.basename(path)}")
    print("=" * 60)

    return {"result": result, "viz_paths": viz_paths}


# ══════════════════════════════════════════════════════════════════
# 직접 실행
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    사용 예시 (데이터 경로를 실제 경로로 수정 후 실행):

        python Timesnet_CIC/run.py

    분리된 CSV 사용 시 DEFAULT_CONFIG 에서 수정:
        "data_path"  : "path/to/features.csv"
        "label_path" : "path/to/labels.csv"

    통합 CSV 사용 시:
        "data_path"  : "path/to/all_data.csv"
        "label_path" : None
        "label_col"  : "label"  (또는 실제 컬럼명)
    """

    cfg = copy.deepcopy(DEFAULT_CONFIG)

    # ── 여기서 데이터 경로 설정 ─────────────────────
    # cfg["data_path"]  = "./data/features.csv"
    # cfg["label_path"] = "./data/labels.csv"
    # cfg["class_names"] = ["Normal", "DoS", "Probe", "R2L"]
    # cfg["n_epochs"] = 50

    # ── 데이터 파일이 없는 경우 더미 데이터로 동작 확인 ──
    if not os.path.exists(cfg["data_path"]):
        print("\n[!] 데이터 파일을 찾을 수 없습니다.")
        print("   더미 데이터 (N=5,000, 76 피처, 4 클래스)로 동작 테스트를 진행합니다.")
        print("   실제 사용 시 run.py 상단 DEFAULT_CONFIG 의 data_path 를 수정하세요.\n")

        np.random.seed(42)
        N = 5_000
        X_dummy = np.random.randn(N, 76).astype(np.float32)
        y_dummy = np.random.randint(0, 4, size=N).astype(np.int64)

        # 클래스별 특성 부여 (분류가 가능하도록 신호 추가)
        for c in range(4):
            idx = y_dummy == c
            X_dummy[idx, c * 19 : c * 19 + 19] += (c + 1) * 1.5

        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_dummy = scaler.fit_transform(X_dummy).astype(np.float32)

        dummy_config = copy.deepcopy(cfg)
        dummy_config["n_epochs"]    = 15
        dummy_config["batch_size"]  = 128
        dummy_config["save_dir"]    = "./output/timesnet_cic_test"
        dummy_config["class_names"] = ["Normal", "DoS", "Probe", "R2L"]

        result = run_timesnet(X_dummy, y_dummy,
                              config=dummy_config,
                              class_names=dummy_config["class_names"])
        viz_paths = visualize_all(
            result      = result,
            y           = y_dummy,
            class_names = dummy_config["class_names"],
            save_dir    = dummy_config["save_dir"],
        )

        m = result["metrics"]
        print("\n" + "=" * 60)
        print("[더미 테스트] 최종 성능 요약")
        print("=" * 60)
        print(f"  Accuracy         : {m['accuracy']:.4f}")
        print(f"  F1-Score (Macro) : {m['f1_macro']:.4f}")
        print(f"  오탐율 FPR (Micro): {m['micro_FPR']:.4f}")
        print(f"  미탐율 FNR (Micro): {m['micro_FNR']:.4f}")
        print(f"\n  그래프 저장: {dummy_config['save_dir']}/")
        print("=" * 60)
    else:
        main(cfg)
