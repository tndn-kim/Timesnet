# TimesNet — CIC-UNSW-NB15 침입 탐지 분류

> **Self-Contained TimesNet** 기반 네트워크 침입 탐지 시스템  
> CIC-UNSW-NB15 데이터셋에서 4-클래스 분류를 수행합니다.

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **모델** | TimesNet (Inverted Embedding + Self-Attention Encoder) |
| **데이터셋** | CIC-UNSW-NB15 |
| **입력 피처** | 76개 |
| **분류 클래스** | 4개 (정상, 정보 수집 및 분석, 시스템 침해 및 악성 행위, 서비스 방해) |
| **외부 의존 레이어** | 없음 (완전 독립 구현) |
---

## 🗂️ 파일 구조

```
Timesnet2/
├── Timesnet.py      # 모델 정의 및 학습/평가 함수 (핵심 모듈)
├── run.py           # 실행 진입점 — 데이터 로드 → 전처리 → 학습 → 시각화
├── visualize.py     # 시각화 모듈 (5종 그래프 생성)
├── __init__.py      # 패키지 초기화
└── README.md
```

---

## 🏗️ 모델 아키텍처

```
입력 [B, 76]
    │
    ▼
DataEmbedding_inverted    # Inverted Embedding: [B,1,76] → [B,76,d_model]
    │                     # 각 피처를 독립 토큰으로 처리
    ▼
ClassificationEncoder     # Self-Attention Encoder (e_layers 개 스택)
    │                     # 76개 variate 토큰 간 상호 관계 모델링
    ▼
ClassificationHead        # Global Avg Pooling → LayerNorm → MLP
    │
    ▼
출력 [B, 4]  (logits)
```

### 기본 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `d_model` | 128 | 임베딩 차원 |
| `d_ff` | 256 | FFN 히든 차원 |
| `n_heads` | 8 | 멀티헤드 어텐션 헤드 수 |
| `e_layers` | 3 | 인코더 레이어 수 |
| `dropout` | 0.1 | 드롭아웃 비율 |
| `activation` | gelu | 활성화 함수 |
| `use_norm` | True | 입력 정규화 여부 |

---

## ⚙️ 설치 및 환경

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn
```

**권장 환경**

- Python 3.8+
- PyTorch 1.12+ (CUDA 선택 사항)

---

## 🚀 빠른 시작

### 1. 데이터 파일 없이 더미 데이터로 테스트

```bash
python run.py
```

데이터 파일이 없으면 자동으로 더미 데이터(5,000샘플, 76 피처, 4 클래스)를 생성하여 동작을 검증합니다.

### 2. 실제 데이터로 실행

`run.py` 상단 `DEFAULT_CONFIG`에서 경로를 수정합니다.

**방법 A — 피처/레이블 분리 CSV:**

```python
DEFAULT_CONFIG = {
    "data_path"  : "./data/features.csv",   # 76개 피처 컬럼
    "label_path" : "./data/labels.csv",     # 'label' 컬럼 (0~3 정수)
    "label_col"  : "label",
    ...
}
```

**방법 B — 통합 CSV (피처 + 레이블):**

```python
DEFAULT_CONFIG = {
    "data_path"  : "./data/cicunswnb15.csv",
    "label_path" : None,                    # None → 통합 CSV 모드
    "label_col"  : "label",
    ...
}
```

### 3. 코드에서 임포트하여 사용

```python
from Timesnet import run_timesnet
from visualize import visualize_all
import numpy as np

# 데이터 준비
X = np.load("features.npy")   # [N, 76] float32
y = np.load("labels.npy")     # [N]     int (0~3)

# 학습 및 평가
result = run_timesnet(
    X, y,
    config={
        "n_epochs"   : 50,
        "batch_size" : 256,
        "lr"         : 1e-3,
        "d_model"    : 256,
        "e_layers"   : 4,
    },
    class_names=["Normal", "DoS", "Probe", "R2L"],
)

# 시각화 저장
visualize_all(result, y,
              class_names=["Normal", "DoS", "Probe", "R2L"],
              save_dir="./output/viz")
```

---

## 🔧 학습 설정

`config` 딕셔너리로 모든 학습 옵션을 제어합니다.

| 키 | 기본값 | 설명 |
|----|--------|------|
| `test_size` | 0.2 | 테스트 분할 비율 |
| `random_state` | 42 | 랜덤 시드 |
| `n_epochs` | 30 | 최대 학습 에포크 |
| `batch_size` | 256 | 배치 크기 |
| `lr` | 1e-3 | 초기 학습률 (AdamW) |
| `weight_decay` | 1e-4 | L2 정규화 |
| `patience` | 7 | Early Stopping 인내 횟수 |
| `normalize` | True | MinMaxScaler [-1, 1] 정규화 |
| `drop_na` | True | 결측치 행 제거 |
| `save_dir` | `./output/timesnet_cic` | 그래프 저장 경로 |

### 학습 전략

- **옵티마이저**: AdamW
- **스케줄러**: CosineAnnealingLR (η_min = lr × 0.01)
- **손실 함수**: CrossEntropyLoss (클래스 불균형 대응 가중치 적용)
- **그래디언트 클리핑**: max_norm = 1.0
- **Early Stopping**: Validation Loss 기준

---

## 📊 평가 지표

학습 완료 후 다음 지표를 출력합니다.

| 지표 | 설명 |
|------|------|
| **Accuracy** | 전체 정확도 |
| **F1-Score (Macro)** | 클래스 균등 평균 F1 |
| **F1-Score (Weighted)** | 샘플 수 가중 F1 |
| **FPR (Micro/Macro)** | 오탐율 — 낮을수록 좋음 |
| **FNR (Micro/Macro)** | 미탐율 — 낮을수록 좋음 |
| **클래스별 FPR/FNR** | 클래스 단위 오탐/미탐 분석 |
| **Confusion Matrix** | 예측 vs 실제 분포 |

---

## 📈 시각화 출력

`visualize_all()` 호출 시 5종의 PNG가 생성됩니다.

| 파일명 | 내용 |
|--------|------|
| `training_curves.png` | 학습/검증 Loss 및 Accuracy 곡선 |
| `confusion_matrix.png` | Confusion Matrix 히트맵 (count + %) |
| `fpr_fnr_analysis.png` | 클래스별 오탐율(FPR) / 미탐율(FNR) 분석 |
| `metrics_summary.png` | 전체 성능 지표 대시보드 + 클래스별 P/R/F1 |
| `class_distribution.png` | 학습/테스트 클래스 분포 (막대 + 파이) |

---

## 📦 반환값 구조

`run_timesnet()` 함수는 다음 딕셔너리를 반환합니다.

```python
{
    "model"  : Model,           # 학습된 PyTorch 모델 인스턴스
    "metrics": {                # compute_metrics() 반환 딕셔너리
        "accuracy", "f1_macro", "f1_weighted",
        "micro_FPR", "micro_FNR", "macro_FPR", "macro_FNR",
        "conf_matrix",          # numpy 행렬
        "class_FPR", "class_FNR",
        "class_precision", "class_recall", "class_f1",
    },
    "config" : TimesNetConfig,  # 사용된 모델 설정
    "history": {                # 에포크별 학습 기록
        "train_loss", "train_acc",
        "val_loss",   "val_acc",
    },
}
```

---

## 🔬 데이터셋 요구사항

| 항목 | 조건 |
|------|------|
| 피처 수 | 76개 (수치형) |
| 레이블 | 정수 (0 ~ num_classes-1) |
| 파일 형식 | CSV |
| 전처리 | MinMaxScaler [-1, 1] (코드 내 자동 적용) |

> CIC-UNSW-NB15 데이터셋은 [공식 UNSW 사이트](https://research.unsw.edu.au/projects/unsw-nb15-dataset) 또는 [Kaggle](https://www.kaggle.com/search?q=UNSW-NB15)에서 확인할 수 있습니다.

---

## 📝 라이선스

본 코드는 학술 및 연구 목적으로 작성되었습니다.
