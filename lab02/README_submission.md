# Lab 2 — 분류·회귀 재현 방법

20212788 최재원 · 머신러닝 프로젝트 (53744-01)

## 1. 환경

| 항목 | 값 |
|------|-----|
| Python | 3.10 이상 (제출자 실행 환경: 3.13.3) |
| 의존성 | `numpy`, `pandas`, `scikit-learn`, `pytest` |
| 하드웨어 | CPU만 사용 |
| 실행 시간 | 1초 미만 (측정값 0.11초) |
| 시드 | 42 (`src/lab02.py`의 `SEED`에 고정) |

```bash
pip install "numpy>=1.26,<3" "pandas>=2.1,<3" "scikit-learn>=1.4,<2" pytest
```

## 2. 데이터 배치

제출물에는 데이터셋이 포함되어 있지 않습니다(과제 규정). `src/lab02.py`는 파일 위치를
기준으로 데이터 경로를 계산하므로, 배포된 두 CSV를 아래 위치에 두어야 합니다.

```
<압축 해제 폴더>/
├── data/
│   ├── clinic_noshow.csv     <- lab02_classification 배포본에서 복사
│   └── clinic_wait.csv       <- 같음
├── src/
│   └── lab02.py
├── results.json
├── report.pdf
└── README.md
```

## 3. 결과 재현

압축 해제 폴더에서 실행합니다.

```bash
python src/lab02.py
```

`results.json`을 덮어쓰고 같은 내용을 표준 출력으로 인쇄합니다. 시드가 고정되어 있으므로
재실행해도 `runtime_seconds`를 제외한 모든 값이 동일합니다.

기대 출력 (`metrics` 부분):

```json
{
  "n_samples": 1200,
  "positive_rate": 0.3517,
  "baseline_accuracy": 0.65,
  "baseline_precision": 0.0,
  "baseline_recall": 0.0,
  "baseline_f1": 0.0,
  "logreg_accuracy": 0.6917,
  "logreg_precision": 0.5735,
  "logreg_recall": 0.4643,
  "logreg_f1": 0.5132,
  "knn_accuracy": 0.6917,
  "knn_precision": 0.5806,
  "knn_recall": 0.4286,
  "knn_f1": 0.4932,
  "n_samples_reg": 900,
  "wait_mean": 57.354,
  "reg_baseline_mae": 28.4714,
  "reg_baseline_rmse": 32.575,
  "reg_baseline_r2": -0.0056,
  "reg_linreg_mae": 4.514,
  "reg_linreg_rmse": 5.5935,
  "reg_linreg_r2": 0.9704,
  "reg_knn_mae": 6.9887,
  "reg_knn_rmse": 8.7285,
  "reg_knn_r2": 0.9278
}
```

## 4. 테스트

```bash
python -m pytest tests/ -q
```

배포된 공개 테스트 9개가 모두 통과합니다(`tests/`는 제출물에 포함되지 않으므로 배포본의
것을 사용하십시오).

## 5. 구현한 함수

| 함수 | 내용 |
|------|------|
| `split_data(X, y, test_size, seed)` | `y`로 층화한 재현 가능한 train/val 분할 |
| `compute_metrics(y_true, y_pred)` | 혼동행렬에서 accuracy/precision/recall/F1 (numpy만 사용) |
| `build_baseline()` | `DummyClassifier(strategy="most_frequent")` |
| `build_model(name)` | `StandardScaler` → 분류기 파이프라인 (`"scaler"`, `"clf"`) |
| `run_experiment(X, y)` | baseline / logreg / knn을 동일 조건에서 비교 |
| `compute_regression_metrics(y_true, y_pred)` | 잔차에서 MAE/RMSE/R² (numpy만 사용) |
| `build_regression_model(name)` | `DummyRegressor` 또는 `StandardScaler` → 회귀기 (`"scaler"`, `"reg"`) |
| `run_regression_experiment(X, y, test_size, seed)` | baseline / linreg / knn을 동일 조건에서 비교 |

누수를 막기 위해 분할 이전에는 아무것도 fit하지 않으며, 스케일러는 각 `Pipeline` 안에 있어
훈련 분할에서만 학습됩니다. 분할은 실험당 한 번만 계산해 세 모델이 공유합니다.

회귀 실험은 `split_data`를 재사용하지 않습니다. 타깃이 연속형이라 층화할 클래스가 없기
때문이며, `train_test_split`을 `stratify` 없이 직접 호출합니다.

## 6. 그림

`report.pdf`의 Figure 1~3은 배포된 `notebooks/lab02_visualization.ipynb`로 생성했습니다.
노트북은 제출물에 포함되지 않습니다.
