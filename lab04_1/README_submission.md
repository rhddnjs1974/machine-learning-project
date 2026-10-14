# Lab 4-1 — 경사하강법과 NumPy MLP 재현 방법

20212788 최재원 · 머신러닝 프로젝트 (53744-01)

## 1. 환경

| 항목 | 값 |
|------|-----|
| Python | 3.10 이상 (제출자 실행 환경: 3.13.3) |
| 의존성 | `numpy`, `pandas`, `pytest` (그림 생성에만 `matplotlib`, `jupyter`) |
| 하드웨어 | CPU만 사용 |
| 실행 시간 | 1분 미만 (측정값 2.35초) |
| 시드 | 42 (`src/lab04_1.py`의 `SEED`에 고정) |

`src/lab04_1.py`는 **numpy, pandas, 표준 라이브러리만** 사용합니다. `torch`,
`tensorflow`, `jax`, sklearn 모델/옵티마이저/지표는 import하지 않습니다.

```bash
pip install "numpy>=1.26,<3" "pandas>=2.1,<3" "pytest>=8"
```

## 2. 데이터 배치

제출물에는 데이터셋이 포함되어 있지 않습니다(과제 규정). `src/lab04_1.py`는 파일 위치를
기준으로 데이터 경로를 계산하므로, 배포된 `press_qc.csv`를 아래 위치에 두어야 합니다.

```
<압축 해제 폴더>/
├── data/
│   └── press_qc.csv       <- lab04_1_gradient_descent 배포본에서 복사
├── src/
│   └── lab04_1.py
├── results.json
├── report.pdf
└── README.md
```

## 3. 결과 재현

압축 해제 폴더에서 실행합니다.

```bash
python src/lab04_1.py
```

`results.json`을 덮어쓰고 같은 내용을 표준 출력으로 인쇄합니다. 초기 가중치는
`init_params(seed=42)`, 미니배치 셔플은 `np.random.default_rng(42)`에서 나오므로
재실행해도 `runtime_seconds`를 제외한 모든 값이 동일합니다.

기대 출력 (`metrics` 부분):

```json
{
  "n_samples": 800,
  "positive_rate": 0.2962,
  "majority_baseline_accuracy": 0.7038,
  "n_hidden": 8,
  "epochs": 400,
  "batch_size": 32,
  "grad_check_max_rel_error": 9e-08,
  "lr_0.01_final_train_loss": 0.6065,
  "lr_0.01_max_train_loss": 0.6853,
  "lr_0.01_final_val_loss": 0.6049,
  "lr_0.01_final_val_accuracy": 0.7063,
  "lr_0.01_best_val_accuracy": 0.7063,
  "lr_1.0_final_train_loss": 0.2822,
  "lr_1.0_max_train_loss": 0.6247,
  "lr_1.0_final_val_loss": 0.3608,
  "lr_1.0_final_val_accuracy": 0.8812,
  "lr_1.0_best_val_accuracy": 0.8938,
  "lr_100.0_final_train_loss": 8.203,
  "lr_100.0_max_train_loss": 19.4281,
  "lr_100.0_final_val_loss": 8.1166,
  "lr_100.0_final_val_accuracy": 0.7063,
  "lr_100.0_best_val_accuracy": 0.7063,
  "best_lr": 1.0
}
```

`lr = 100`은 발산하는 실행이므로 `final_train_loss`와 `final_val_loss`는 기계마다 다를 수
있습니다. 안정적인 값은 `max_train_loss`입니다.

## 4. 테스트

```bash
python -m pytest tests/ -q
```

배포된 공개 테스트 6개가 모두 통과합니다(`tests/`는 제출물에 포함되지 않으므로 배포본의
것을 사용하십시오).

## 5. 구현한 함수

| 함수 | 내용 |
|------|------|
| `sigmoid(z)` | `z`의 부호로 나누어 계산하는 수치적으로 안정한 로지스틱 함수 |
| `forward(X, params)` | `z1`, `a1`, `z2`, `a2`를 모두 담은 캐시 반환 |
| `bce_loss(y, y_hat, eps)` | `eps`로 클리핑한 평균 이진 교차엔트로피 |
| `backward(X, y, params, cache)` | 해석적 기울기. `1/n`은 `dz2`에 한 번만 들어감 |
| `gradient_check(X, y, params, eps)` | `backward()`를 호출해 중앙차분과 비교, 파라미터별 최대 상대오차 |
| `train(...)` | 에폭마다 재셔플하는 미니배치 경사하강, 에폭 단위 손실·정확도 기록 |
| `run_experiment(X, y)` | 분할·시드·초기값·에폭·배치를 고정하고 학습률만 바꾼 비교 |

구현상 유의한 점:

- `sigmoid`는 `z >= 0`에서 `1/(1+exp(-z))`를, `z < 0`에서 `exp(z)/(1+exp(z))`를 사용해
  지수의 인자가 항상 0 이하가 되므로 오버플로가 발생하지 않습니다.
- `gradient_check`는 기울기를 다시 유도하지 않고 `backward()`를 그대로 호출합니다.
  섭동은 `params[key].ravel()`이 돌려주는 뷰에 직접 쓰고 즉시 원값으로 되돌립니다.
- 검증셋은 측정에만 쓰이며 기울기 계산에는 들어가지 않습니다. 파라미터 갱신은
  `train`의 미니배치 슬라이스에서만 일어납니다.

## 6. 그림

`report.pdf`의 Figure 1~3은 배포된 `notebooks/lab04_1_visualization.ipynb`로 생성했습니다.
노트북은 제출물에 포함되지 않습니다.
