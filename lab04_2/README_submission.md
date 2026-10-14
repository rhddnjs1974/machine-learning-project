# Lab 4-2 — PyTorch 학습과 하이퍼파라미터 튜닝 재현 방법

20212788 최재원 · 머신러닝 프로젝트 (53744-01)

## 1. 환경

| 항목 | 값 |
|------|-----|
| Python | 3.10 이상 (제출자 실행 환경: 3.13.3) |
| 의존성 | `numpy`, `pandas`, `scikit-learn`, `torch`, `pytest` (그림 생성에만 `matplotlib`, `jupyter`) |
| 하드웨어 | CPU만 사용 (GPU·CUDA 불필요) |
| 실행 시간 | 여섯 설정 전체 약 15초 (측정값 14.85초) |
| 시드 | 42 (`src/lab04_2.py`의 `SEED`에 고정) |

```bash
pip install -r requirements.txt
```

배포된 `requirements.txt`는 `numpy>=1.26,<2`를 고정합니다. 제출자는 `numpy 2.2.4`와
`torch 2.11.0+cpu` 조합에서 작성·검증했으며 배열과 텐서 변환에 문제가 없었습니다.
`requirements.txt`가 고정한 `numpy 1.x`에서도 동일하게 동작합니다. 코드에 버전에 의존하는
부분은 없습니다.

## 2. 데이터 배치

제출물에는 데이터셋이 포함되어 있지 않습니다(과제 규정). `src/lab04_2.py`는 파일 위치를
기준으로 데이터 경로를 계산하므로, 배포된 `motor_health.csv`를 아래 위치에 두어야 합니다.

```
<압축 해제 폴더>/
├── data/
│   └── motor_health.csv     <- lab04_2_pytorch 배포본에서 복사
├── src/
│   └── lab04_2.py
├── results.json
├── report.pdf
└── README.md
```

## 3. 결과 재현

압축 해제 폴더에서 실행합니다.

```bash
python src/lab04_2.py
```

`results.json`을 덮어쓰고 같은 내용을 표준 출력으로 인쇄합니다. `set_seed()`가 python,
numpy, torch의 난수를 한 번에 고정하고 설정마다 다시 호출되므로, 재실행해도
`runtime_seconds`를 제외한 모든 값이 동일합니다.

기대 출력 (`grid`를 제외한 `metrics` 부분):

```json
{
  "n_train": 1200,
  "n_val": 400,
  "n_test": 400,
  "positive_rate": 0.299,
  "n_configs": 6,
  "best_hidden_size": 32,
  "best_lr": 0.1,
  "val_loss": 0.335632,
  "val_accuracy": 0.855,
  "test_loss": 0.370398,
  "test_accuracy": 0.8475
}
```

테스트 정확도 **0.8475**는 "항상 정비 불필요"라고 답하는 기준선 0.7025를 넘습니다.

## 4. 테스트

```bash
python -m pytest tests/ -q
```

배포된 공개 테스트 6개가 모두 통과합니다(`tests/`는 제출물에 포함되지 않으므로 배포본의
것을 사용하십시오). 그리드 탐색이 포함되어 있어 40초 안팎이 걸립니다.

## 5. 구현한 함수

| 함수 | 내용 |
|------|------|
| `build_model(input_dim, hidden_size)` | Linear → ReLU → Linear → ReLU → Linear 다섯 모듈, 출력은 로짓 1개 |
| `train_one_epoch(model, loader, loss_fn, optimizer)` | 학습 모드로 한 에폭. `zero_grad` → forward → `backward` → `step` 순서 |
| `evaluate(model, loader, loss_fn)` | 평가 모드 + `torch.no_grad()`. 손실과 정확도를 샘플 단위로 평균 |
| `train_model(model, train_loader, val_loader, lr, ...)` | 에폭 루프, 검증 손실 기준 조기 종료, 최적 가중치 복원 |
| `run_experiment(data)` | 은닉 크기 × 학습률 그리드, 검증으로 선택, 테스트는 한 번만 |

구현상 유의한 점:

- **최적 가중치 복원은 루프가 끝난 뒤에** 합니다. 검증 손실이 갱신될 때마다
  `copy.deepcopy(model.state_dict())`로 저장해 두었다가, 루프 종료 후
  `model.load_state_dict(...)`로 되돌립니다. 따라서 함수가 반환될 때 모델은 마지막 에폭이
  아니라 최적 에폭의 가중치를 들고 있습니다.
- **설정마다 `set_seed()`를 먼저 호출**하므로, 같은 은닉 크기에서 비교하는 세 학습률이
  동일한 초기 가중치와 동일한 배치 순서에서 출발합니다. 바뀌는 것은 학습률 하나뿐입니다.
- **테스트 로더는 정확히 한 번 평가**됩니다. 그리드 여섯 설정의 모델을 모두 보관했다가,
  검증 손실이 가장 낮은 설정이 정해진 뒤에만 `evaluate`를 호출합니다.
- `evaluate`는 `model.eval()`과 `torch.no_grad()` 안에서만 돌며 `backward`나 `step`을
  호출하지 않습니다. 검증·테스트 데이터는 측정에만 쓰입니다.

## 6. 그림

`report.pdf`의 Figure 1~3은 배포된 `notebooks/lab04_2_visualization.ipynb`로 생성했습니다.
노트북은 제출물에 포함되지 않습니다.
