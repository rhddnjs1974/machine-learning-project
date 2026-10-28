# Lab 6 — RNN 시계열 예측 재현 방법

20212788 최재원 · 머신러닝 프로젝트 (53744-01)

## 1. 환경

| 항목 | 값 |
|------|-----|
| Python | 3.10 이상 (제출자 실행 환경: 3.13.3) |
| 의존성 | `numpy`, `pandas`, `torch`, `pytest` (그림 생성에만 `matplotlib`, `jupyter`) |
| 하드웨어 | CPU만 사용 (GPU·CUDA 불필요) |
| 메모리 | 1 GB 미만 |
| 실행 시간 | `python src/lab06.py` 약 12초, `pytest tests/` 약 32초 |
| 시드 | 42 (`src/lab06.py`의 `SEED`에 고정) |

```bash
pip install -r requirements.txt
```

배포된 `requirements.txt`는 `numpy<2`를 고정합니다. 제출자는 `numpy 2.2.4`와
`torch 2.11.0+cpu` 조합에서 작성·검증했으며 배열과 텐서 변환에 문제가 없었습니다. 코드에
버전에 의존하는 부분은 없으므로 `requirements.txt`가 고정한 `numpy 1.x`에서도 동일하게
동작합니다.

## 2. 데이터 배치

제출물에는 데이터셋이 포함되어 있지 않습니다(과제 규정). `src/lab06.py`는 파일 위치를
기준으로 경로를 계산하므로, 배포된 `cafe_daily.csv`를 아래 위치에 두어야 합니다.

```
<압축 해제 폴더>/
├── data/
│   └── cafe_daily.csv       <- lab06_rnn_timeseries 배포본에서 복사
├── src/
│   └── lab06.py
├── results.json
├── report.pdf
└── README.md
```

## 3. 결과 재현

압축 해제 폴더에서 실행합니다.

```bash
python src/lab06.py
```

`results.json`을 덮어쓰고 같은 내용을 표준 출력으로 인쇄합니다. `run_experiment`가 가장
먼저 `set_seed()`를 호출하고, 주어진 `train_model`이 배치를 섞지 않으므로 가중치 초기화
이후에는 난수를 전혀 쓰지 않습니다. 따라서 재실행해도 `runtime_seconds`를 제외한 모든
값이 동일합니다.

기대 출력 (`metrics` 부분):

```json
{
  "n_points": 1095,
  "window": 14,
  "horizon": 1,
  "gap": 7,
  "n_train": 739,
  "n_val": 150,
  "n_test": 150,
  "scale_mean": 170.4754,
  "scale_std": 25.5496,
  "naive_scale": 16.2314,
  "persistence_mae": 15.5067,
  "persistence_rmse": 20.0934,
  "persistence_mase": 0.9554,
  "rnn_mae": 9.2921,
  "rnn_rmse": 11.6667,
  "rnn_mase": 0.5725
}
```

RNN의 테스트 MASE **0.5725**는 과제가 제시한 기준선인 persistence의 **0.9554**보다
낮습니다(낮을수록 좋음).

## 4. 테스트

```bash
python -m pytest tests/ -q
```

배포된 공개 테스트 6개가 모두 통과합니다(`tests/`는 제출물에 포함되지 않으므로 배포본의
것을 사용하십시오).

## 5. 구현한 함수

| 함수 | 내용 |
|------|------|
| `make_windows(series, window, horizon)` | `X[i] = s[i:i+window]`, `y[i] = s[i+window+horizon-1]` 인 (입력, 타깃) 쌍 |
| `temporal_split(n, val_frac, test_frac, gap)` | 시간 순서 train / val / test, 사이마다 `gap`일 엠바고 |
| `ForecastRNN` | `self.rnn` = `nn.GRU(..., batch_first=True)`, `self.head` = `nn.Linear(hidden_size, 1)` |
| `build_rnn(...)` | 학습 전 `ForecastRNN` 반환 |
| `run_experiment(series)` | persistence 기준선과 RNN을 같은 분할·윈도우·지표로 비교 |

누수를 막기 위해 지킨 것:

- **어디에서도 섞지 않습니다.** `train_test_split`, `KFold`, `shuffle`, `permutation`을
  사용하지 않았습니다. 시계열을 무작위로 나누면 미래로 과거를 학습하게 됩니다.
- **윈도우의 입력은 타깃보다 엄격히 앞섭니다.** `X[i]`의 마지막 원소는 `s[i+window-1]`,
  타깃은 `s[i+window+horizon-1]` 이므로 타깃이 입력 안에 들어가는 일이 없습니다.
- **표준화 상수는 훈련 구간에서만** 계산합니다(`scale_mean`, `scale_std`). 전체 시계열로
  계산하면 검증·테스트 구간의 수준이 변환에 섞여 들어갑니다.
- **각 구간을 따로 윈도우로 자릅니다.** 한 윈도우가 구간 경계를 넘으면 엠바고로 버린 날이나
  다른 구간의 날이 입력에 섞이기 때문입니다.
- `naive_scale`도 훈련 구간의 평균 일간 변화량이며, MASE의 분모로만 쓰입니다.

## 6. 그림

`report.pdf`의 Figure 1~3은 배포된 `notebooks/lab06_visualization.ipynb`로 생성했습니다.
노트북은 제출물에 포함되지 않습니다.
