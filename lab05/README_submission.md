# Lab 5 — CNN과 전이학습 재현 방법

20212788 최재원 · 머신러닝 프로젝트 (53744-01)

## 1. 환경

| 항목 | 값 |
|------|-----|
| Python | 3.10 이상 (제출자 실행 환경: 3.13.3) |
| 의존성 | `numpy`, `scikit-learn`, `torch`, `pytest` (그림 생성에만 `matplotlib`, `jupyter`) |
| 하드웨어 | CPU만 사용 (GPU·CUDA 불필요) |
| 메모리 | 1 GB 미만 |
| 실행 시간 | `python src/lab05.py` 약 7초, `pytest tests/` 약 23초 |
| 시드 | 42 (`src/lab05.py`의 `SEED`에 고정) |

```bash
pip install -r requirements.txt
```

배포된 `requirements.txt`는 `numpy<2`와 `scipy<1.15`를 고정합니다. 제출자는 `numpy 2.2.4`,
`scipy 1.17.0`, `torch 2.11.0+cpu` 조합에서 작성·검증했으며 배열과 텐서 변환, `sklearn`
import 모두 문제가 없었습니다. 코드에 버전에 의존하는 부분은 없으므로
`requirements.txt`가 고정한 버전에서도 동일하게 동작합니다.

## 2. 데이터 배치

제출물에는 데이터셋과 사전학습 가중치가 포함되어 있지 않습니다(과제 규정).
`src/lab05.py`는 파일 위치를 기준으로 경로를 계산하므로, 배포본의 두 파일을 아래 위치에
두어야 합니다.

```
<압축 해제 폴더>/
├── data/
│   ├── parts_shapes.npz        <- lab05_cnn 배포본에서 복사
│   └── backbone_shapes6.pt     <- 같음
├── src/
│   └── lab05.py
├── results.json
├── report.pdf
└── README.md
```

## 3. 결과 재현

압축 해제 폴더에서 실행합니다.

```bash
python src/lab05.py
```

`results.json`을 덮어쓰고 같은 내용을 표준 출력으로 인쇄합니다. `run_experiment`가 모델을
만들기 직전마다 `set_seed()`를 호출하므로, 두 번 실행해도 `runtime_seconds`를 제외한 모든
값이 동일합니다(확인함).

기대 출력 (`metrics` 부분):

```json
{
  "n_samples": 1200,
  "n_classes": 4,
  "n_labelled_train": 240,
  "epochs": 8,
  "baseline_accuracy": 0.2573,
  "baseline_trainable_params": 0,
  "scratch_accuracy": 0.8229,
  "scratch_trainable_params": 7940,
  "transfer_accuracy": 0.9469,
  "transfer_trainable_params": 2052
}
```

전이 모델의 검증 정확도 **0.9469**는 과제가 제시한 기준선(참조 구현의 밑바닥 CNN,
0.8240)을 넘습니다.

## 4. 테스트

```bash
python -m pytest tests/ -q
```

배포된 공개 테스트 6개가 모두 통과합니다(`tests/`는 제출물에 포함되지 않으므로 배포본의
것을 사용하십시오).

## 5. 구현한 함수

| 함수 | 내용 |
|------|------|
| `conv_output_shape(h, w, kernel_size, stride, padding)` | `floor((L+2p-k)/s)+1` 을 축마다 독립적으로 계산 |
| `SmallCNN` | `features` = Conv/ReLU/MaxPool 3회(1→8→16→32 채널), `head` = `Linear(512, n_classes)` |
| `build_cnn(n_classes)` | 모든 파라미터가 학습 가능한 `SmallCNN` 반환 |
| `freeze_backbone(model)` | `model.features`의 모든 파라미터에 `requires_grad = False`, 헤드는 유지 |
| `build_transfer_model(n_classes, weights_path)` | 사전학습 백본만 싣고 헤드는 새로 만든 뒤 동결 |
| `run_experiment(X, y)` | baseline / scratch / transfer를 같은 분할·시드·예산에서 비교 |

구현상 유의한 점:

- `conv_output_shape`는 **레이어를 만들지 않고 산술로만** 계산합니다. 더미 텐서를 만들면
  할당할 수 없을 만큼 큰 입력에서 실패하기 때문입니다.
- `build_transfer_model`은 저장된 `state_dict`에서 `features.` 로 시작하는 키만 골라
  `model.features`에 싣습니다. 원래 과제의 헤드는 6클래스용 `(6, 512)` 이라 그대로 쓸 수
  없으므로 버리고, `build_cnn`이 만든 새 헤드를 유지합니다.
- 동결은 `requires_grad = False`로 이루어지며, `make_optimizer`가 `requires_grad`가 켜진
  파라미터만 모으므로 옵티마이저에는 헤드 2,052개만 전달됩니다.
- `run_experiment`는 분할을 **한 번만** 계산해 세 모델이 공유하고, 모델을 만들기 직전마다
  `set_seed()`를 호출해 두 모델이 동일한 초기 조건에서 출발하도록 합니다.

## 6. 그림

`report.pdf`의 Figure 1~3은 배포된 `notebooks/lab05_visualization.ipynb`로 생성했습니다.
노트북은 제출물에 포함되지 않습니다.
