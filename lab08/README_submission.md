# Lab 8 — 실험 관리와 오류 분석 재현 방법

20212788 최재원 · 머신러닝 프로젝트 (53744-01)

## 1. 환경

| 항목 | 값 |
|------|-----|
| Python | 3.10 이상 (제출자 실행 환경: 3.13.3) |
| 의존성 | `numpy`, `pandas`, `scikit-learn`, `pytest` (그림 생성에만 `matplotlib`, `jupyter`) |
| 하드웨어 | CPU만 사용 |
| 메모리 | 1 GB 미만 |
| 실행 시간 | `python src/lab08.py` 약 1초, `pytest tests/` 약 9초 |
| 시드 | 42 (`src/lab08.py`의 `SEED`), 반복은 `SEEDS = (42, 43, 44, 45, 46)` |

```bash
pip install -r requirements.txt
```

배포된 `requirements.txt`는 `numpy<2`와 `scipy<1.15`를 고정합니다. 제출자는 `numpy 2.2.4`,
`scipy 1.17.0`, `scikit-learn 1.9.1` 조합에서 작성·검증했으며 경고나 오류 없이 동작했습니다.
코드에 버전에 의존하는 부분은 없으므로 `requirements.txt`가 고정한 버전에서도 동일하게
동작합니다.

## 2. 데이터 배치

제출물에는 데이터셋이 포함되어 있지 않습니다(과제 규정). `src/lab08.py`는 파일 위치를
기준으로 경로를 계산하므로, 배포된 `delivery_late.csv`를 아래 위치에 두어야 합니다.

```
<압축 해제 폴더>/
├── data/
│   └── delivery_late.csv     <- lab08_experiments 배포본에서 복사
├── src/
│   └── lab08.py
├── results.json
├── report.pdf
└── README.md
```

## 3. 결과 재현

압축 해제 폴더에서 실행합니다.

```bash
python src/lab08.py
```

`results.json`을 덮어쓰고 같은 내용을 표준 출력으로 인쇄합니다. 한 번의 실행은
`(config, seed)` 로 완전히 결정되므로, 재실행해도 `runtime_seconds`를 제외한 모든 값이
동일합니다.

기대 출력 (`metrics` 부분, 일부 생략):

```json
{
  "n_samples": 2400,
  "positive_rate": 0.4008,
  "n_train": 1800,
  "n_val": 600,
  "base_accuracy": 0.745,
  "base_f1": 0.645,
  "ablation_variants": 6,
  "ablation_rows": 30,
  "ablation_base_f1_mean": 0.6439,
  "ablation_base_f1_std": 0.023,
  "ablation_best_variant": "features=with_region",
  "ablation_best_f1_mean": 0.7355,
  "overall_accuracy": 0.745,
  "worst_slice": "island",
  "worst_slice_n": 89,
  "worst_slice_accuracy": 0.4382,
  "worst_gap": 0.3068,
  "smallest_slice": "overseas",
  "smallest_slice_n": 10,
  "ranking_top_mean": 0.6787,
  "ranking_top_std": 0.018,
  "ranking_second_mean": 0.6439,
  "ranking_second_std": 0.023,
  "ranking_second_overlaps_best": true
}
```

`ranking_second_overlaps_best`가 `true`라는 점에 주의해 주십시오. 1위 설정의 평균이 더
높지만 두 구간이 겹치므로, 시드 5개로는 둘을 구분할 수 없다는 뜻입니다.

## 4. 테스트

```bash
python -m pytest tests/ -q
```

배포된 공개 테스트 6개가 모두 통과합니다(`tests/`는 제출물에 포함되지 않으므로 배포본의
것을 사용하십시오).

## 5. 구현한 함수

| 함수 | 내용 |
|------|------|
| `run_config(config, seed, data)` | 설정 하나와 시드 하나로 한 번 학습·평가하고 기록 하나를 반환 |
| `run_ablation(base_config, factors, seeds, data)` | 한 번에 한 요인씩만 바꾼 변형들을 모든 시드로 실행한 tidy 표 |
| `error_analysis(y_true, y_pred, groups, min_count)` | 슬라이스별 지표, 슬라이스 크기, 보고할 만한 최악 슬라이스 |
| `summarize(records, metric)` | 설정별 평균과 표본표준편차, 순위, 최상위와의 구간 겹침 여부 |

구현상 유의한 점:

- `run_config`는 `set(config) != set(BASE_CONFIG)` 이면 `ValueError`를 냅니다. 키가 빠지거나
  오타가 나면 기본값으로 조용히 넘어가지 않고 즉시 실패합니다.
- 기록에 담는 설정은 `dict(config)` 사본입니다. 호출자가 자기 딕셔너리를 나중에 바꿔도
  이미 만들어진 기록은 변하지 않습니다.
- `run_ablation`은 데이터를 한 번만 읽어 모든 실행에 `data`로 넘기고, 각 변형은
  `base_config` 사본에서 **키 하나만** 바꿔 만듭니다. 한 행에 두 가지가 바뀌면 그 행은
  해석할 수 없기 때문입니다.
- `error_analysis`의 `min_count`는 표본이 적은 슬라이스를 worst_slice 후보에서만 제외하며,
  그런 슬라이스도 `n`과 함께 `slices`에 그대로 남습니다.
- `summarize`의 `std`는 표본표준편차(ddof=1)이고, 기록이 하나뿐인 설정은 `0.0`이 아니라
  `nan`입니다. 한 번의 관측으로는 분산을 추정할 수 없기 때문입니다. `overlaps_best`는
  표에 적힌 반올림된 `mean`과 `std`로 계산하며, `nan`은 0.0으로 취급합니다.

## 6. 그림

`report.pdf`의 Figure 1~3은 배포된 `notebooks/lab08_experiments.ipynb`로 생성했습니다.
노트북은 제출물에 포함되지 않습니다.
