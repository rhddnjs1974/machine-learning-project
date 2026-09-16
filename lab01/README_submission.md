# Lab 1 — EDA 재현 방법

20212788 최재원 · 머신러닝 프로젝트 (53744-01)

## 1. 환경

| 항목 | 값 |
|------|-----|
| Python | 3.10 이상 (제출자 실행 환경: 3.13.3) |
| 의존성 | `numpy`, `pandas`, `pytest` |
| 하드웨어 | CPU만 사용 |
| 실행 시간 | 1초 미만 |
| 시드 | 42 (`src/lab01.py`의 `SEED`에 고정) |

```bash
pip install "numpy>=1.26,<3" "pandas>=2.1,<3" pytest
```

## 2. 데이터 배치

제출물에는 데이터셋이 포함되어 있지 않습니다(과제 규정). `src/lab01.py`는
파일 위치를 기준으로 데이터 경로를 계산하므로, 배포된 `cafe_sales.csv`를
아래 위치에 두어야 합니다.

```
<압축 해제 폴더>/
├── data/
│   └── cafe_sales.csv     <- lab01_eda 배포본에서 복사
├── src/
│   └── lab01.py
├── results.json
├── report.pdf
└── README.md
```

## 3. 결과 재현

압축 해제 폴더에서 실행합니다.

```bash
python src/lab01.py
```

`results.json`을 덮어쓰고 같은 내용을 표준 출력으로 인쇄합니다. 재실행해도
`runtime_seconds`를 제외한 모든 값은 동일합니다.

기대 출력:

```json
{
  "lab": "lab01",
  "student_id": "20212788",
  "seed": 42,
  "metrics": {
    "n_rows_raw": 412,
    "n_rows_clean": 400,
    "n_duplicates_removed": 12,
    "missing_total_raw": 103,
    "missing_total_clean": 0,
    "n_outliers_quantity": 3,
    "top_category_by_revenue": "coffee",
    "mean_rating_clean": 4.037
  }
}
```

## 4. 테스트

```bash
python -m pytest tests/ -q
```

배포된 공개 테스트 6개가 모두 통과합니다(`tests/`는 제출물에 포함되지 않으므로
배포본의 것을 사용하십시오).

## 5. 구현한 함수

| 함수 | 내용 |
|------|------|
| `summarize_missing(df)` | 컬럼별 결측 개수. 결측이 있는 컬럼만, 개수 내림차순 |
| `clean_data(df)` | 명세의 5단계를 순서대로 적용한 복사본 반환 (입력 미변경) |
| `detect_outliers_iqr(df, column, k)` | IQR 규칙으로 표시된 행의 인덱스 목록 |
| `compute_group_stats(df, group_col, value_col)` | 그룹별 `count`/`mean`/`sum`, `sum` 내림차순 |

`clean_data`의 단계 순서는 명세를 그대로 따랐습니다. 중복 제거를 먼저 해야
대체에 쓰는 중앙값과 평균이 중복에 오염되지 않고, `unit_price`를 숫자로 바꾼
뒤에야 `total_price = unit_price * quantity` 계산이 가능하기 때문입니다.

## 6. 그림

`report.pdf`의 Figure 1~3은 배포된 `notebooks/lab01_visualization.ipynb`로
생성했습니다. 노트북은 제출물에 포함되지 않습니다. Figure 2는 y축을 로그
스케일로 그렸으며, 그 이유는 보고서 1절에 적었습니다.
