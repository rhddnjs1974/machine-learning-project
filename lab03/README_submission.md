# Lab 3 — K-means, GMM, EM 재현 방법

20212788 최재원 · 머신러닝 프로젝트 (53744-01)

## 1. 환경

| 항목 | 값 |
|------|-----|
| Python | 3.10 이상 (제출자 실행 환경: 3.13.3) |
| 의존성 | `numpy`, `pytest` (그림 생성에만 `matplotlib`, `jupyter`) |
| 하드웨어 | CPU만 사용 |
| 실행 시간 | 1분 미만 (측정값 1.78초) |
| 시드 | 42 (`src/lab03.py`의 `SEED`에 고정) |

`src/lab03.py`는 **numpy와 표준 라이브러리만** 사용합니다. `scikit-learn`과 `scipy`는
import하지 않습니다.

```bash
pip install "numpy>=1.26,<3" "pytest>=8,<9"
```

## 2. 데이터 배치

제출물에는 데이터셋이 포함되어 있지 않습니다(과제 규정). `src/lab03.py`는 파일 위치를
기준으로 데이터 경로를 계산하므로, 배포된 `lab03_scene.npz`를 아래 위치에 두어야 합니다.

```
<압축 해제 폴더>/
├── data/
│   └── lab03_scene.npz     <- lab03_clustering_em 배포본에서 복사
├── src/
│   └── lab03.py
├── results.json
├── report.pdf
└── README.md
```

## 3. 결과 재현

압축 해제 폴더에서 실행합니다.

```bash
python src/lab03.py
```

`results.json`을 덮어쓰고 같은 내용을 표준 출력으로 인쇄합니다. 모든 난수는
`main()`이 만든 `np.random.default_rng(42)` 하나에서 나오므로, 재실행해도
`runtime_seconds`를 제외한 모든 값이 동일합니다.

기대 출력 (`metrics` 부분):

```json
{
  "n_points": 1614,
  "kmeans_inertia": 281.8135,
  "ari_kmeans": 0.6082,
  "ari_gmm": 0.9104,
  "ll_first": -939.085,
  "ll_final": -653.7012,
  "ll_min_delta": -7e-09,
  "bic_k2": 2791.0291,
  "bic_k3": 2244.1721,
  "bic_k4": 1595.4747,
  "bic_k5": 1645.8338,
  "bic_k6": 1684.4573,
  "bic_k7": 1731.1309,
  "bic_k8": 1772.8624,
  "best_k": 4
}
```

## 4. 테스트

```bash
python -m pytest tests/ -q
```

배포된 공개 테스트 6개가 모두 통과합니다(`tests/`는 제출물에 포함되지 않으므로 배포본의
것을 사용하십시오).

## 5. 구현한 함수

| 함수 | 내용 |
|------|------|
| `kmeans(X, k, rng, n_init, max_iter)` | k-means++ D² 시딩, Lloyd 반복, 빈 클러스터 재시딩, inertia 기준 최선의 재시작 선택 |
| `gmm_em(X, k, rng, max_iter, reg)` | k-means로 초기화한 완전 공분산 GMM. 로그 공간 E-step, 닫힌 형태 M-step, 로그우도 추이 반환 |
| `bic(ll, n_params, n)` | `-2 * ll + n_params * log(n)` |
| `select_k(X, k_range, rng)` | k마다 GMM 하나를 적합하고 BIC 최소인 k 선택 |
| `run_experiment(data, rng)` | K-means와 GMM의 ARI, EM 단조성 확인, k = 2~8의 BIC 곡선 |

구현상 유의한 점:

- **E-step은 전부 로그 공간**입니다. 로그밀도는 `np.linalg.slogdet`으로 행렬식의 로그를
  직접 얻고, 마할라노비스 항은 역행렬 대신 `np.linalg.solve`로 계산합니다. 정규화는 행
  최댓값을 뺀 뒤 log-sum-exp로 처리하므로 언더플로가 발생하지 않습니다.
- `reg * I`는 초기화 시점과 매 M-step마다 모든 공분산에 더해, 평면에 가까운 클러스터에서도
  공분산이 특이해지지 않습니다.
- 정답 라벨은 `run_experiment`의 `adjusted_rand_index` 호출 두 곳에서만 쓰이며,
  `kmeans` / `gmm_em` / `select_k`에는 어떤 형태로도 전달되지 않습니다.
- 모든 난수는 인자로 받은 `rng` 하나에서만 나옵니다(`np.random.*` 미사용).

## 6. 그림

`report.pdf`의 Figure 1~3은 배포된 `notebooks/lab03_visualization.ipynb`로 생성했습니다.
노트북은 제출물에 포함되지 않습니다.
