# Lab 9 — GAN과 확산 모델 재현 방법

20212788 최재원 · 머신러닝 프로젝트 (53744-01)

## 1. 환경

| 항목 | 값 |
|------|-----|
| Python | 3.10 이상 (제출자 실행 환경: 3.13.3) |
| 의존성 | `numpy`, `torch`, `pytest` (그림 생성에만 `matplotlib`, `jupyter`) |
| 하드웨어 | CPU만 사용 (PyTorch CPU 빌드로 충분) |
| 메모리 | 1 GB 미만 |
| 실행 시간 | `python src/lab09.py` 약 25초, `pytest tests/` 약 14초 |
| 시드 | 42 (`src/lab09.py`의 `SEED`) |

환경 자가진단:

```bash
python -c "import numpy, torch; print('numpy', numpy.__version__, '/ torch', torch.__version__)"
```

배포된 `requirements.txt`는 `numpy<2`를 고정합니다. 제출자는 `numpy 2.2.4`와
`torch 2.11.0+cpu` 조합에서 작성·검증했으며 배열과 텐서 변환에 문제가 없었습니다. 코드에
버전에 의존하는 부분은 없으므로 `requirements.txt`가 고정한 `numpy 1.x`에서도 동일하게
동작합니다.

## 2. 데이터 배치

제출물에는 데이터셋이 포함되어 있지 않습니다(과제 규정). `src/lab09.py`는 파일 위치를
기준으로 경로를 계산하므로, 배포된 `lab09_ring.npz`를 아래 위치에 두어야 합니다.

```
<압축 해제 폴더>/
├── data/
│   └── lab09_ring.npz       <- lab09_generative 배포본에서 복사
├── src/
│   └── lab09.py
├── results.json
├── report.pdf
└── README.md
```

## 3. 결과 재현

압축 해제 폴더에서 실행합니다.

```bash
python src/lab09.py
```

`results.json`을 덮어쓰고 같은 내용을 표준 출력으로 인쇄합니다. 약 25초가 걸립니다.

**두 번 실행해 확인했습니다.** `gan_sample_seconds`와 `ddpm_sample_seconds`(벽시계)를
제외한 모든 값이 두 실행에서 동일했습니다.

기대 출력 (`metrics` 중 곡선을 제외한 부분):

```json
{
  "n_train": 2000,
  "n_gen": 2000,
  "coverage_gan": 6,
  "coverage_ddpm": 8,
  "hq_gan": 0.942,
  "hq_ddpm": 0.8585,
  "gan_mode_shares": [0.1495, 0.013, 0.1435, 0.1925, 0.213, 0.0105, 0.135, 0.143],
  "ddpm_mode_shares": [0.146, 0.119, 0.118, 0.0925, 0.1085, 0.153, 0.1305, 0.1325],
  "d_loss_final": 1.3181,
  "g_loss_final": 1.0039,
  "ddpm_loss_final": 0.2443
}
```

`gan_d_curve`와 `gan_g_curve`는 각 50개, `ddpm_curve`는 60개 항목입니다
(`LOG_EVERY = 100` 간격).

## 4. 테스트

```bash
python -m pytest tests/ -q
```

배포된 공개 테스트 6개가 모두 통과합니다(`tests/`는 제출물에 포함되지 않으므로 배포본의
것을 사용하십시오).

## 5. 구현한 함수

| 함수 | 내용 |
|------|------|
| `gan_step(G, D, optG, optD, real_batch, rng_noise)` | 판별자 → 생성자 순의 교대 1스텝, 비포화 생성자 손실 |
| `make_schedule(T, beta_min, beta_max)` | 선형 베타 일정과 `alpha`, `alphabar` (길이 T, 앞에 1을 붙이지 않음) |
| `q_sample(x0, t, noise, alphabar)` | 닫힌 형태 전방 노이징 `sqrt(ab)*x0 + sqrt(1-ab)*noise` |
| `ddpm_train_loss(model, x0, t, noise)` | 예측 잡음과 실제 잡음의 MSE |
| `ddpm_sample(model, n, rng)` | 역방향 조상 샘플링, `sigma_t = sqrt(beta_t)`, t = 0에서는 잡음 없음 |
| `run_experiment(data, rng, ...)` | 두 모델을 고정 예산으로 학습하고 커버리지·품질·샘플링 시간 비교 |

구현상 유의한 점:

- `gan_step`의 판별자 단계에서 가짜 샘플에 `.detach()`를 겁니다. 떼어 내지 않으면 그
  역전파가 생성자까지 흘러들어 `optG.step()`이 오염된 기울기로 갱신됩니다.
- 생성자 손실은 비포화 형태 `BCEWithLogits(D(G(z)), 1)` 입니다. 포화 형태
  `-log(1 - D(G(z)))` 는 판별자가 이길 때 기울기가 사라집니다.
- 난수는 전부 주입받은 `torch.Generator`에서만 꺼냅니다. `ddpm_sample`은 초기 1회와
  t > 0 인 동안의 (T−1)회, 정확히 그 순서로만 뽑습니다.
- `alphabar[t]`는 **누적곱**입니다. `alpha[t]`를 쓰면 전방 노이징이 한 스텝 분량만
  반영됩니다.
- `data["mode_centers"]`는 `run_experiment` 안의 `mode_stats` 호출에서만 읽으며, 학습
  경로에는 어떤 형태로도 들어가지 않습니다.

## 6. 그림

`report.pdf`의 Figure 1~3은 배포된 `notebooks/lab09_visualization.ipynb`로 생성했습니다.
노트북은 제출물에 포함되지 않습니다.
