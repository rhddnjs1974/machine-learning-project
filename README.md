# Machine Learning Project

Chung-Ang University · Machine Learning Project (53744-01) · Fall 2026

Lab work for the course, one folder per lab.

## Labs

| Lab | Topic | Folder |
| --- | ----- | ------ |
| 00 | Warm-up: version control and six ML equations | [`lab00/`](lab00/) |
| 01 | EDA: environment setup and pandas data handling | [`lab01/`](lab01/) |
| 02 | Classification & regression with scikit-learn | [`lab02/`](lab02/) |
| 03 | K-means, GMM and EM | [`lab03/`](lab03/) |
| 04-1 | Gradient descent and a NumPy MLP | [`lab04_1/`](lab04_1/) |
| 04-2 | PyTorch training and hyperparameter tuning | [`lab04_2/`](lab04_2/) |

## Running a lab

```bash
cd lab00
pip install -r requirements.txt
python src/lab00.py
python -m pytest tests/ -q
```
