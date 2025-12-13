import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 재현 가능성을 위한 랜덤 시드 설정
np.random.seed(42)

# 챔피언 모델과 챌린저 모델의 성능 점수 분포 시뮬레이션
# 챔피언 모델: 평균 0.78, 표준편차 0.02
champion_scores = np.random.normal(loc=0.78, scale=0.02, size=100)

# 챌린저 모델: 평균 0.80, 표준편차 0.02
challenger_scores = np.random.normal(loc=0.80, scale=0.02, size=100)

# 그래프 생성 및 설정
plt.figure(figsize=(10, 6))

# 각 모델의 성능 점수 분포를 KDE(커널 밀도 추정) 플롯으로 표시
sns.kdeplot(champion_scores, label="Champion Model", color="blue")
sns.kdeplot(challenger_scores, label="Challenger Model", color="red")

# 그래프 제목 및 레이블 설정
plt.title("Distributions of Model Performance Scores")
plt.xlabel("Score")
plt.ylabel("Density")

# 범례 및 격자 표시
plt.legend()
plt.grid(True)

# 그래프 출력
plt.show()