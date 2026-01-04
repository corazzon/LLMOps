import matplotlib.pyplot as plt
import numpy as np


# 스케일링 법칙 데이터 시뮬레이션
model_sizes = np.logspace(1, 4, 100)  # 모델 크기: 10^1부터 10^4까지
performance = np.log(model_sizes) / np.log(10)  # 성능 향상을 모사한 값


# 스케일링 법칙 시각화
plt.plot(model_sizes, performance, label="Scaling Law")
plt.xscale("log")
plt.xlabel("Model Size (log scale)")
plt.ylabel("Performance")
plt.title("Scaling Law for LLMs")
plt.legend()
plt.show()