import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

# 예제용 단순 선형 모델
model = torch.nn.Linear(10, 2)

# 초기 학습률 설정
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# 코사인 학습률 스케줄러
# T_max는 학습률이 최소값에 도달하기까지의 epoch 수를 의미합니다.
scheduler = CosineAnnealingLR(optimizer, T_max=50)

# 훈련 루프 예시
for epoch in range(100):
    # 실제 훈련 코드에서는 아래 순서로 진행됩니다.
    # 1. 입력 데이터로 예측 수행
    # 2. 손실값 계산
    # 3. optimizer.zero_grad()
    # 4. loss.backward()
    # 5. optimizer.step()

    optimizer.step()
    scheduler.step()

    print(f"Epoch {epoch + 1}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")