"""
안드레이 카파시(Andrej Karpathy)의 
"Let's build GPT: from scratch, in code, spelled out." 
강의를 바탕으로 구현된 코드입니다.

원본 소스 코드 및 강의 링크:
- GitHub: https://github.com/karpathy/ng-video-lecture
- YouTube: https://youtu.be/kCc8FmEb1nY
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import requests

# 하이퍼파라미터 정의
batch_size = 16 # 병렬로 처리할 독립적인 시퀀스의 개수
block_size = 32 # 예측을 위해 참조할 최대 문맥 길이
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------
torch.manual_seed(1337)

# 데이터셋이 없으면 다운로드
if not os.path.exists('input.txt'):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    with open('input.txt', 'wb') as f:
        f.write(requests.get(url).content)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 텍스트에 나타나는 모든 고유한 문자 집합
chars = sorted(list(set(text)))
vocab_size = len(chars)
# 문자를 정수로, 정수를 문자로 매핑하는 딕셔너리 생성
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # 인코더: 문자열을 입력받아 정수 리스트 출력
decode = lambda l: ''.join([itos[i] for i in l]) # 디코더: 정수 리스트를 입력받아 문자열 출력

# 학습용 및 검증용 데이터 분할
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # 처음 90%는 학습용, 나머지는 검증용
train_data = data[:n]
val_data = data[n:]

# 데이터 로딩
def get_batch(split):
    # 입력 x와 타겟 y의 작은 배치 생성
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ 셀프 어텐션의 개별 헤드 """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        # 어텐션 스코어("affinities") 계산
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # 값(values)의 가중 합 수행
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ 여러 개의 셀프 어텐션 헤드를 병렬로 실행 """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ 단순한 선형 레이어와 비선형 활성화 함수 계층 """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ 트랜스포머 블록: 통신(어텐션) 후 계산(피드포워드) 수행 """

    def __init__(self, n_embd, n_head):
        # n_embd: 임베딩 차원, n_head: 사용할 헤드의 개수
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# 매우 단순한 바이그램 모델
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # 각 토큰은 조회 테이블에서 다음 토큰에 대한 로짓을 직접 읽음
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd) # 최종 레이어 정규화
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx와 targets는 모두 (B,T) 형태의 정수 텐서
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx는 현재 문맥의 인덱스를 담은 (B, T) 배열
        for _ in range(max_new_tokens):
            # idx를 마지막 block_size 토큰으로 자름
            idx_cond = idx[:, -block_size:]
            # 예측값 계산
            logits, loss = self(idx_cond)
            # 마지막 타임 스텝에만 집중
            logits = logits[:, -1, :] # (B, C) 형태가 됨
            # 소프트맥스를 적용하여 확률 계산
            probs = F.softmax(logits, dim=-1) # (B, C)
            # 분포로부터 샘플링
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # 샘플링된 인덱스를 실행 중인 시퀀스에 추가
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)
# 모델의 파라미터 수 출력
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# PyTorch 옵티마이저 생성
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # 주기적으로 학습 및 검증 세트에 대한 손실 평가
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(
            f"단계 {iter}: 학습 손실 {losses['train']:.4f}, 검증 손실 {losses['val']:.4f}"
        )

    # 데이터 배치 샘플링
    xb, yb = get_batch('train')

    # 손실 평가
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 모델로부터 텍스트 생성
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))