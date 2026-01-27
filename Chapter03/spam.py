# 필요한 라이브러리 임포트
import pandas as pd  # 데이터 처리를 위한 pandas
import numpy as np   # 수치 연산을 위한 numpy
import random        # 무작위 선택을 위한 random
from statistics import mean, stdev  # 통계 계산(평균, 표준편차)을 위한 함수
import os            # 운영체제 기능 사용 (환경변수 등)
from openai import OpenAI  # OpenAI API 클라이언트
from dotenv import load_dotenv  # .env 파일 로드

# 환경 변수 로드 (.env 파일에서 API 키 등을 가져옴)
load_dotenv()

# OpenAI 클라이언트 초기화 (API 키 설정)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ====================================================================
# 프롬프트 정의
# ====================================================================

# 프롬프트 A: 직접적인 질문 방식
PROMPT_A = (
    "다음 이메일이 스팸인가요? "
    "이메일이 스팸이면 spam, 스팸이 아니면 ham으로 답하세요. "
    "spam 또는 ham만 답변으로 사용하고 다른 것은 쓰지 마세요.\n\n"
    "제목: {subject}\n\n"
    "내용: {message}"
)

# 프롬프트 B: 신중한 고려를 요청하는 방식
PROMPT_B = (
    "매우 신중하게 고려한 후, 아래 이메일이 스팸일 가능성이 높다고 생각하시나요? "
    "이메일이 스팸이면 spam, 스팸이 아니면 ham으로 답하세요. "
    "spam 또는 ham만 답변으로 사용하고 다른 것은 쓰지 마세요.\n\n"
    "제목: {subject}\n\n"
    "내용: {message}"
)

# ====================================================================
# 데이터셋 로드 및 샘플링
# ====================================================================

# Enron 스팸 데이터셋 로드 (CSV 파일 읽기)
df = pd.read_csv("./Chapter03/enron_spam_data.csv")

# ====================================================================
# 평가 함수
# ====================================================================

def inject_noise(text, label):
    """
    이메일 텍스트에 노이즈를 주입하여 분류 난이도를 높임
    """
    spam_footers = [
        "\n\n[광고] 지금 바로 클릭하고 1등 당첨번호 확인하세요! 무료 증정 이벤트 중.",
        "\n\n(주)대출나라 - 누구나 즉시 대출 가능, 최저 금리 보장. 긴급 자금 지원.",
        "\n\n본 메일은 정보통신망법 규정에 의거하여 전송되는 홍보성 메일입니다."
    ]
    ham_signatures = [
        "\n\nBest regards,\nJohn Doe\nSenior Marketing Manager | Enron Corp",
        "\n\n이 메일은 보안 검색을 통과했습니다. 첨부파일 확인 시 주의하시기 바랍니다.",
        "\n\n-----------------\nSent from my iPhone"
    ]
    
    # 정상 메일(Ham)에 스팸성 바닥글을 주입하여 오탐(FP) 유도 (50% 확률)
    if label == 'ham' and random.random() > 0.5:
        text += random.choice(spam_footers)
    # 스팸 메일(Spam)에 정상적인 서명을 추가하여 미탐(FN) 유도 (50% 확률)
    elif label == 'spam' and random.random() > 0.5:
        text += random.choice(ham_signatures)
        
    return text

def evaluate_prompt(prompt_template):
    """
    주어진 프롬프트 템플릿으로 스팸 분류 성능을 평가
    """
    # 매 평가마다 새로 샘플링하여 변동성 확보 (난이도 상향 핵심)
    spam_samples = df[df['Spam/Ham'] == 'spam'].sample(n=30)
    ham_samples = df[df['Spam/Ham'] == 'ham'].sample(n=30)
    current_sampled_df = pd.concat([spam_samples, ham_samples])

    # 혼동 행렬(Confusion Matrix) 초기화
    true_positive = 0   # TP: 스팸을 스팸으로 정확히 분류
    false_positive = 0  # FP: 정상을 스팸으로 잘못 분류 (오탐)
    true_negative = 0   # TN: 정상을 정상으로 정확히 분류
    false_negative = 0  # FN: 스팸을 정상으로 잘못 분류 (미탐)

    # 각 이메일에 대해 평가 수행
    for _, row in current_sampled_df.iterrows():
        subject = row['Subject']
        message = inject_noise(str(row['Message']), row['Spam/Ham']) # 노이즈 주입
        actual_label = row['Spam/Ham']
        
        # 프롬프트 생성
        prompt = prompt_template.format(subject=subject, message=message)
        
        # OpenAI API 호출 및 예측
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4o-mini",
                temperature=0.7 # 변동성을 위해 약간의 온도를 줌
            )
            predicted_label = response.choices[0].message.content.strip().lower()
        except Exception as e:
            print(f"OpenAI API 호출 오류: {e}")
            continue

        # 혼동 행렬 업데이트
        if 'spam' in predicted_label and actual_label == 'spam':
            true_positive += 1
        elif 'spam' in predicted_label and actual_label == 'ham':
            false_positive += 1
        elif 'ham' in predicted_label and actual_label == 'ham':
            true_negative += 1
        elif 'ham' in predicted_label and actual_label == 'spam':
            false_negative += 1

    # 정밀도 및 재현율 계산
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    return precision, recall

def run_experiments(prompt_template, n_experiments=5): # 시간 관계상 기본 5회로 조정
    """
    동일한 프롬프트로 여러 번 실험을 수행하여 평균 성능 측정
    LLM의 응답은 확률적이므로, 한 번의 실험만으로는 신뢰할 수 없음
    
    Args:
        prompt_template (str): 평가할 프롬프트 템플릿
        n_experiments (int): 실험 반복 횟수
        
    Returns:
        tuple: (정밀도 평균, 정밀도 표준편차, 재현율 평균, 재현율 표준편차)
    """
    precisions = []
    recalls = []
    
    # 여러 번 실험 수행
    for n in range(n_experiments):
        print(f"실험 {n+1}/{n_experiments} 진행 중...")
        precision, recall = evaluate_prompt(prompt_template)
        print(f"정밀도: {precision:.4f}, 재현율: {recall:.4f}")
        precisions.append(precision)
        recalls.append(recall)
    
    # 통계 계산 (평균 및 표준편차)
    # 표준편차가 작을수록 모델의 성능이 일관됨을 의미
    precision_mean = mean(precisions)
    precision_stdev = stdev(precisions)
    recall_mean = mean(recalls)
    recall_stdev = stdev(recalls)
    
    return precision_mean, precision_stdev, recall_mean, recall_stdev

# ====================================================================
# 메인 실행
# ====================================================================

if __name__ == "__main__":
    # 프롬프트 A 평가 (직접적인 질문)
    print("=" * 60)
    print("프롬프트 A 평가 중...")
    print("=" * 60)
    precision_mean_a, precision_stdev_a, recall_mean_a, recall_stdev_a = run_experiments(PROMPT_A)
    print(f"\n[결과] 프롬프트 A")
    print(f"정밀도: {precision_mean_a:.4f} ± {precision_stdev_a:.4f}")
    print(f"재현율: {recall_mean_a:.4f} ± {recall_stdev_a:.4f}")

    # 프롬프트 B 평가 (신중한 고려 요청 유도)
    print("\n" + "=" * 60)
    print("프롬프트 B 평가 중...")
    print("=" * 60)
    precision_mean_b, precision_stdev_b, recall_mean_b, recall_stdev_b = run_experiments(PROMPT_B)
    print(f"\n[결과] 프롬프트 B")
    print(f"정밀도: {precision_mean_b:.4f} ± {precision_stdev_b:.4f}")
    print(f"재현율: {recall_mean_b:.4f} ± {recall_stdev_b:.4f}")
    
    # 최종 비교
    print("\n" + "=" * 60)
    print("최종 비교")
    print("=" * 60)
    print(f"프롬프트 A - 정밀도: {precision_mean_a:.4f} ± {precision_stdev_a:.4f}, "
          f"재현율: {recall_mean_a:.4f} ± {recall_stdev_a:.4f}")
    print(f"프롬프트 B - 정밀도: {precision_mean_b:.4f} ± {precision_stdev_b:.4f}, "
          f"재현율: {recall_mean_b:.4f} ± {recall_stdev_b:.4f}")