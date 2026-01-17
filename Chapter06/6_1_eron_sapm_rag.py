"""
이 스크립트는 Enron 스팸 데이터셋을 사용하여 
OpenAI 모델의 스팸 분류 성능을 테스트합니다.
두 가지 서로 다른 프롬프트(PROMPT_A, PROMPT_B)를 정의하고,
데이터셋에서 샘플링한 이메일에 대해 각 프롬프트를 사용하여 분류 작업을 수행합니다.
이를 통해 프롬프트 엔지니어링에 따른 모델의 응답 차이를 비교 분석할 수 있습니다.
"""

import pandas as pd
import numpy as np
import random
from statistics import mean, stdev
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# 테스트할 프롬프트 정의
PROMPT_A = "Is the following email spam? Respond with spam if the email is spam or ham if the email is not spam. Use only spam or ham as the answers, nothing else.\n\nSubject: {subject}\n\nMessage: {message}"
PROMPT_B = "After considering it very carefully, do you think it's likely that the email below is spam? Respond with spam if the email is spam or ham if the email is not spam. Use only spam or ham as the answers, nothing else.\n\nSubject: {subject}\n\nMessage: {message}"

# 데이터셋 불러오고 샘플 추출
df = pd.read_csv("enron_spam_data.csv")
spam_df = df[df['Spam/Ham'] == 'spam'].sample(n=30)
ham_df = df[df['Spam/Ham'] == 'ham'].sample(n=30)
sampled_df = pd.concat([spam_df, ham_df])

# 평가 함수 정의

# 실행 및 결과 출력
