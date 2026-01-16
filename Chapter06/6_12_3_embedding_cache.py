"""
이 스크립트는 SentenceTransformer를 사용하여 텍스트 임베딩을 생성하고,
joblib 라이브러리를 이용해 결과를 로컬 파일에 캐싱하는 예제입니다.

주요 기능:
- SentenceTransformer 모델('MiniLM') 로드
- `get_embeddings` 함수: 쿼리에 대한 임베딩을 반환하며, 
캐시(embedding_cache.pkl)를 확인하여 중복 연산을 방지합니다.
- 캐시가 없으면 새로 계산 후 저장하고, 있으면 저장된 값을 불러옵니다.
"""
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('MiniLM')

# 임베딩이 캐시되어 있는지 확인
def get_embeddings(query):
    cache_file = "embedding_cache.pkl"

    # 캐시가 존재하는지 확인
    try:
        embeddings_cache = joblib.load(cache_file)
    except FileNotFoundError:
        embeddings_cache = {}

    # 쿼리가 캐시에 없으면 임베딩을 계산하고 캐시에 저장
    if query not in embeddings_cache:
        embedding = model.encode([query])
        embeddings_cache[query] = embedding
        joblib.dump(embeddings_cache, cache_file)  # 캐시를 디스크에 저장

    return embeddings_cache[query]

# 쿼리
query = "What is the capital of France?"
embedding = get_embeddings(query)
print("Embedding for the query:", embedding)