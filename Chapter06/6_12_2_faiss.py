from whoosh.index import create_in
from whoosh.fields import Schema, TEXT
import faiss
import numpy as np

# 밀집 검색을 위한 FAISS 인덱스 초기화
dimension = 128
dense_index = faiss.IndexFlatL2(dimension)

# Whoosh를 사용한 희소 검색 시뮬레이션
schema = Schema(content=TEXT(stored=True))
ix = create_in("index", schema)
writer = ix.writer()

writer.add_document(content="This is a test document.")
writer.add_document(content="Another document for retrieval.")
writer.commit()

# 밀집 검색과 희소 검색을 위한 쿼리
def retrieve_dense(query_vector):
    return dense_index.search(np.array([query_vector]), k=5)

def retrieve_sparse(query):
    searcher = ix.searcher()
    results = searcher.find("content", query)
    return [hit['content'] for hit in results]

query_vector = np.random.rand(1, dimension).astype('float32')
sparse_query = "document"

# 결합된 검색 수행
dense_results = retrieve_dense(query_vector)

sparse_results = retrieve_sparse(sparse_query)

# 밀집 결과와 희소 결과 결합
combined_results = dense_results + sparse_results
print("Combined results:", combined_results)
