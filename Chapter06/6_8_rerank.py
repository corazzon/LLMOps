import os
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from pinecone import init, Index

# 단계 1. API 키를 위한 환경 변수 설정
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
os.environ["PINECONE_API_KEY"] = "your_pinecone_api_key"
os.environ["PINECONE_ENV"] = "your_pinecone_environment"

# 단계 2. Pinecone 초기화
init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENV"])
index_name = "your_index_name"

# 인덱스가 존재하는지 확인
if index_name not in Pinecone.list_indexes():
    print(f"Index '{index_name}' not found. Please create it in Pinecone console.")
    exit()

# 단계 3. 검색기 설정
embedding_model = OpenAIEmbeddings()
retriever = Pinecone(index_name=index_name, embedding=embedding_model.embed_query)

# 단계 4. 재정렬 함수 정의
def rerank_documents(documents, query):
    """
    임베딩을 사용한 간단한 유사도 점수를 기반으로 문서를 재정렬합니다.
    """
    reranked_docs = sorted(
        documents,
        key=lambda doc: embedding_model.similarity(query, doc.page_content),
        reverse=True,
    )
    return reranked_docs[:5]  # 상위 5개 문서 반환

# 단계 5. LLM과 프롬프트 설정
llm = OpenAI(model="gpt-4")

prompt_template = """
당신은 나의 조력자입니다. 다음 문맥을 사용하여 사용자의 질문에 답변하세요:
문맥: {context}
질문: {question}
답변:
"""

prompt = PromptTemplate(template=prompt_template,
                        input_variables=["context", "question"])
