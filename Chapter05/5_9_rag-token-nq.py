import datasets
original_load_dataset = datasets.load_dataset
def patched_load_dataset(*args, **kwargs):
    kwargs["trust_remote_code"] = True
    return original_load_dataset(*args, **kwargs)
datasets.load_dataset = patched_load_dataset

import huggingface_hub.utils._validators
original_validate_repo_id = huggingface_hub.utils._validators.validate_repo_id
def patched_validate_repo_id(repo_id):
    if repo_id == "https://storage.googleapis.com/huggingface-nlp/datasets/wiki_dpr/":
        return
    return original_validate_repo_id(repo_id)
huggingface_hub.utils._validators.validate_repo_id = patched_validate_repo_id

from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", dataset="wiki_dpr", index_name="exact")

# RAG 모델 로드
model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq")

question = "What is the capital of France?"

inputs = tokenizer(question, return_tensors="pt")

retrieved_docs = retriever.retrieve(question, return_tensors="pt")

# RAG 모델과 검색된 문서를 사용하여 답변 생성
outputs = model.generate(input_ids=inputs['input_ids'],
                          context_input_ids=retrieved_docs['context_input_ids'],
                          context_attention_mask=retrieved_docs['context_attention_mask'])

answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)