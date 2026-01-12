from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/generate")
async def generate_text(request: TextRequest):
    # 임시 응답, LLM 추론 로직으로 대체해야 합니다
    generated_text = f"Generated text based on: {request.text}"
    return {"input": request.text, "output": generated_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)