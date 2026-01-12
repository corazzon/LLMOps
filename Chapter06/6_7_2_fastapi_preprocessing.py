from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PreprocessRequest(BaseModel):
    text: str

@app.post("/preprocess")
async def preprocess(request: PreprocessRequest):
    # 기본 전처리 로직
    preprocessed_text = request.text.lower().strip()
    return {"original": request.text, "processed": preprocessed_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
