from FlagEmbedding import FlagModel
from fastapi import FastAPI
from pydantic import BaseModel


model = FlagModel("BAAI/bge-large-zh-v1.5", use_fp16=True)

app = FastAPI()


# Define the request model
class EmbeddingRequest(BaseModel):
    input: list[str] | str
    model: str
    encoding_format: str


# Define the response model
class EmbeddingResponse(BaseModel):
    object: str
    data: list[dict]
    model: str
    usage: dict


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    input = request.input if isinstance(request.input, list) else [request.input]
    token_count = sum(len(i) for i in input)

    embeddings = model.encode(input, convert_to_numpy=False)

    response = {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": embedding.tolist(), "index": i}
            for i, embedding in enumerate(embeddings)
        ],
        "model": "BAAI/bge-large-zh-v1.5",
        "usage": {
            "prompt_tokens": token_count,
            "total_tokens": token_count,
        },
    }
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
