from openai import OpenAI

client = OpenAI(api_key="123", base_url="http://localhost:8000/v1")

emb = client.embeddings.create(
    model="BAAI/bge-large-zh-v1.5",
    input="Hello, world.",
)

print(emb)
