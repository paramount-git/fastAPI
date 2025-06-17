from fastapi import FastAPI, Query
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# Load data and model at startup
df = pd.read_pickle("products.pkl")
embeddings = np.load("embeddings.npy")
index = faiss.read_index("faiss.index")
model = SentenceTransformer('all-MiniLM-L6-v2')

app = FastAPI()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/search")
def search(req: SearchRequest):
    query_emb = model.encode([req.query]).astype('float32')
    D, I = index.search(query_emb, req.top_k)
    results = []
    for idx, score in zip(I[0], D[0]):
        product = df.iloc[idx]
        results.append({
            "product_name": product["product_name"],
            "dimension": product["dimension"],
            "brand": product.get("brand", ""),
            "score": float(score)
        })
    return {"results": results}
