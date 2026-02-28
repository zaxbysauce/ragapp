"""FlagEmbedding server for BGE-M3 tri-vector embeddings."""
import logging
import os
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from FlagEmbedding import BGEM3FlagModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model with device auto-detection
device = "cuda" if torch.cuda.is_available() else "cpu"
use_fp16 = device == "cuda"
logger.info(f"Loading BGE-M3 model on {device} (fp16={use_fp16})")

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=use_fp16, device=device)
logger.info("BGE-M3 model loaded successfully")

app = FastAPI(title="FlagEmbedding Server", version="1.0.0")


class EmbedRequest(BaseModel):
    input: str | List[str]
    model: str = "BAAI/bge-m3"


class EmbedResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str
    usage: Dict[str, int]


class TriVectorResponse(BaseModel):
    dense: List[float]
    sparse: Dict[str, float]
    colbert: Optional[List[float]] = None


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": "BAAI/bge-m3",
        "supports_sparse": True,
        "device": device
    }


@app.post("/embed", response_model=List[TriVectorResponse])
async def embed_tri_vector(request: EmbedRequest):
    """
    Generate tri-vector embeddings (dense + sparse + colbert).
    
    Returns dense vectors, sparse token weights, and null colbert (deferred).
    """
    try:
        texts = request.input if isinstance(request.input, list) else [request.input]
        
        # Encode with BGE-M3
        outputs = model.encode(
            texts,
            batch_size=32,
            max_length=8192,
            return_dense=True,
            return_sparse=True,
            return_colbert=False  # Deferred to Phase 7
        )
        
        results = []
        for i, text in enumerate(texts):
            # Dense vector
            dense = outputs['dense_vecs'][i].tolist()
            
            # Sparse vector - convert to token:weight dict
            sparse_vec = outputs['lexical_weights'][i]
            sparse = {str(k): float(v) for k, v in sparse_vec.items()}
            
            results.append(TriVectorResponse(
                dense=dense,
                sparse=sparse,
                colbert=None  # Deferred to Phase 7
            ))
        
        return results
        
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/embeddings", response_model=EmbedResponse)
async def openai_compatible_embeddings(request: EmbedRequest):
    """
    OpenAI-compatible embedding endpoint (dense vectors only).
    """
    try:
        texts = request.input if isinstance(request.input, list) else [request.input]
        
        outputs = model.encode(
            texts,
            batch_size=32,
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert=False
        )
        
        data = []
        for i, vec in enumerate(outputs['dense_vecs']):
            data.append({
                "object": "embedding",
                "embedding": vec.tolist(),
                "index": i
            })
        
        return EmbedResponse(
            data=data,
            model="BAAI/bge-m3",
            usage={
                "prompt_tokens": sum(len(t.split()) for t in texts),
                "total_tokens": sum(len(t.split()) for t in texts)
            }
        )
        
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "18080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
