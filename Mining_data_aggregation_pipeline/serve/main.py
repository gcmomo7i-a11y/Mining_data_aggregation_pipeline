"""FastAPI服务 - 矿业数据检索接口.

提供基于自然语言的语义检索API，使用ChromaDB进行向量检索。
"""

from contextlib import asynccontextmanager
from typing import List, Optional

import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from configs import settings


# ---------- 请求/响应模型 ----------

class QueryRequest(BaseModel):
    """查询请求模型.

    Attributes:
        question: 自然语言查询问题.
        top_k: 返回结果数量，默认5.
    """

    question: str = Field(min_length=1, description="自然语言查询问题")
    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数量")


class QueryResult(BaseModel):
    """单条查询结果.

    Attributes:
        title: 数据标题.
        source_type: 数据源类型.
        snippet: 内容摘要（截取前200字）.
        score: 相似度得分.
    """

    title: str
    source_type: str
    snippet: str
    score: float


class QueryResponse(BaseModel):
    """查询响应模型.

    Attributes:
        question: 原始查询问题.
        results: 查询结果列表.
        total: 结果总数.
    """

    question: str
    results: List[QueryResult]
    total: int


class HealthResponse(BaseModel):
    """健康检查响应.

    Attributes:
        status: 服务状态.
        collection_count: 数据库中的数据条数.
    """

    status: str
    collection_count: int


# ---------- 全局状态 ----------

_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动时加载模型和数据库."""
    global _model, _chroma_client, _collection

    logger.info(f"加载向量化模型: {settings.EMBEDDING_MODEL_NAME}")
    _model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

    logger.info(f"初始化ChromaDB: {settings.CHROMA_DB_DIR}")
    _chroma_client = chromadb.PersistentClient(path=str(settings.CHROMA_DB_DIR))
    _collection = _chroma_client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(f"ChromaDB集合已加载，当前数据量: {_collection.count()}")

    yield

    logger.info("服务关闭")


# ---------- FastAPI应用 ----------

app = FastAPI(
    title="矿业数据聚合检索API",
    description="基于向量数据库的矿业数据语义检索服务",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """自然语言语义检索接口.

    接收自然语言问题，向量化后在ChromaDB中检索最相关的Top K结果。

    Args:
        request: 查询请求，包含question和top_k.

    Returns:
        查询响应，包含结果列表.

    Raises:
        HTTPException: 模型或数据库未加载时返回503.
    """
    if _model is None or _collection is None:
        raise HTTPException(
            status_code=503,
            detail="服务尚未就绪，模型或数据库未加载",
        )

    try:
        query_embedding = _model.encode(
            [request.question],
            normalize_embeddings=True,
        ).tolist()

        results = _collection.query(
            query_embeddings=query_embedding,
            n_results=request.top_k,
            include=["documents", "metadatas", "distances"],
        )

        query_results: List[QueryResult] = []

        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            documents = results["documents"][0] if results["documents"] else [""] * len(ids)
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)

            for doc, meta, dist in zip(documents, metadatas, distances):
                snippet = doc[:200] + "..." if len(doc) > 200 else doc
                score = round(1 - dist, 4)  # cosine distance -> similarity

                query_results.append(QueryResult(
                    title=meta.get("title", "未知标题"),
                    source_type=meta.get("source_type", "unknown"),
                    snippet=snippet,
                    score=score,
                ))

        return QueryResponse(
            question=request.question,
            results=query_results,
            total=len(query_results),
        )

    except Exception as e:
        logger.error(f"查询处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询处理失败: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """健康检查接口.

    Returns:
        服务状态和数据库数据量.
    """
    count = _collection.count() if _collection else 0
    return HealthResponse(status="ok", collection_count=count)


def main() -> None:
    """启动API服务."""
    import uvicorn

    uvicorn.run(
        "serve.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
