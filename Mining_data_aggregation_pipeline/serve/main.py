"""FastAPI服务 - 矿业数据检索接口.

提供基于自然语言的语义检索API，支持时间/地区/矿种过滤和RAG答案生成。
"""

import re
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List, Optional

import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from configs import settings


# ---------- 时间解析 ----------

def parse_time_filter(question: str) -> Optional[dict]:
    """从自然语言问题中解析时间过滤条件.

    Args:
        question: 用户问题.

    Returns:
        时间过滤字典，包含start_date和end_date，或None.
    """
    now = datetime.now()

    patterns: List[tuple] = [
        (r"近(\d+)天", lambda m: now - timedelta(days=int(m.group(1)))),
        (r"最近(\d+)天", lambda m: now - timedelta(days=int(m.group(1)))),
        (r"过去(\d+)天", lambda m: now - timedelta(days=int(m.group(1)))),
        (r"近(\d+)周", lambda m: now - timedelta(weeks=int(m.group(1)))),
        (r"最近(\d+)周", lambda m: now - timedelta(weeks=int(m.group(1)))),
        (r"近(\d+)个月", lambda m: now - timedelta(days=int(m.group(1)) * 30)),
        (r"最近(\d+)个月", lambda m: now - timedelta(days=int(m.group(1)) * 30)),
        (r"近一个月", lambda m: now - timedelta(days=30)),
        (r"最近一个月", lambda m: now - timedelta(days=30)),
        (r"近一周", lambda m: now - timedelta(days=7)),
        (r"最近一周", lambda m: now - timedelta(days=7)),
    ]

    for pattern, calc_fn in patterns:
        match = re.search(pattern, question)
        if match:
            start_date = calc_fn(match)
            return {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": now.strftime("%Y-%m-%d"),
            }

    return None


def parse_metadata_filters(question: str) -> dict:
    """从自然语言问题中解析元数据过滤条件.

    Args:
        question: 用户问题.

    Returns:
        元数据过滤字典，包含可能的commodity、country_or_region等.
    """
    filters: dict = {}

    # 地区过滤
    region_keywords = {
        "China": ["中国", "国内", "中国稀土", "工信部", "自然资源部", "SHFE"],
        "Australia": ["澳洲", "澳大利亚", "澳", "DISR", "Australian"],
    }
    for region, keywords in region_keywords.items():
        if any(kw in question for kw in keywords):
            filters["country_or_region"] = region
            break

    # 矿种过滤
    commodity_keywords = {
        "copper": ["铜", "copper"],
        "zinc": ["锌", "zinc"],
        "nickel": ["镍", "nickel"],
        "lithium": ["锂", "lithium"],
        "iron_ore": ["铁矿石", "铁矿", "iron ore"],
        "rare_earth": ["稀土", "rare earth"],
        "gold": ["黄金", "金", "gold"],
        "silver": ["白银", "silver"],
    }
    for commodity, keywords in commodity_keywords.items():
        if any(kw in question for kw in keywords):
            filters["commodity"] = commodity
            break

    # 来源类型过滤
    if any(kw in question for kw in ["政策", "法规", "规定", "通知", "意见", "方案"]):
        if not any(kw in question for kw in ["价格", "行情", "期货"]):
            filters["source_type"] = "policy"
    elif any(kw in question for kw in ["价格", "行情", "期货", "走势"]):
        filters["source_type"] = "price"
    elif any(kw in question for kw in ["新闻", "事件", "动态", "并购"]):
        filters["source_type"] = "news"

    return filters


def build_chroma_where(filters: dict) -> Optional[dict]:
    """构建ChromaDB where过滤条件.

    Args:
        filters: 元数据过滤字典.

    Returns:
        ChromaDB where条件字典或None.
    """
    conditions: List[dict] = []

    for key, value in filters.items():
        conditions.append({key: value})

    if not conditions:
        return None

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}


def generate_rag_answer(
    question: str, results: List[dict], model: SentenceTransformer
) -> str:
    """基于检索结果生成RAG答案.

    使用检索到的文档作为证据，生成结构化回答。

    Args:
        question: 用户问题.
        results: 检索结果列表.
        model: 向量化模型（用于相关性排序）.

    Returns:
        生成的答案文本.
    """
    if not results:
        return "抱歉，未找到与您问题相关的数据。"

    # 按score排序，取top结果
    sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)

    # 收集证据
    evidences: List[str] = []
    for i, r in enumerate(sorted_results[:5], 1):
        source_type = r.get("source_type", "")
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        score = r.get("score", 0)

        source_label = {
            "news": "新闻", "policy": "政策", "price": "价格"
        }.get(source_type, source_type)

        evidences.append(
            f"[{i}] ({source_label}) {title}\n    {snippet} (相关度: {score:.2f})"
        )

    # 检测问题类型并生成回答
    is_change_question = any(
        kw in question for kw in ["变化", "趋势", "走势", "动态", "涨", "跌"]
    )
    is_price_question = any(
        kw in question for kw in ["价格", "行情", "多少钱", "期货"]
    )
    is_policy_question = any(
        kw in question for kw in ["政策", "法规", "规定", "通知"]
    )

    answer_parts: List[str] = []

    if is_price_question:
        price_items = [
            r for r in sorted_results if r.get("source_type") == "price"
        ]
        if price_items:
            answer_parts.append("根据检索到的价格数据：")
            for r in price_items[:3]:
                answer_parts.append(f"  - {r['title']}：{r['snippet'][:100]}")
        else:
            answer_parts.append("检索结果中未直接包含价格数据，以下为相关参考信息：")

    elif is_policy_question:
        policy_items = [
            r for r in sorted_results if r.get("source_type") == "policy"
        ]
        if policy_items:
            answer_parts.append("根据检索到的政策信息：")
            for r in policy_items[:3]:
                answer_parts.append(f"  - {r['title']}：{r['snippet'][:100]}")
        else:
            answer_parts.append("检索结果中未直接包含政策数据，以下为相关参考信息：")

    elif is_change_question:
        answer_parts.append("根据检索到的数据变化趋势：")
        for r in sorted_results[:3]:
            answer_parts.append(f"  - {r['title']}：{r['snippet'][:100]}")
    else:
        answer_parts.append("根据检索结果，以下信息与您的问题最相关：")
        for r in sorted_results[:3]:
            answer_parts.append(f"  - {r['title']}：{r['snippet'][:100]}")

    # 证据引用
    answer_parts.append("\n参考来源：")
    answer_parts.extend(evidences)

    return "\n".join(answer_parts)


# ---------- 请求/响应模型 ----------

class QueryRequest(BaseModel):
    """查询请求模型.

    Attributes:
        question: 自然语言查询问题.
        top_k: 返回结果数量，默认5.
        commodity: 按矿种过滤，可选.
        region: 按地区过滤，可选.
        source_type: 按来源类型过滤，可选.
    """

    question: str = Field(min_length=1, description="自然语言查询问题")
    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数量")
    commodity: Optional[str] = Field(default=None, description="矿种过滤")
    region: Optional[str] = Field(default=None, description="地区过滤")
    source_type: Optional[str] = Field(default=None, description="来源类型过滤")


class QueryResult(BaseModel):
    """单条查询结果.

    Attributes:
        title: 数据标题.
        source_type: 数据源类型.
        commodity: 矿产品种.
        country_or_region: 国家/地区.
        snippet: 内容摘要.
        score: 相似度得分.
    """

    title: str
    source_type: str
    commodity: str = "other"
    country_or_region: str = "Global"
    snippet: str
    score: float


class QueryResponse(BaseModel):
    """查询响应模型.

    Attributes:
        question: 原始查询问题.
        answer: RAG生成的结构化答案.
        results: 查询结果列表.
        total: 结果总数.
        filters_applied: 应用的过滤条件.
    """

    question: str
    answer: str
    results: List[QueryResult]
    total: int
    filters_applied: Optional[dict] = None


class HealthResponse(BaseModel):
    """健康检查响应."""

    status: str
    collection_count: int


# ---------- 全局状态 ----------

_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理."""
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
    description="基于向量数据库的矿业数据语义检索服务，支持时间/地区/矿种过滤和RAG答案生成",
    version="2.0.0",
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

    支持时间解析（如"近7天"）、地区过滤、矿种过滤，
    并基于检索结果生成RAG结构化答案。

    Args:
        request: 查询请求.

    Returns:
        查询响应，包含RAG答案和结果列表.

    Raises:
        HTTPException: 服务未就绪或查询失败.
    """
    if _model is None or _collection is None:
        raise HTTPException(
            status_code=503,
            detail="服务尚未就绪，模型或数据库未加载",
        )

    try:
        # 1. 解析过滤条件
        filters = parse_metadata_filters(request.question)

        # 合并显式过滤参数
        if request.commodity:
            filters["commodity"] = request.commodity
        if request.region:
            filters["country_or_region"] = request.region
        if request.source_type:
            filters["source_type"] = request.source_type

        time_filter = parse_time_filter(request.question)

        where_clause = build_chroma_where(filters)

        # 2. 向量化查询
        query_embedding = _model.encode(
            [request.question],
            normalize_embeddings=True,
        ).tolist()

        # 3. ChromaDB检索
        query_params: dict = {
            "query_embeddings": query_embedding,
            "n_results": min(request.top_k * 3, 50),
            "include": ["documents", "metadatas", "distances"],
        }

        if where_clause:
            query_params["where"] = where_clause

        results = _collection.query(**query_params)

        # 4. 时间过滤（后处理，ChromaDB不支持日期范围查询）
        query_results: List[QueryResult] = []

        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            documents = results["documents"][0] if results["documents"] else [""] * len(ids)
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)

            for doc, meta, dist in zip(documents, metadatas, distances):
                # 时间过滤
                if time_filter:
                    pub_date = meta.get("publish_date", "")
                    if pub_date and pub_date < time_filter["start_date"]:
                        continue

                snippet = doc[:200] + "..." if len(doc) > 200 else doc
                score = round(1 - dist, 4)

                query_results.append(QueryResult(
                    title=meta.get("title", "未知标题"),
                    source_type=meta.get("source_type", "unknown"),
                    commodity=meta.get("commodity", "other"),
                    country_or_region=meta.get("country_or_region", "Global"),
                    snippet=snippet,
                    score=score,
                ))

        # 截取top_k
        query_results = query_results[:request.top_k]

        # 5. RAG答案生成
        rag_results = [r.model_dump() for r in query_results]
        answer = generate_rag_answer(request.question, rag_results, _model)

        # 6. 构建过滤信息
        filters_applied: dict = {}
        if filters:
            filters_applied["metadata"] = filters
        if time_filter:
            filters_applied["time"] = time_filter

        return QueryResponse(
            question=request.question,
            answer=answer,
            results=query_results,
            total=len(query_results),
            filters_applied=filters_applied if filters_applied else None,
        )

    except Exception as e:
        logger.error(f"查询处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询处理失败: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """健康检查接口."""
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
