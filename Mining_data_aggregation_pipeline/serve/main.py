"""FastAPI服务 - 矿业数据检索接口.

提供基于自然语言的语义检索API，支持时间/地区/矿种过滤和推理式RAG答案生成。
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
    """从自然语言问题中解析时间过滤条件."""
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
            return {"start_date": start_date.strftime("%Y-%m-%d"), "end_date": now.strftime("%Y-%m-%d")}
    return None


def parse_metadata_filters(question: str) -> dict:
    """从自然语言问题中解析元数据过滤条件."""
    filters: dict = {}
    region_keywords = {
        "China": ["中国", "国内", "中国稀土", "工信部", "自然资源部", "SHFE"],
        "Australia": ["澳洲", "澳大利亚", "澳", "DISR", "Australian"],
    }
    for region, keywords in region_keywords.items():
        if any(kw in question for kw in keywords):
            filters["country_or_region"] = region
            break
    commodity_keywords = {
        "copper": ["铜", "copper"], "zinc": ["锌", "zinc"],
        "nickel": ["镍", "nickel"], "lithium": ["锂", "lithium"],
        "iron_ore": ["铁矿石", "铁矿", "iron ore"],
        "rare_earth": ["稀土", "rare earth"],
        "gold": ["黄金", "金", "gold"], "silver": ["白银", "silver"],
    }
    for commodity, keywords in commodity_keywords.items():
        if any(kw in question for kw in keywords):
            filters["commodity"] = commodity
            break
    if any(kw in question for kw in ["政策", "法规", "规定", "通知", "意见", "方案"]):
        if not any(kw in question for kw in ["价格", "行情", "期货"]):
            filters["source_type"] = "policy"
    elif any(kw in question for kw in ["价格", "行情", "期货", "走势"]):
        filters["source_type"] = "price"
    elif any(kw in question for kw in ["新闻", "事件", "动态", "并购"]):
        filters["source_type"] = "news"
    return filters


def build_chroma_where(filters: dict) -> Optional[dict]:
    """构建ChromaDB where过滤条件."""
    conditions = [{k: v} for k, v in filters.items()]
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ---------- 推理式RAG答案生成 ----------

def _classify_question(question: str) -> str:
    """分类问题意图."""
    if any(kw in question for kw in ["价格", "行情", "多少钱", "期货", "报价"]):
        return "price_inquiry"
    if any(kw in question for kw in ["政策", "法规", "规定", "通知", "战略"]):
        return "policy_inquiry"
    if any(kw in question for kw in ["变化", "趋势", "走势", "对比", "差异", "涨跌", "涨了", "跌了"]):
        return "change_analysis"
    if any(kw in question for kw in ["区别", "不同", "比较", "相比"]):
        return "comparison"
    return "general"


def _extract_entities(question: str) -> dict:
    """从问题中提取实体."""
    filters = parse_metadata_filters(question)
    entities: dict = {}
    commodity_map = {
        "copper": "铜", "zinc": "锌", "nickel": "镍", "lithium": "锂/碳酸锂",
        "iron_ore": "铁矿石", "rare_earth": "稀土", "gold": "黄金",
        "silver": "白银", "aluminium": "铝", "lead": "铅", "tin": "锡",
    }
    if "commodity" in filters:
        entities["commodity"] = commodity_map.get(filters["commodity"], filters["commodity"])
    region_map = {"China": "中国", "Australia": "澳大利亚", "Global": "全球"}
    if "country_or_region" in filters:
        entities["region"] = region_map.get(filters["country_or_region"], filters["country_or_region"])
    type_map = {"news": "新闻", "policy": "政策", "price": "价格"}
    if "source_type" in filters:
        entities["source_type"] = type_map.get(filters["source_type"], filters["source_type"])
    return entities


def _infer_price_trend(price_evidence: List[dict], commodity_name: str) -> str:
    """从价格证据中推理价格趋势."""
    up_count = sum(1 for r in price_evidence if "上涨" in r.get("snippet", ""))
    down_count = sum(1 for r in price_evidence if "下跌" in r.get("snippet", ""))
    if up_count > down_count:
        trend = "偏强上行"
    elif down_count > up_count:
        trend = "偏弱下行"
    else:
        trend = "震荡整理"
    return (f"\n趋势推理：{commodity_name}价格整体呈{trend}态势，"
            f"在检索到的{len(price_evidence)}条数据中，上涨{up_count}条、下跌{down_count}条。")


def _infer_policy_impact(policies: List[dict], commodity_name: str, region_name: str) -> str:
    """推理政策影响."""
    impact_keywords = {"限制": "收紧供给", "管制": "收紧供给", "扶持": "提振需求",
                       "补贴": "提振需求", "投资": "扩大产能", "战略": "战略布局", "储备": "保障供应安全"}
    impacts: List[str] = []
    all_snippets = " ".join(r.get("snippet", "") for r in policies)
    for kw, impact in impact_keywords.items():
        if kw in all_snippets:
            impacts.append(impact)
    if impacts:
        return (f"\n政策影响推理：{region_name}{commodity_name}相关政策"
                f"主要方向为{'、'.join(set(impacts))}，共涉及{len(policies)}项政策文件。")
    return f"\n共检索到{len(policies)}项相关政策文件。"


def _infer_change_trend(evidence: List[dict], commodity_name: str) -> str:
    """推理变化趋势."""
    types_found: List[str] = []
    for r in evidence:
        st = r.get("source_type", "")
        label = {"price": "价格数据", "policy": "政策信号", "news": "新闻动态"}.get(st, st)
        if label not in types_found:
            types_found.append(label)
    snippets = " ".join(r.get("snippet", "") for r in evidence)
    direction = ""
    if "上涨" in snippets or "上行" in snippets:
        direction = "上行"
    elif "下跌" in snippets or "下行" in snippets:
        direction = "下行"
    elif "收紧" in snippets or "限制" in snippets:
        direction = "政策收紧方向"
    elif "放松" in snippets or "扶持" in snippets:
        direction = "政策宽松方向"
    cross_ref = "、".join(types_found)
    result = f"综合{cross_ref}交叉验证"
    if direction:
        result += f"，{commodity_name}整体变化方向为{direction}"
    else:
        result += f"，{commodity_name}近期变化方向尚不明确"
    return result + "。建议关注后续数据更新。"


def _format_citations(results: List[dict]) -> str:
    """格式化证据引用."""
    lines = ["参考来源："]
    for i, r in enumerate(results, 1):
        st = r.get("source_type", "")
        label = {"news": "新闻", "policy": "政策", "price": "价格"}.get(st, st)
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        score = r.get("score", 0)
        commodity = r.get("commodity", "")
        region = r.get("country_or_region", "")
        meta_parts = [label]
        if commodity and commodity != "other":
            meta_parts.append(commodity)
        if region and region != "Global":
            meta_parts.append(region)
        lines.append(f"[{i}] ({'/'.join(meta_parts)}) {title}\n    {snippet[:150]}\n    相关度: {score:.2f}")
    return "\n".join(lines)


def generate_rag_answer(
    question: str, results: List[dict], model: SentenceTransformer
) -> str:
    """推理式RAG答案生成.

    通过问题分解→证据提取→交叉验证→综合推理→结论生成，
    而非简单拼接摘要。

    Args:
        question: 用户问题.
        results: 检索结果列表.
        model: 向量化模型.

    Returns:
        推理式答案文本.
    """
    if not results:
        return "抱歉，未找到与您问题相关的数据。"

    sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)

    # Step 1: 问题分解
    q_intent = _classify_question(question)
    q_entities = _extract_entities(question)
    time_scope = parse_time_filter(question)

    # Step 2: 按类型分组证据
    evidence_by_type: dict[str, List[dict]] = {"news": [], "policy": [], "price": []}
    for r in sorted_results:
        st = r.get("source_type", "")
        if st in evidence_by_type:
            evidence_by_type[st].append(r)

    # Step 3: 推理式答案生成
    commodity_name = q_entities.get("commodity", "")
    region_name = q_entities.get("region", "")
    time_desc = ""
    if time_scope:
        time_desc = f"（{time_scope['start_date']}至{time_scope['end_date']}）"

    parts: List[str] = []

    if q_intent == "price_inquiry":
        price_ev = evidence_by_type["price"]
        if price_ev:
            parts.append(f"根据检索到的{commodity_name}价格数据{time_desc}：")
            for r in price_ev[:5]:
                parts.append(f"  · {r.get('title', '')}：{r.get('snippet', '')[:120]}")
            if len(price_ev) >= 2:
                parts.append(_infer_price_trend(price_ev, commodity_name))
        else:
            parts.append(f"未检索到{commodity_name}的直接价格数据。")
            for r in sorted_results[:3]:
                parts.append(f"  · {r.get('title', '')}：{r.get('snippet', '')[:100]}")

    elif q_intent == "policy_inquiry":
        policy_ev = evidence_by_type["policy"]
        if policy_ev:
            parts.append(f"根据检索到的{region_name}{commodity_name}相关政策{time_desc}：")
            seen: set[str] = set()
            unique: List[dict] = []
            for r in policy_ev:
                base = re.sub(r"\s*\(\d{4}-\d{2}-\d{2}\)", "", r.get("title", ""))
                if base not in seen:
                    seen.add(base)
                    unique.append(r)
            for r in unique[:5]:
                parts.append(f"  · {r.get('title', '')}")
                parts.append(f"    核心内容：{r.get('snippet', '')[:150]}")
            if len(unique) > 1:
                parts.append(_infer_policy_impact(unique, commodity_name, region_name))
        else:
            parts.append(f"未检索到{region_name}{commodity_name}的直接政策数据。")
            for r in sorted_results[:3]:
                parts.append(f"  · {r.get('title', '')}：{r.get('snippet', '')[:100]}")

    elif q_intent == "change_analysis":
        parts.append(f"关于{commodity_name}{region_name}的变化分析{time_desc}：")
        all_ev = evidence_by_type["price"] + evidence_by_type["policy"] + evidence_by_type["news"]
        if all_ev:
            parts.append("\n最新动态：")
            for r in all_ev[:2]:
                parts.append(f"  · {r.get('title', '')}：{r.get('snippet', '')[:120]}")
            parts.append("\n变化趋势推理：")
            parts.append(_infer_change_trend(all_ev, commodity_name))
        else:
            parts.append("未检索到足够的数据进行变化分析。")

    elif q_intent == "comparison":
        parts.append(f"关于{commodity_name}的对比分析：")
        for st, label in [("price", "价格数据"), ("policy", "政策法规"), ("news", "新闻动态")]:
            ev = evidence_by_type[st]
            if ev:
                parts.append(f"\n{label}：")
                for r in ev[:2]:
                    parts.append(f"  · {r.get('title', '')}：{r.get('snippet', '')[:100]}")

    else:
        parts.append("根据检索结果综合分析：")
        for st, label in [("price", "价格数据"), ("policy", "政策法规"), ("news", "新闻动态")]:
            ev = evidence_by_type[st]
            if ev:
                parts.append(f"\n【{label}】")
                for r in ev[:2]:
                    parts.append(f"  · {r.get('title', '')}：{r.get('snippet', '')[:120]}")

    # Step 4: 证据引用
    citations = _format_citations(sorted_results[:5])
    return f"{chr(10).join(parts)}\n\n{citations}"


# ---------- 请求/响应模型 ----------

class QueryRequest(BaseModel):
    question: str = Field(min_length=1, description="自然语言查询问题")
    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数量")
    commodity: Optional[str] = Field(default=None, description="矿种过滤")
    region: Optional[str] = Field(default=None, description="地区过滤")
    source_type: Optional[str] = Field(default=None, description="来源类型过滤")


class QueryResult(BaseModel):
    title: str
    source_type: str
    commodity: str = "other"
    country_or_region: str = "Global"
    snippet: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    results: List[QueryResult]
    total: int
    filters_applied: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str
    collection_count: int


# ---------- 全局状态 ----------

_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _chroma_client, _collection
    logger.info(f"加载向量化模型: {settings.EMBEDDING_MODEL_NAME}")
    _model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    logger.info(f"初始化ChromaDB: {settings.CHROMA_DB_DIR}")
    _chroma_client = chromadb.PersistentClient(path=str(settings.CHROMA_DB_DIR))
    _collection = _chroma_client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_NAME, metadata={"hnsw:space": "cosine"},
    )
    logger.info(f"ChromaDB集合已加载，当前数据量: {_collection.count()}")
    yield
    logger.info("服务关闭")


app = FastAPI(
    title="矿业数据聚合检索API",
    description="推理式RAG问答系统，支持时间/地区/矿种过滤和多源交叉验证",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """推理式自然语言问答接口."""
    if _model is None or _collection is None:
        raise HTTPException(status_code=503, detail="服务尚未就绪")

    try:
        filters = parse_metadata_filters(request.question)
        if request.commodity:
            filters["commodity"] = request.commodity
        if request.region:
            filters["country_or_region"] = request.region
        if request.source_type:
            filters["source_type"] = request.source_type
        time_filter = parse_time_filter(request.question)
        where_clause = build_chroma_where(filters)

        query_embedding = _model.encode([request.question], normalize_embeddings=True).tolist()

        query_params: dict = {
            "query_embeddings": query_embedding,
            "n_results": min(request.top_k * 3, 50),
            "include": ["documents", "metadatas", "distances"],
        }
        if where_clause:
            query_params["where"] = where_clause

        results = _collection.query(**query_params)

        query_results: List[QueryResult] = []
        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            documents = results["documents"][0] if results["documents"] else [""] * len(ids)
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)

            for doc, meta, dist in zip(documents, metadatas, distances):
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
                    snippet=snippet, score=score,
                ))

        query_results = query_results[:request.top_k]
        rag_results = [r.model_dump() for r in query_results]
        answer = generate_rag_answer(request.question, rag_results, _model)

        filters_applied: dict = {}
        if filters:
            filters_applied["metadata"] = filters
        if time_filter:
            filters_applied["time"] = time_filter

        return QueryResponse(
            question=request.question, answer=answer,
            results=query_results, total=len(query_results),
            filters_applied=filters_applied if filters_applied else None,
        )
    except Exception as e:
        logger.error(f"查询处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询处理失败: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    count = _collection.count() if _collection else 0
    return HealthResponse(status="ok", collection_count=count)


def main() -> None:
    import uvicorn
    uvicorn.run("serve.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=False)


if __name__ == "__main__":
    main()
