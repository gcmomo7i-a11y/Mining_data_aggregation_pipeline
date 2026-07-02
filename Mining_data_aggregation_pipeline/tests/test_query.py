"""API查询接口单元测试."""

import sys
from unittest.mock import MagicMock, patch

# 在导入serve.main之前mock模型和ChromaDB
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["chromadb"] = MagicMock()

from serve.main import (
    parse_time_filter,
    parse_metadata_filters,
    build_chroma_where,
    generate_rag_answer,
)


def test_parse_time_filter_7_days() -> None:
    """测试解析'近7天'."""
    result = parse_time_filter("近7天澳洲锂出口政策有何变化？")
    assert result is not None
    assert "start_date" in result
    assert "end_date" in result


def test_parse_time_filter_30_days() -> None:
    """测试解析'近30天'."""
    result = parse_time_filter("近30天铜价走势如何？")
    assert result is not None


def test_parse_time_filter_recent_week() -> None:
    """测试解析'最近一周'."""
    result = parse_time_filter("最近一周矿业新闻")
    assert result is not None


def test_parse_time_filter_no_time() -> None:
    """测试无时间表达的问题."""
    result = parse_time_filter("稀土行业政策有哪些？")
    assert result is None


def test_parse_metadata_filters_australia() -> None:
    """测试解析澳洲相关过滤."""
    filters = parse_metadata_filters("近7天澳洲锂出口政策有何变化？")
    assert filters.get("country_or_region") == "Australia"
    assert filters.get("commodity") == "lithium"


def test_parse_metadata_filters_china() -> None:
    """测试解析中国相关过滤."""
    filters = parse_metadata_filters("中国稀土集团发布了什么？")
    assert filters.get("country_or_region") == "China"


def test_parse_metadata_filters_copper() -> None:
    """测试解析铜相关过滤."""
    filters = parse_metadata_filters("铜价走势如何？")
    assert filters.get("commodity") == "copper"
    assert filters.get("source_type") == "price"


def test_parse_metadata_filters_policy() -> None:
    """测试解析政策类型过滤."""
    filters = parse_metadata_filters("工信部关于矿业有什么新规定？")
    assert filters.get("source_type") == "policy"


def test_parse_metadata_filters_lithium() -> None:
    """测试解析锂相关过滤."""
    filters = parse_metadata_filters("碳酸锂期货价格走势")
    assert filters.get("commodity") == "lithium"
    assert filters.get("source_type") == "price"


def test_build_chroma_where_single() -> None:
    """测试构建单条件where."""
    where = build_chroma_where({"commodity": "copper"})
    assert where == {"commodity": "copper"}


def test_build_chroma_where_multiple() -> None:
    """测试构建多条件where."""
    where = build_chroma_where({
        "commodity": "copper",
        "country_or_region": "China",
    })
    assert "$and" in where
    assert len(where["$and"]) == 2


def test_build_chroma_where_empty() -> None:
    """测试空过滤条件."""
    where = build_chroma_where({})
    assert where is None


def test_generate_rag_answer_with_results() -> None:
    """测试RAG答案生成（有结果）."""
    results = [
        {
            "title": "LME铜价行情",
            "source_type": "price",
            "snippet": "2026-07-01 LME铜价上涨至72000元/吨",
            "score": 0.85,
        }
    ]
    mock_model = MagicMock()
    answer = generate_rag_answer("铜价走势如何？", results, mock_model)
    assert "铜" in answer or "价格" in answer
    assert "参考来源" in answer


def test_generate_rag_answer_empty() -> None:
    """测试RAG答案生成（无结果）."""
    mock_model = MagicMock()
    answer = generate_rag_answer("测试问题", [], mock_model)
    assert "未找到" in answer
