"""评估模块单元测试."""

from eval.run_eval import (
    compute_recall_at_k,
    compute_answer_faithfulness,
    compute_answer_relevance,
)


def test_recall_all_keywords_hit() -> None:
    """测试所有关键词命中."""
    results = [
        {"title": "LME铜价上涨", "snippet": "铜价格大幅上涨至72000元/吨"},
    ]
    recall = compute_recall_at_k(results, ["铜", "价格", "上涨"])
    assert recall == 1.0


def test_recall_partial_hit() -> None:
    """测试部分关键词命中."""
    results = [
        {"title": "矿业新闻", "snippet": "铜矿产量增加"},
    ]
    recall = compute_recall_at_k(results, ["铜", "锂", "镍"])
    assert 0 < recall < 1.0


def test_recall_no_hit() -> None:
    """测试无关键词命中."""
    results = [
        {"title": "铁矿新闻", "snippet": "铁矿石出口增长"},
    ]
    recall = compute_recall_at_k(results, ["铜", "锂"])
    assert recall == 0.0


def test_recall_empty_results() -> None:
    """测试空结果."""
    recall = compute_recall_at_k([], ["铜"])
    assert recall == 0.0


def test_recall_empty_keywords() -> None:
    """测试空关键词."""
    recall = compute_recall_at_k(
        [{"title": "test", "snippet": "test"}], []
    )
    assert recall == 0.0


def test_faithfulness_supported() -> None:
    """测试答案被证据支持."""
    results = [
        {
            "title": "LME铜价",
            "snippet": "LME铜价上涨至72000元/吨，较前日上涨500元",
        },
    ]
    answer = "LME铜价上涨至72000元/吨"
    faith = compute_answer_faithfulness(answer, results)
    assert faith > 0.5


def test_faithfulness_unsupported() -> None:
    """测试答案不被证据支持."""
    results = [
        {
            "title": "铁矿新闻",
            "snippet": "铁矿石出口增长，澳大利亚创历史新高",
        },
    ]
    answer = "锂矿价格大幅上涨，碳酸锂突破15万元/吨"
    faith = compute_answer_faithfulness(answer, results)
    assert faith < 0.5


def test_faithfulness_empty_answer() -> None:
    """测试空答案."""
    results = [{"title": "test", "snippet": "test"}]
    faith = compute_answer_faithfulness("", results)
    assert faith == 0.0


def test_faithfulness_empty_results() -> None:
    """测试空证据."""
    faith = compute_answer_faithfulness("铜价上涨", [])
    assert faith == 0.0


def test_relevance_high() -> None:
    """测试高相关性."""
    answer = "LME铜价上涨至72000元/吨，受电动汽车需求推动"
    expected_answer = "铜价近期上涨，LME报价72000元/吨"
    keywords = ["铜", "价格", "上涨", "LME"]
    rel = compute_answer_relevance(answer, expected_answer, keywords)
    assert rel >= 0.75


def test_relevance_low() -> None:
    """测试低相关性."""
    answer = "铁矿石出口创历史新高，澳大利亚表现强劲"
    expected_answer = "铜价近期走势分析"
    keywords = ["铜", "价格", "LME"]
    rel = compute_answer_relevance(answer, expected_answer, keywords)
    assert rel < 0.5


def test_relevance_empty_answer() -> None:
    """测试空答案相关性."""
    rel = compute_answer_relevance("", "期望答案", ["关键词"])
    assert rel == 0.0
