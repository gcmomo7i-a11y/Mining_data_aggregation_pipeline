"""ETL流程单元测试."""

from datetime import datetime, timedelta

from pipeline.etl import ETLStats, ETLPipeline
from pipeline.models import Commodity, MiningData, Region, SourceType


def _make_data(
    n: int, source_type: SourceType, is_mock: bool = False
) -> list[MiningData]:
    """生成测试用MiningData列表.

    Args:
        n: 数量.
        source_type: 数据源类型.
        is_mock: 是否为模拟数据.

    Returns:
        测试数据列表.
    """
    return [
        MiningData(
            source_type=source_type,
            title=f"测试{i}",
            content=f"测试内容{i}",
            publish_date=datetime.now() - timedelta(days=i % 30),
            is_mock=is_mock,
        )
        for i in range(n)
    ]


def test_deduplicate_removes_duplicates() -> None:
    """测试去重逻辑移除重复数据."""
    data = _make_data(5, SourceType.NEWS)
    # 添加重复项
    duplicate = MiningData(
        source_type=SourceType.NEWS,
        title=data[0].title,
        content="不同内容",
        publish_date=data[0].publish_date,
    )
    data.append(duplicate)

    result = ETLPipeline._deduplicate(data)
    assert len(result) == 5  # 重复项被去除


def test_deduplicate_preserves_unique() -> None:
    """测试去重逻辑保留唯一数据."""
    data = _make_data(10, SourceType.NEWS)
    result = ETLPipeline._deduplicate(data)
    assert len(result) == 10


def test_deduplicate_same_title_different_date() -> None:
    """测试标题相同但日期不同的数据不会被去重."""
    data1 = MiningData(
        source_type=SourceType.NEWS,
        title="同一标题",
        content="内容1",
        publish_date=datetime(2026, 7, 1),
    )
    data2 = MiningData(
        source_type=SourceType.NEWS,
        title="同一标题",
        content="内容2",
        publish_date=datetime(2026, 7, 2),
    )

    result = ETLPipeline._deduplicate([data1, data2])
    assert len(result) == 2


def test_etl_stats_record_raw() -> None:
    """测试ETLStats记录原始数据统计."""
    stats = ETLStats()
    data = _make_data(10, SourceType.NEWS, is_mock=False) + _make_data(
        5, SourceType.NEWS, is_mock=True
    )
    # 手动设置is_mock
    for d in data[10:]:
        d.is_mock = True

    stats.record_raw("news", data)
    assert stats.raw_counts["news"] == 15
    assert stats.real_counts["news"] == 10
    assert stats.mock_counts["news"] == 5


def test_etl_stats_validate_warnings() -> None:
    """测试ETLStats数据量校验."""
    stats = ETLStats()
    stats.raw_counts["news"] = 50
    stats.real_counts["news"] = 10
    stats.mock_counts["news"] = 40

    warnings = stats.validate_warnings()
    assert any("不足" in w or "真实数据" in w for w in warnings)


def test_etl_stats_validate_pass() -> None:
    """测试ETLStats校验通过."""
    stats = ETLStats()
    stats.raw_counts["news"] = 250
    stats.real_counts["news"] = 250
    stats.mock_counts["news"] = 0
    stats.recent_real_counts["news"] = 250

    warnings = stats.validate_warnings()
    assert not any("不足" in w for w in warnings)


def test_etl_stats_summary() -> None:
    """测试ETLStats摘要输出."""
    stats = ETLStats()
    data = _make_data(5, SourceType.PRICE)
    stats.record_raw("price", data)
    stats.deduped_count = 5
    stats.loaded_count = 5

    summary = stats.summary()
    assert "price" in summary
    assert "5" in summary
