"""数据模型单元测试."""

import uuid
from datetime import datetime

from pipeline.models import Commodity, MiningData, Region, SourceType


def test_source_type_values() -> None:
    """测试SourceType枚举值."""
    assert SourceType.NEWS.value == "news"
    assert SourceType.POLICY.value == "policy"
    assert SourceType.PRICE.value == "price"


def test_commodity_values() -> None:
    """测试Commodity枚举值."""
    assert Commodity.COPPER.value == "copper"
    assert Commodity.LITHIUM.value == "lithium"
    assert Commodity.IRON_ORE.value == "iron_ore"
    assert Commodity.RARE_EARTH.value == "rare_earth"


def test_region_values() -> None:
    """测试Region枚举值."""
    assert Region.CHINA.value == "China"
    assert Region.AUSTRALIA.value == "Australia"
    assert Region.GLOBAL.value == "Global"


def test_mining_data_creation() -> None:
    """测试MiningData创建."""
    data = MiningData(
        source_type=SourceType.NEWS,
        title="测试新闻",
        content="测试内容",
        publish_date=datetime.now(),
        commodity=Commodity.COPPER,
        country_or_region=Region.GLOBAL,
    )
    assert data.source_type == SourceType.NEWS
    assert data.title == "测试新闻"
    assert data.content == "测试内容"
    assert data.commodity == Commodity.COPPER
    assert data.country_or_region == Region.GLOBAL
    assert data.is_mock is False
    assert data.embedding is None
    assert isinstance(data.id, uuid.UUID)


def test_mining_data_mock_flag() -> None:
    """测试is_mock字段."""
    mock_data = MiningData(
        source_type=SourceType.PRICE,
        title="模拟价格",
        content="模拟内容",
        publish_date=datetime.now(),
        is_mock=True,
    )
    assert mock_data.is_mock is True

    real_data = MiningData(
        source_type=SourceType.PRICE,
        title="真实价格",
        content="真实内容",
        publish_date=datetime.now(),
        is_mock=False,
    )
    assert real_data.is_mock is False


def test_mining_data_defaults() -> None:
    """测试MiningData默认值."""
    data = MiningData(
        source_type=SourceType.NEWS,
        title="标题",
        content="内容",
        publish_date=datetime.now(),
    )
    assert data.commodity == Commodity.OTHER
    assert data.country_or_region == Region.GLOBAL
    assert data.is_mock is False
    assert data.metadata == {}
    assert data.embedding is None


def test_mining_data_validation() -> None:
    """测试MiningData校验."""
    try:
        MiningData(
            source_type=SourceType.NEWS,
            title="",
            content="内容",
            publish_date=datetime.now(),
        )
        assert False, "空标题应该校验失败"
    except Exception:
        pass

    try:
        MiningData(
            source_type=SourceType.NEWS,
            title="标题",
            content="",
            publish_date=datetime.now(),
        )
        assert False, "空内容应该校验失败"
    except Exception:
        pass
