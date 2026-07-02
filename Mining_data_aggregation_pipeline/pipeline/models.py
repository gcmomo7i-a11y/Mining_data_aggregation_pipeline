"""核心数据模型定义.

定义矿业数据的枚举类型和Pydantic模型，作为全项目的数据契约。
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """数据源类型枚举.

    Attributes:
        NEWS: 矿业新闻数据.
        POLICY: 政策法规数据.
        PRICE: 价格行情数据.
    """

    NEWS = "news"
    POLICY = "policy"
    PRICE = "price"


class Commodity(str, Enum):
    """矿产品种枚举.

    Attributes:
        COPPER: 铜.
        ZINC: 锌.
        NICKEL: 镍.
        LITHIUM: 锂.
        IRON_ORE: 铁矿石.
        ALUMINIUM: 铝.
        LEAD: 铅.
        TIN: 锡.
        GOLD: 黄金.
        SILVER: 白银.
        RARE_EARTH: 稀土.
        OTHER: 其他.
    """

    COPPER = "copper"
    ZINC = "zinc"
    NICKEL = "nickel"
    LITHIUM = "lithium"
    IRON_ORE = "iron_ore"
    ALUMINIUM = "aluminium"
    LEAD = "lead"
    TIN = "tin"
    GOLD = "gold"
    SILVER = "silver"
    RARE_EARTH = "rare_earth"
    OTHER = "other"


class Region(str, Enum):
    """国家/地区枚举.

    Attributes:
        CHINA: 中国.
        AUSTRALIA: 澳大利亚.
        GLOBAL: 全球.
        OTHER: 其他.
    """

    CHINA = "China"
    AUSTRALIA = "Australia"
    GLOBAL = "Global"
    OTHER = "Other"


class MiningData(BaseModel):
    """矿业数据统一模型.

    所有采集器输出的数据均遵循此模型，确保数据格式一致。

    Attributes:
        id: 数据唯一标识，自动生成UUID4.
        source_type: 数据源类型（新闻/政策/价格）.
        title: 数据标题.
        content: 数据正文内容.
        publish_date: 发布日期.
        commodity: 关联矿产品种.
        country_or_region: 关联国家或地区.
        is_mock: 是否为模拟数据.
        metadata: 扩展元数据，存储各采集器的附加信息.
        embedding: 向量化表示，由sentence-transformers生成，入库前为None.
    """

    id: UUID = Field(default_factory=uuid4, description="数据唯一标识")
    source_type: SourceType = Field(description="数据源类型")
    title: str = Field(min_length=1, description="数据标题")
    content: str = Field(min_length=1, description="数据正文内容")
    publish_date: datetime = Field(description="发布日期")
    commodity: Commodity = Field(
        default=Commodity.OTHER, description="关联矿产品种"
    )
    country_or_region: Region = Field(
        default=Region.GLOBAL, description="关联国家或地区"
    )
    is_mock: bool = Field(default=False, description="是否为模拟数据")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="扩展元数据"
    )
    embedding: Optional[list[float]] = Field(
        default=None, description="向量化表示"
    )

    model_config = {"from_attributes": True}
