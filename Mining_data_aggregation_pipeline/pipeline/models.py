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


class MiningData(BaseModel):
    """矿业数据统一模型.

    所有采集器输出的数据均遵循此模型，确保数据格式一致。

    Attributes:
        id: 数据唯一标识，自动生成UUID4.
        source_type: 数据源类型（新闻/政策/价格）.
        title: 数据标题.
        content: 数据正文内容.
        publish_date: 发布日期.
        metadata: 扩展元数据，存储各采集器的附加信息.
        embedding: 向量化表示，由sentence-transformers生成，入库前为None.
    """

    id: UUID = Field(default_factory=uuid4, description="数据唯一标识")
    source_type: SourceType = Field(description="数据源类型")
    title: str = Field(min_length=1, description="数据标题")
    content: str = Field(min_length=1, description="数据正文内容")
    publish_date: datetime = Field(description="发布日期")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="扩展元数据"
    )
    embedding: Optional[list[float]] = Field(
        default=None, description="向量化表示"
    )

    model_config = {"from_attributes": True}
