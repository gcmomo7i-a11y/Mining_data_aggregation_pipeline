"""采集器抽象基类.

定义统一的采集器接口，所有采集器必须实现extract方法。
"""

from abc import ABC, abstractmethod
from typing import List

from loguru import logger

from pipeline.models import MiningData


class BaseExtractor(ABC):
    """采集器抽象基类.

    所有数据源采集器必须继承此类并实现extract方法，
    确保返回统一格式的MiningData列表。

    Example::

        class MyExtractor(BaseExtractor):
            def extract(self) -> List[MiningData]:
                # 实现采集逻辑
                return [...]
    """

    @abstractmethod
    def extract(self) -> List[MiningData]:
        """执行数据采集.

        Returns:
            采集到的MiningData列表.

        Raises:
            NotImplementedError: 子类未实现此方法.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def source_name(self) -> str:
        """数据源名称，用于日志标识.

        Returns:
            数据源名称字符串.
        """
        raise NotImplementedError

    def log_progress(self, message: str) -> None:
        """记录采集进度日志.

        Args:
            message: 日志消息.
        """
        logger.info(f"[{self.source_name}] {message}")
