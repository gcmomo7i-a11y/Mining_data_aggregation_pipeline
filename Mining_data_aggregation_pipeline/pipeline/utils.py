"""公共工具函数.

提供HTML清洗、哈希去重等通用工具。
"""

import hashlib
import random
import re
from typing import Optional

from bs4 import BeautifulSoup
from loguru import logger

from configs import settings


class HTMLCleaner:
    """HTML内容清洗器.

    去除script/style等无关标签，提取主要文本内容。

    Example::

        cleaner = HTMLCleaner()
        text = cleaner.clean("<html><script>...</script><p>Hello</p></html>")
        # "Hello"
    """

    REMOVE_TAGS: list[str] = [
        "script", "style", "noscript", "iframe",
        "nav", "footer", "header", "aside",
    ]

    def clean(self, html: str) -> str:
        """清洗HTML，返回纯文本.

        Args:
            html: 原始HTML字符串.

        Returns:
            清洗后的纯文本，去除多余空白.
        """
        soup = BeautifulSoup(html, "html.parser")

        for tag_name in self.REMOVE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        text = soup.get_text(separator="\n")
        text = self._normalize_whitespace(text)
        return text.strip()

    def extract_main_content(
        self, html: str, content_selector: Optional[str] = None
    ) -> str:
        """提取页面主要内容区域.

        Args:
            html: 原始HTML字符串.
            content_selector: CSS选择器，用于定位主要内容区域.

        Returns:
            提取的主要文本内容.
        """
        soup = BeautifulSoup(html, "html.parser")

        if content_selector:
            main = soup.select_one(content_selector)
            if main:
                return self.clean(str(main))

        return self.clean(html)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """规范化空白字符.

        Args:
            text: 待处理文本.

        Returns:
            去除多余空行和空格的文本.
        """
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text


def compute_dedup_hash(title: str, publish_date: str) -> str:
    """计算去重哈希值.

    基于 title + publish_date 生成唯一哈希，用于数据去重。

    Args:
        title: 数据标题.
        publish_date: 发布日期字符串.

    Returns:
        SHA256哈希字符串.
    """
    raw = f"{title.strip()}|{publish_date.strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_random_user_agent() -> str:
    """从User-Agent池中随机选取一个.

    Returns:
        随机User-Agent字符串.
    """
    return random.choice(settings.USER_AGENTS)


def get_request_headers() -> dict[str, str]:
    """构建带随机User-Agent的请求头.

    Returns:
        包含User-Agent的HTTP请求头字典.
    """
    return {
        "User-Agent": get_random_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    }
