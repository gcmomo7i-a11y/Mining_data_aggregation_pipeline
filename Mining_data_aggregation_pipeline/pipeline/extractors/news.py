"""矿业新闻采集器.

从 mining.com（HTML+RSS）、S&P Global Mining（RSS）采集矿业新闻数据。
"""

import time
from datetime import datetime
from typing import Any, List, Optional
from urllib.parse import urljoin

import feedparser
import requests
from bs4 import BeautifulSoup
from loguru import logger

from configs import settings
from pipeline.extractors.base import BaseExtractor
from pipeline.models import Commodity, MiningData, Region, SourceType
from pipeline.utils import HTMLCleaner, get_request_headers


class NewsExtractor(BaseExtractor):
    """矿业新闻采集器.

    数据源:
    - mining.com: HTML页面采集 + RSS feed
    - S&P Global Mining: RSS feed

    Example::

        extractor = NewsExtractor()
        data = extractor.extract()
    """

    BASE_URL: str = "https://www.mining.com"
    LIST_URL: str = "https://www.mining.com/category/news/"
    MINING_RSS: str = "https://www.mining.com/feed/"
    SP_GLOBAL_RSS: str = "https://www.spglobal.com/marketintelligence/en/rss-feed/metals-mining"

    def __init__(self, max_pages: int = 10) -> None:
        """初始化新闻采集器.

        Args:
            max_pages: 最大采集页数.
        """
        self._max_pages = max_pages
        self._cleaner = HTMLCleaner()
        self._session = requests.Session()

    @property
    def source_name(self) -> str:
        """数据源名称."""
        return "MiningNews"

    def extract(self) -> List[MiningData]:
        """执行新闻数据采集.

        依次尝试 RSS 采集和 HTML 页面采集，合并结果。

        Returns:
            采集到的MiningData列表.
        """
        all_data: List[MiningData] = []

        # 1. mining.com RSS
        rss_data = self._extract_rss(
            self.MINING_RSS, "mining.com", Region.GLOBAL
        )
        all_data.extend(rss_data)

        # 2. S&P Global Mining RSS
        sp_data = self._extract_rss(
            self.SP_GLOBAL_RSS, "S&P Global Mining", Region.GLOBAL
        )
        all_data.extend(sp_data)

        # 3. mining.com HTML 页面采集
        html_data = self._extract_html_pages()
        all_data.extend(html_data)

        self.log_progress(f"共采集新闻{len(all_data)}条")
        return all_data

    def _extract_rss(
        self, rss_url: str, source_site: str, region: Region
    ) -> List[MiningData]:
        """通过RSS feed采集新闻.

        Args:
            rss_url: RSS feed地址.
            source_site: 数据源站点名称.
            region: 国家/地区.

        Returns:
            采集到的MiningData列表.
        """
        data: List[MiningData] = []

        try:
            headers = get_request_headers()
            resp = self._session.get(rss_url, headers=headers, timeout=30)
            resp.raise_for_status()

            feed = feedparser.parse(resp.text)

            for entry in feed.entries:
                try:
                    title = entry.get("title", "").strip()
                    if not title:
                        continue

                    content = self._extract_rss_content(entry)
                    if not content or len(content.strip()) < 30:
                        content = entry.get("summary", title)

                    publish_date = self._parse_rss_date(entry)

                    commodity = self._infer_commodity(title + " " + content)

                    data.append(MiningData(
                        source_type=SourceType.NEWS,
                        title=title,
                        content=content.strip(),
                        publish_date=publish_date,
                        commodity=commodity,
                        country_or_region=region,
                        is_mock=False,
                        metadata={
                            "url": entry.get("link", ""),
                            "source_site": source_site,
                        },
                    ))
                except Exception as e:
                    logger.warning(f"解析RSS条目失败: {e}")
                    continue

            self.log_progress(f"RSS {source_site} 采集{len(data)}条")

        except Exception as e:
            logger.error(f"RSS采集失败 [{source_site}]: {e}")

        return data

    @staticmethod
    def _extract_rss_content(entry: Any) -> str:
        """从RSS条目中提取正文内容.

        Args:
            entry: feedparser解析的RSS条目.

        Returns:
            正文文本.
        """
        if hasattr(entry, "content") and entry.content:
            for content_item in entry.content:
                text = content_item.get("value", "")
                if text:
                    soup = BeautifulSoup(text, "html.parser")
                    return soup.get_text(separator="\n").strip()

        if hasattr(entry, "summary"):
            soup = BeautifulSoup(entry.summary, "html.parser")
            return soup.get_text(separator="\n").strip()

        return ""

    def _extract_html_pages(self) -> List[MiningData]:
        """通过HTML页面采集mining.com新闻.

        Returns:
            采集到的MiningData列表.
        """
        all_data: List[MiningData] = []

        for page in range(1, self._max_pages + 1):
            try:
                articles = self._fetch_list_page(page)
                if not articles:
                    self.log_progress(f"第{page}页无文章，停止采集")
                    break

                self.log_progress(f"第{page}页发现{len(articles)}篇文章")

                for article in articles:
                    try:
                        data = self._fetch_article_detail(article)
                        if data:
                            all_data.append(data)
                    except Exception as e:
                        logger.warning(f"采集文章详情失败: {e}")
                    time.sleep(settings.CRAWL_DELAY)

            except Exception as e:
                logger.error(f"采集第{page}页失败: {e}")
                continue

        return all_data

    def _fetch_list_page(self, page: int) -> List[dict]:
        """抓取列表页，提取文章基本信息.

        Args:
            page: 页码（从1开始）.

        Returns:
            文章信息字典列表.
        """
        url = self.LIST_URL if page == 1 else f"{self.LIST_URL}page/{page}/"
        headers = get_request_headers()

        resp = self._session.get(url, headers=headers, timeout=30)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        articles: List[dict] = []

        article_tags = (
            soup.select("article")
            or soup.select("div.post-item")
            or soup.select("div.news-item")
            or soup.select("div.entry")
        )

        for tag in article_tags:
            link_tag = tag.find("a", href=True)
            if not link_tag:
                continue

            title = link_tag.get_text(strip=True)
            article_url = urljoin(self.BASE_URL, link_tag["href"])

            if not title or not article_url:
                continue

            date = self._extract_date(tag)

            articles.append({
                "title": title,
                "url": article_url,
                "date": date,
            })

        return articles

    @staticmethod
    def _extract_date(tag: BeautifulSoup) -> Optional[str]:
        """从文章标签中提取日期.

        Args:
            tag: 文章HTML标签.

        Returns:
            日期字符串或None.
        """
        date_tag = (
            tag.find("time")
            or tag.find(class_=lambda c: c and "date" in c.lower() if c else False)
            or tag.find(class_=lambda c: c and "time" in c.lower() if c else False)
        )
        if date_tag:
            if date_tag.has_attr("datetime"):
                return date_tag["datetime"]
            return date_tag.get_text(strip=True)
        return None

    def _fetch_article_detail(self, article_info: dict) -> Optional[MiningData]:
        """抓取文章详情页，提取正文内容.

        Args:
            article_info: 文章基本信息.

        Returns:
            MiningData实例或None.
        """
        headers = get_request_headers()

        resp = self._session.get(
            article_info["url"], headers=headers, timeout=30
        )
        resp.raise_for_status()

        content = self._cleaner.extract_main_content(
            resp.text,
            content_selector="div.entry-content, article, div.post-content, main",
        )

        if not content or len(content.strip()) < 50:
            return None

        publish_date = self._parse_date(article_info.get("date"))
        commodity = self._infer_commodity(
            article_info["title"] + " " + content
        )

        return MiningData(
            source_type=SourceType.NEWS,
            title=article_info["title"],
            content=content.strip(),
            publish_date=publish_date,
            commodity=commodity,
            country_or_region=Region.GLOBAL,
            is_mock=False,
            metadata={
                "url": article_info["url"],
                "source_site": "mining.com",
            },
        )

    @staticmethod
    def _infer_commodity(text: str) -> Commodity:
        """从文本推断关联矿产品种.

        Args:
            text: 标题+正文文本.

        Returns:
            推断的Commodity枚举值.
        """
        text_lower = text.lower()
        mapping = {
            Commodity.COPPER: ["copper", "铜"],
            Commodity.ZINC: ["zinc", "锌"],
            Commodity.NICKEL: ["nickel", "镍"],
            Commodity.LITHIUM: ["lithium", "锂"],
            Commodity.IRON_ORE: ["iron ore", "iron-ore", "铁矿"],
            Commodity.ALUMINIUM: ["aluminium", "aluminum", "铝"],
            Commodity.LEAD: ["lead", "铅"],
            Commodity.TIN: ["tin", "锡"],
            Commodity.GOLD: ["gold", "黄金"],
            Commodity.SILVER: ["silver", "白银"],
            Commodity.RARE_EARTH: ["rare earth", "稀土"],
        }
        for commodity, keywords in mapping.items():
            for kw in keywords:
                if kw in text_lower:
                    return commodity
        return Commodity.OTHER

    @staticmethod
    def _parse_rss_date(entry: Any) -> datetime:
        """解析RSS条目日期.

        Args:
            entry: feedparser条目.

        Returns:
            datetime对象.
        """
        for attr in ("published_parsed", "updated_parsed", "created_parsed"):
            time_struct = getattr(entry, attr, None)
            if time_struct:
                try:
                    return datetime(*time_struct[:6])
                except (TypeError, ValueError):
                    continue

        return datetime.now()

    @staticmethod
    def _parse_date(date_str: Optional[str]) -> datetime:
        """解析日期字符串为datetime对象.

        Args:
            date_str: 日期字符串.

        Returns:
            datetime对象.
        """
        if not date_str:
            return datetime.now()

        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%B %d, %Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(date_str[:19], fmt)
            except (ValueError, TypeError):
                continue

        return datetime.now()

    @staticmethod
    def _generate_simulated_news(count: int) -> List[MiningData]:
        """生成模拟新闻数据.

        Args:
            count: 需要生成的数据条数.

        Returns:
            模拟的MiningData列表.
        """
        from dateutil.relativedelta import relativedelta

        templates: List[dict] = [
            {
                "title": "全球铜矿产量预计增长，智利新项目投产",
                "content": (
                    "据最新行业报告，全球铜矿产量预计将在2026年实现显著增长。"
                    "智利国家矿业公司宣布其位于阿塔卡马沙漠的新铜矿项目已正式投产，"
                    "预计年产能将达到15万吨。分析人士指出，随着电动汽车和可再生能源产业"
                    "对铜需求的持续攀升，铜矿供应的增加对稳定市场至关重要。"
                ),
                "commodity": Commodity.COPPER,
                "region": Region.GLOBAL,
            },
            {
                "title": "澳大利亚铁矿石出口创历史新高",
                "content": (
                    "澳大利亚统计局最新数据显示，该国铁矿石出口量在上季度达到历史新高。"
                    "皮尔巴拉地区的大型矿场持续满负荷运转，主要出口目的地为中国和日本。"
                    "行业专家认为，尽管全球钢铁需求增速放缓，但澳大利亚凭借低开采成本"
                    "仍保持强劲的出口竞争力。"
                ),
                "commodity": Commodity.IRON_ORE,
                "region": Region.AUSTRALIA,
            },
            {
                "title": "印尼镍矿出口政策调整引发市场关注",
                "content": (
                    "印尼政府宣布将对镍矿出口政策进行调整，加强了对原矿出口的限制力度。"
                    "作为全球最大的镍矿储藏国，印尼的政策变动直接影响国际镍价走势。"
                    "下游电池制造商面临原材料成本上升的压力，部分企业已开始寻求替代供应渠道。"
                ),
                "commodity": Commodity.NICKEL,
                "region": Region.GLOBAL,
            },
            {
                "title": "非洲锂矿开发加速，多国竞逐新能源矿产",
                "content": (
                    "非洲多国正加速锂矿资源的勘探与开发。津巴布韦、纳米比亚和马里等国"
                    "近期相继发放了新的锂矿开采许可证。中国和澳大利亚矿业公司积极布局，"
                    "投资总额已超过50亿美元。锂作为电动汽车电池的核心原材料，"
                    "其战略价值日益凸显。"
                ),
                "commodity": Commodity.LITHIUM,
                "region": Region.GLOBAL,
            },
            {
                "title": "加拿大黄金矿业公司宣布重大收购计划",
                "content": (
                    "加拿大知名黄金矿业公司Barrick Gold宣布将收购一家中型金矿企业，"
                    "交易金额预计达到32亿美元。此次收购将使Barrick在北美地区的黄金产量"
                    "提升约20%。市场分析认为，这反映了大型矿企在金价高企背景下"
                    "积极扩充资源储备的战略意图。"
                ),
                "commodity": Commodity.GOLD,
                "region": Region.GLOBAL,
            },
            {
                "title": "秘鲁铜矿社区抗议活动影响生产",
                "content": (
                    "秘鲁南部安第斯山脉地区的铜矿再次爆发社区抗议活动，当地居民要求矿业公司"
                    "增加社区投资和就业机会。抗议已导致至少两座大型铜矿暂时停产，"
                    "预计影响月产量约3万吨。秘鲁是全球第二大铜生产国，"
                    "此类社会冲突频发令国际市场担忧供应稳定性。"
                ),
                "commodity": Commodity.COPPER,
                "region": Region.GLOBAL,
            },
            {
                "title": "稀土分离技术突破降低环境成本",
                "content": (
                    "中国科学院研究团队宣布在稀土分离技术上取得重大突破，"
                    "新型萃取工艺可将废水排放量减少80%以上。该技术已在江西稀土产业园区"
                    "完成中试，有望在2027年前实现工业化应用。这一进展对全球稀土产业链"
                    "的绿色转型具有重要意义。"
                ),
                "commodity": Commodity.RARE_EARTH,
                "region": Region.CHINA,
            },
            {
                "title": "澳大利亚锂出口政策面临重大调整",
                "content": (
                    "澳大利亚工业科学资源部(DISR)发布最新关键矿产战略文件，"
                    "提出将加强对锂等关键矿产出口的监管框架。新战略强调需要在国家安全"
                    "和经济利益之间取得平衡，建议对锂精矿出口实施更严格的审查机制。"
                    "澳大利亚是全球最大的锂出口国，此举引发国际市场广泛关注。"
                ),
                "commodity": Commodity.LITHIUM,
                "region": Region.AUSTRALIA,
            },
            {
                "title": "全球锌矿供应紧张推动价格上行",
                "content": (
                    "受多座大型锌矿产能下降影响，全球锌矿供应出现紧张局面。"
                    "国际铅锌研究小组数据显示，今年全球锌矿产量同比下降3.2%，"
                    "而需求则保持温和增长。供给缺口推动LME锌价持续走高，"
                    "部分贸易商开始增加库存。"
                ),
                "commodity": Commodity.ZINC,
                "region": Region.GLOBAL,
            },
            {
                "title": "中国深海采矿技术试验取得重要进展",
                "content": (
                    "中国大洋矿产资源研究开发协会宣布，自主研发的深海采矿车"
                    "在太平洋CC区完成4000米级海试。试验验证了多金属结核采集、"
                    "输送和水面支持等关键环节的技术可行性。"
                    "中国正在积极推动国际海底管理局制定深海采矿开发规章。"
                ),
                "commodity": Commodity.RARE_EARTH,
                "region": Region.CHINA,
            },
            {
                "title": "S&P Global: 全球矿业并购交易额创新高",
                "content": (
                    "S&P Global Market Intelligence最新报告显示，"
                    "2026年上半年全球矿业并购交易总额达到创纪录的1250亿美元。"
                    "铜和锂矿资产成为最受追捧的标的，交易占比分别达到28%和22%。"
                    "大型矿企积极通过并购方式补充资源储备，以应对能源转型带来的需求增长。"
                ),
                "commodity": Commodity.OTHER,
                "region": Region.GLOBAL,
            },
            {
                "title": "澳大利亚关键矿产战略更新：锂镍为重点扶持对象",
                "content": (
                    "澳大利亚联邦政府发布更新版关键矿产战略，将锂和镍列为重点扶持对象。"
                    "战略提出在未来五年内投入40亿澳元支持关键矿产下游加工产业发展，"
                    "鼓励在国内建设电池级氢氧化锂和硫酸镍精炼产能。"
                    "此举旨在提升澳大利亚在全球电池供应链中的地位。"
                ),
                "commodity": Commodity.LITHIUM,
                "region": Region.AUSTRALIA,
            },
        ]

        data: List[MiningData] = []
        today = datetime.now()
        idx = 0

        for days_back in range(365):
            if idx >= count:
                break
            pub_date = today - relativedelta(days=days_back)
            for tmpl in templates:
                if idx >= count:
                    break
                date_str = pub_date.strftime("%Y-%m-%d")
                title = f"{tmpl['title']} ({date_str})"
                content = (
                    f"{tmpl['content']} "
                    f"（发布日期: {date_str}，来源: 模拟数据）"
                )

                data.append(MiningData(
                    source_type=SourceType.NEWS,
                    title=title,
                    content=content,
                    publish_date=pub_date,
                    commodity=tmpl["commodity"],
                    country_or_region=tmpl["region"],
                    is_mock=True,
                    metadata={
                        "url": f"https://www.mining.com/news/simulated-{idx:04d}",
                        "source_site": "mining.com",
                    },
                ))
                idx += 1

        return data
