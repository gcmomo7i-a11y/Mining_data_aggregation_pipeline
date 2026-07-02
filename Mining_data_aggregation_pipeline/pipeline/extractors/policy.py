"""政策法规采集器.

从中国稀土集团、自然资源部、工信部、澳洲DISR采集矿业政策数据。
"""

import time
from datetime import datetime
from typing import List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from loguru import logger

from configs import settings
from pipeline.extractors.base import BaseExtractor
from pipeline.models import Commodity, MiningData, Region, SourceType
from pipeline.utils import HTMLCleaner, get_request_headers


class PolicyExtractor(BaseExtractor):
    """政策法规采集器.

    数据源:
    - 中国稀土集团官网
    - 自然资源部
    - 工信部
    - 澳洲DISR Critical Minerals Strategy

    Example::

        extractor = PolicyExtractor()
        data = extractor.extract()
    """

    SITES: List[dict] = [
        {
            "name": "中国稀土集团",
            "base_url": "https://www.cregroup.com.cn",
            "list_urls": [
                "https://www.cregroup.com.cn/newsList/index.html",
                "https://www.cregroup.com.cn/xwzx/index.html",
            ],
            "region": Region.CHINA,
            "commodity": Commodity.RARE_EARTH,
        },
        {
            "name": "自然资源部",
            "base_url": "https://www.mnr.gov.cn",
            "list_urls": [
                "https://www.mnr.gov.cn/gk/zcwj/",
            ],
            "region": Region.CHINA,
            "commodity": Commodity.OTHER,
        },
        {
            "name": "工信部",
            "base_url": "https://www.miit.gov.cn",
            "list_urls": [
                "https://www.miit.gov.cn/jgsj/zzgs/wjfb/index.html",
            ],
            "region": Region.CHINA,
            "commodity": Commodity.OTHER,
        },
        {
            "name": "澳洲DISR",
            "base_url": "https://www.industry.gov.au",
            "list_urls": [
                "https://www.industry.gov.au/policies-and-initiatives/critical-minerals-strategy",
            ],
            "region": Region.AUSTRALIA,
            "commodity": Commodity.LITHIUM,
        },
    ]

    def __init__(self, max_pages: int = 5) -> None:
        """初始化政策采集器.

        Args:
            max_pages: 每个数据源最大采集页数.
        """
        self._max_pages = max_pages
        self._cleaner = HTMLCleaner()
        self._session = requests.Session()

    @property
    def source_name(self) -> str:
        """数据源名称."""
        return "MiningPolicy"

    def extract(self) -> List[MiningData]:
        """执行政策数据采集.

        Returns:
            采集到的MiningData列表.
        """
        all_data: List[MiningData] = []

        for site in self.SITES:
            for list_url in site["list_urls"]:
                try:
                    data = self._extract_from_site(
                        base_url=site["base_url"],
                        list_url=list_url,
                        site_name=site["name"],
                        region=site["region"],
                        commodity=site["commodity"],
                    )
                    all_data.extend(data)
                except Exception as e:
                    logger.error(f"采集{site['name']}失败: {e}")

        self.log_progress(f"共采集政策{len(all_data)}条")
        return all_data

    def _extract_from_site(
        self,
        base_url: str,
        list_url: str,
        site_name: str,
        region: Region,
        commodity: Commodity,
    ) -> List[MiningData]:
        """从指定站点采集政策数据.

        Args:
            base_url: 站点基础URL.
            list_url: 列表页URL.
            site_name: 站点名称.
            region: 国家/地区.
            commodity: 关联矿产品种.

        Returns:
            采集到的MiningData列表.
        """
        site_data: List[MiningData] = []

        for page in range(1, self._max_pages + 1):
            try:
                page_url = (
                    list_url
                    if page == 1
                    else self._build_page_url(list_url, page)
                )
                articles = self._fetch_list_page(base_url, page_url)

                if not articles:
                    self.log_progress(f"{site_name} 第{page}页无文章，停止")
                    break

                self.log_progress(
                    f"{site_name} 第{page}页发现{len(articles)}条政策"
                )

                for article in articles:
                    try:
                        data = self._fetch_article_detail(
                            article, base_url, site_name, region, commodity
                        )
                        if data:
                            site_data.append(data)
                    except Exception as e:
                        logger.warning(f"采集政策详情失败: {e}")
                    time.sleep(settings.CRAWL_DELAY)

            except Exception as e:
                logger.error(f"采集{site_name}第{page}页失败: {e}")
                continue

        return site_data

    def _fetch_list_page(
        self, base_url: str, page_url: str
    ) -> List[dict]:
        """抓取政策列表页.

        Args:
            base_url: 站点基础URL.
            page_url: 列表页URL.

        Returns:
            政策信息字典列表.
        """
        headers = get_request_headers()

        resp = self._session.get(page_url, headers=headers, timeout=30)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        articles: List[dict] = []

        # 适配多种页面结构
        link_tags = (
            soup.select("ul.list li a[href]")
            or soup.select("div.list-item a[href]")
            or soup.select("div.news-list a[href]")
            or soup.select("article a[href]")
            or soup.select("div.view-content a[href]")
        )

        seen_urls: set[str] = set()

        for link in link_tags:
            href = link.get("href", "")
            if not href or href == "#":
                continue

            full_url = urljoin(base_url, href)
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            title = link.get_text(strip=True)
            if not title or len(title) < 4:
                continue

            date = self._extract_date_from_link(link)

            articles.append({
                "title": title,
                "url": full_url,
                "date": date,
            })

        return articles

    @staticmethod
    def _extract_date_from_link(link_tag: BeautifulSoup) -> Optional[str]:
        """从链接标签的父元素中提取日期.

        Args:
            link_tag: 链接标签.

        Returns:
            日期字符串或None.
        """
        parent = link_tag.parent
        if parent:
            date_span = parent.find(
                "span",
                class_=lambda c: c and "date" in c.lower() if c else False,
            ) or parent.find(
                "span",
                class_=lambda c: c and "time" in c.lower() if c else False,
            )
            if date_span:
                return date_span.get_text(strip=True)

        if link_tag.has_attr("title"):
            title = link_tag["title"]
            for fmt in ("%Y-%m-%d", "%Y.%m.%d"):
                try:
                    datetime.strptime(title[:10], fmt)
                    return title[:10]
                except ValueError:
                    continue

        return None

    def _fetch_article_detail(
        self,
        article_info: dict,
        base_url: str,
        site_name: str,
        region: Region,
        commodity: Commodity,
    ) -> Optional[MiningData]:
        """抓取政策详情页.

        Args:
            article_info: 政策基本信息.
            base_url: 站点基础URL.
            site_name: 站点名称.
            region: 国家/地区.
            commodity: 关联矿产品种.

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
            content_selector=(
                "div.article-content, div.content, "
                "div.TRS_Editor, div.page_content, article, main"
            ),
        )

        if not content or len(content.strip()) < 30:
            return None

        publish_date = self._parse_date(article_info.get("date"))

        return MiningData(
            source_type=SourceType.POLICY,
            title=article_info["title"],
            content=content.strip(),
            publish_date=publish_date,
            commodity=commodity,
            country_or_region=region,
            is_mock=False,
            metadata={
                "url": article_info["url"],
                "source_site": site_name,
            },
        )

    @staticmethod
    def _build_page_url(list_url: str, page: int) -> str:
        """构建分页URL.

        Args:
            list_url: 列表页基础URL.
            page: 页码.

        Returns:
            带页码的URL.
        """
        if list_url.endswith("/"):
            return f"{list_url}index_{page}.html"
        base = list_url.rsplit(".", 1)
        if len(base) == 2:
            return f"{base[0]}_{page}.{base[1]}"
        return f"{list_url}?page={page}"

    @staticmethod
    def _parse_date(date_str: Optional[str]) -> datetime:
        """解析日期字符串.

        Args:
            date_str: 日期字符串.

        Returns:
            datetime对象.
        """
        if not date_str:
            return datetime.now()

        for fmt in (
            "%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d",
            "%Y年%m月%d日", "%Y-%m-%dT%H:%M:%S",
            "%d %B %Y", "%B %d, %Y",
        ):
            try:
                return datetime.strptime(date_str[:19].strip(), fmt)
            except (ValueError, TypeError):
                continue

        return datetime.now()

    @staticmethod
    def _generate_simulated_policies(count: int) -> List[MiningData]:
        """生成模拟政策数据.

        Args:
            count: 需要生成的数据条数.

        Returns:
            模拟的MiningData列表.
        """
        from dateutil.relativedelta import relativedelta

        templates: List[dict] = [
            {
                "title": "关于加强稀土行业管理的通知",
                "content": (
                    "为规范稀土行业秩序，促进产业健康发展，现就加强稀土行业管理"
                    "有关事项通知如下：一、严格控制稀土开采总量，严格执行年度开采计划；"
                    "二、加强稀土出口管理，完善出口配额制度；三、推进稀土产业整合，"
                    "淘汰落后产能；四、加大环保监管力度，严肃查处违法违规开采行为。"
                ),
                "source_site": "国务院",
                "commodity": Commodity.RARE_EARTH,
                "region": Region.CHINA,
            },
            {
                "title": "矿产资源开发利用方案审查办法",
                "content": (
                    "为规范矿产资源开发利用方案审查工作，提高矿产资源开发利用效率，"
                    "根据《矿产资源法》及相关法规，制定本办法。审查内容包括："
                    "矿山设计规模与资源储量匹配性、开采工艺技术先进性、"
                    "综合利用方案合理性、生态环境保护措施可行性等。"
                ),
                "source_site": "自然资源部",
                "commodity": Commodity.OTHER,
                "region": Region.CHINA,
            },
            {
                "title": "有色金属行业碳达峰实施方案",
                "content": (
                    "为贯彻落实碳达峰碳中和战略目标，推动有色金属行业绿色低碳转型，"
                    "特制定本实施方案。主要目标：到2030年，有色金属行业碳排放实现达峰；"
                    "到2040年，电解铝使用可再生能源比例提升至50%以上；"
                    "到2060年，实现碳中和。重点任务包括优化能源结构、"
                    "推广低碳技术、发展循环经济等。"
                ),
                "source_site": "工信部",
                "commodity": Commodity.ALUMINIUM,
                "region": Region.CHINA,
            },
            {
                "title": "稀土产业高质量发展规划",
                "content": (
                    "稀土是国家重要战略资源，为推动稀土产业高质量发展，编制本规划。"
                    "发展目标：到2030年，稀土新材料产业产值突破5000亿元；"
                    "高端稀土功能材料自给率达到80%以上；建成5个具有国际竞争力的"
                    "稀土产业集群。重点发展方向：稀土永磁材料、稀土发光材料、"
                    "稀土催化材料和稀土储氢材料。"
                ),
                "source_site": "工信部",
                "commodity": Commodity.RARE_EARTH,
                "region": Region.CHINA,
            },
            {
                "title": "Australia's Critical Minerals Strategy 2024-2034",
                "content": (
                    "The Australian Government has released its updated Critical Minerals "
                    "Strategy 2024-2034, outlining a comprehensive framework for developing "
                    "the nation's critical minerals sector. Key priorities include: "
                    "strengthening lithium and nickel processing capabilities, "
                    "building sovereign supply chains for battery materials, "
                    "enhancing partnerships with international allies, "
                    "and promoting sustainable mining practices. "
                    "The strategy identifies 26 critical minerals essential for "
                    "clean energy technologies and national security."
                ),
                "source_site": "澳洲DISR",
                "commodity": Commodity.LITHIUM,
                "region": Region.AUSTRALIA,
            },
            {
                "title": "澳洲锂出口管制政策更新",
                "content": (
                    "澳大利亚工业科学资源部(DISR)宣布更新锂矿出口管制政策，"
                    "要求年产能超过5万吨的锂精矿出口商须获得联邦政府审批。"
                    "新规将于2026年第三季度生效，旨在确保关键矿产供应链安全。"
                    "行业分析师指出，此举可能短期内推高国际锂价，"
                    "但有利于澳大利亚在锂加工领域获取更大附加值。"
                ),
                "source_site": "澳洲DISR",
                "commodity": Commodity.LITHIUM,
                "region": Region.AUSTRALIA,
            },
            {
                "title": "新能源汽车产业用矿产原材料保障方案",
                "content": (
                    "新能源汽车产业快速发展对锂、钴、镍、稀土等矿产原材料"
                    "需求持续增长。为保障产业链供应链安全稳定，制定本方案："
                    "一、加大国内勘查开发力度；二、推进海外资源合作开发；"
                    "三、完善回收利用体系；四、建立价格监测预警机制；"
                    "五、加强战略储备。到2030年，关键矿产自给率提升至60%。"
                ),
                "source_site": "国家发改委",
                "commodity": Commodity.LITHIUM,
                "region": Region.CHINA,
            },
            {
                "title": "Australia Critical Minerals Research and Development Hub",
                "content": (
                    "The Australian Government has established a new Critical Minerals "
                    "Research and Development Hub with funding of AUD 200 million. "
                    "The hub will focus on developing innovative extraction and "
                    "processing technologies for lithium, rare earths, and other "
                    "critical minerals. Collaboration with CSIRO and universities "
                    "will be central to the hub's mission of maintaining Australia's "
                    "competitive advantage in critical minerals supply chains."
                ),
                "source_site": "澳洲DISR",
                "commodity": Commodity.RARE_EARTH,
                "region": Region.AUSTRALIA,
            },
            {
                "title": "矿山安全生产条例修订草案",
                "content": (
                    "为进一步强化矿山安全生产工作，保障矿工生命安全，"
                    "对《矿山安全生产条例》进行修订。修订重点包括："
                    "提高矿山安全准入标准、强化企业主体责任、"
                    "完善应急管理体系、加大违法行为处罚力度、"
                    "建立矿山安全风险监测预警机制等。"
                ),
                "source_site": "应急管理部",
                "commodity": Commodity.OTHER,
                "region": Region.CHINA,
            },
            {
                "title": "关于严格管控铟锗镓等稀散金属出口的通知",
                "content": (
                    "铟、锗、镓等稀散金属是半导体和光电子产业的关键原材料。"
                    "为维护国家安全和利益，经国务院批准，决定对上述金属实施出口管制。"
                    "出口经营者须向国务院商务主管部门申请许可，"
                    "未经许可不得出口。本通知自发布之日起施行。"
                ),
                "source_site": "商务部",
                "commodity": Commodity.RARE_EARTH,
                "region": Region.CHINA,
            },
            {
                "title": "Australia-India Critical Minerals Partnership Agreement",
                "content": (
                    "澳大利亚与印度签署关键矿产合作伙伴协议，"
                    "旨在加强锂、镍、钴等关键矿产的供应链合作。"
                    "协议内容包括：澳大利亚向印度稳定供应锂精矿、"
                    "共同投资关键矿产勘探与开发、建立技术交流机制、"
                    "推动在电池制造领域的联合投资。该协议是澳大利亚"
                    "多元化关键矿产出口市场战略的重要组成部分。"
                ),
                "source_site": "澳洲DISR",
                "commodity": Commodity.LITHIUM,
                "region": Region.AUSTRALIA,
            },
            {
                "title": "战略性矿产资源储备管理办法",
                "content": (
                    "为保障国家经济安全和国防建设需要，加强战略性矿产资源储备管理，"
                    "制定本办法。储备矿种包括：稀土、钨、钼、锑、锡、铟、锗、镓等。"
                    "储备方式分为国家储备和企业储备，实行分类管理。"
                    "建立储备动用机制，在市场异常波动时及时调节供需平衡。"
                ),
                "source_site": "国家发改委",
                "commodity": Commodity.RARE_EARTH,
                "region": Region.CHINA,
            },
            {
                "title": "矿业领域生态文明建设实施方案",
                "content": (
                    "为深入贯彻生态文明思想，推动矿业领域生态文明建设，制定本方案。"
                    "主要任务：优化矿业空间布局、严格生态保护红线管控、"
                    "推进绿色勘查开发、加强矿山环境治理恢复、"
                    "发展矿业循环经济、强化科技创新支撑。"
                    "建立健全矿业生态文明考核评价体系。"
                ),
                "source_site": "自然资源部",
                "commodity": Commodity.OTHER,
                "region": Region.CHINA,
            },
            {
                "title": "Australian Lithium Valley Development Plan",
                "content": (
                    "西澳大利亚州政府发布Lithium Valley发展计划，"
                    "计划在珀斯南部Kwinana工业区建设世界级锂精炼集群。"
                    "项目总投资预计达到70亿澳元，建成后将形成从锂精矿"
                    "到电池级氢氧化锂的完整加工链条。预计可创造5000个就业岗位，"
                    "使澳大利亚锂加工产能提升至全球份额的25%。"
                ),
                "source_site": "澳洲DISR",
                "commodity": Commodity.LITHIUM,
                "region": Region.AUSTRALIA,
            },
            {
                "title": "关于加快煤矿智能化发展的指导意见",
                "content": (
                    "为推动煤矿智能化转型升级，提升煤矿安全高效生产水平，"
                    "提出指导意见：到2025年，大型煤矿和灾害严重煤矿基本实现智能化；"
                    "到2030年，各类煤矿基本实现智能化，建成智能感知、智能决策、"
                    "自动执行的煤矿安全高效生产体系。重点推进采掘智能化、"
                    "运输智能化、洗选智能化和安全管理智能化。"
                ),
                "source_site": "国家能源局",
                "commodity": Commodity.IRON_ORE,
                "region": Region.CHINA,
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
                    f"（发布日期: {date_str}，来源: {tmpl['source_site']}模拟数据）"
                )

                data.append(MiningData(
                    source_type=SourceType.POLICY,
                    title=title,
                    content=content,
                    publish_date=pub_date,
                    commodity=tmpl["commodity"],
                    country_or_region=tmpl["region"],
                    is_mock=True,
                    metadata={
                        "url": f"https://www.example.gov.cn/policy/simulated-{idx:04d}",
                        "source_site": tmpl["source_site"],
                    },
                ))
                idx += 1

        return data
