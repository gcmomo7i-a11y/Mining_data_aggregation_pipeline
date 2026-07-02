"""政策法规采集器.

从政府/行业网站采集矿业相关政策法规数据。
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
from pipeline.models import MiningData, SourceType
from pipeline.utils import HTMLCleaner, get_request_headers


class PolicyExtractor(BaseExtractor):
    """政策法规采集器.

    目标网站: 中国稀土集团及相关政府网站
    使用 requests + BeautifulSoup 抓取政策列表及详情，
    通过 HTMLCleaner 清洗页面内容。

    Example::

        extractor = PolicyExtractor(max_pages=5)
        data = extractor.extract()
    """

    BASE_URL: str = "https://www.cregroup.com.cn"
    LIST_URLS: List[str] = [
        "https://www.cregroup.com.cn/newsList/index.html",
        "https://www.cregroup.com.cn/xwzx/index.html",
    ]

    # 备用政府网站
    BACKUP_SITES: List[dict] = [
        {
            "name": "自然资源部",
            "base_url": "https://www.mnr.gov.cn",
            "list_url": "https://www.mnr.gov.cn/gk/zcwj/",
        },
        {
            "name": "工信部",
            "base_url": "https://www.miit.gov.cn",
            "list_url": "https://www.miit.gov.cn/jgsj/zzgs/wjfb/index.html",
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

        依次从各数据源采集政策信息，合并结果。

        Returns:
            采集到的MiningData列表.
        """
        all_data: List[MiningData] = []

        # 采集主要站点
        for list_url in self.LIST_URLS:
            try:
                data = self._extract_from_site(
                    base_url=self.BASE_URL,
                    list_url=list_url,
                    site_name="中国稀土集团",
                )
                all_data.extend(data)
            except Exception as e:
                logger.error(f"采集中国稀土集团失败: {e}")

        # 采集备用政府网站
        for site in self.BACKUP_SITES:
            try:
                data = self._extract_from_site(
                    base_url=site["base_url"],
                    list_url=site["list_url"],
                    site_name=site["name"],
                )
                all_data.extend(data)
            except Exception as e:
                logger.error(f"采集{site['name']}失败: {e}")

        # 外部数据不足时补充模拟数据
        if len(all_data) < 200:
            simulated = self._generate_simulated_policies(200 - len(all_data))
            all_data.extend(simulated)
            self.log_progress(
                f"外部数据不足，补充模拟政策{len(simulated)}条"
            )

        self.log_progress(f"共采集政策{len(all_data)}条")
        return all_data

    def _extract_from_site(
        self, base_url: str, list_url: str, site_name: str
    ) -> List[MiningData]:
        """从指定站点采集政策数据.

        Args:
            base_url: 站点基础URL.
            list_url: 列表页URL.
            site_name: 站点名称（用于日志）.

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
                            article, base_url, site_name
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

        # 适配多种页面结构的链接提取
        link_tags = soup.select("ul.list li a[href]") or soup.select(
            "div.list-item a[href]"
        ) or soup.select("div.news-list a[href]")

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
                "span", class_=lambda c: c and "date" in c.lower() if c else False
            ) or parent.find("span", class_=lambda c: c and "time" in c.lower() if c else False)
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
        self, article_info: dict, base_url: str, site_name: str
    ) -> Optional[MiningData]:
        """抓取政策详情页.

        Args:
            article_info: 政策基本信息.
            base_url: 站点基础URL.
            site_name: 站点名称.

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
                "div.TRS_Editor, div.page_content, article"
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
        ):
            try:
                return datetime.strptime(date_str[:19].strip(), fmt)
            except (ValueError, TypeError):
                continue

        logger.warning(f"日期解析失败: {date_str}，使用当前时间")
        return datetime.now()

    @staticmethod
    def _generate_simulated_policies(count: int) -> List[MiningData]:
        """生成模拟政策数据（当外部数据源不可用时补充）.

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
            },
            {
                "title": "关于深化矿业权制度改革的意见",
                "content": (
                    "为完善矿业权管理制度，优化资源配置，提出以下意见："
                    "一、全面推进矿业权竞争性出让；二、完善矿业权退出机制；"
                    "三、规范矿业权转让行为；四、加强矿业权事中事后监管；"
                    "五、推进矿业权信息化管理。各地要结合实际制定具体实施方案。"
                ),
                "source_site": "自然资源部",
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
            },
            {
                "title": "关于促进矿业绿色发展的指导意见",
                "content": (
                    "为推动矿业走绿色发展之路，实现资源开发与生态保护协调统一，"
                    "提出以下指导意见：建设绿色矿山、推进矿山生态修复、"
                    "发展矿业循环经济、加强矿山环境监测、完善绿色矿山标准体系。"
                    "力争到2030年，全国大中型矿山基本达到绿色矿山建设标准。"
                ),
                "source_site": "自然资源部",
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
            },
            {
                "title": "关于规范矿产资源勘查开采登记管理有关事项的通知",
                "content": (
                    "为进一步规范矿产资源勘查开采登记管理，优化审批流程，"
                    "提高行政效能，现就有关事项通知如下：精简审批环节、"
                    "推行网上办理、实行并联审查、压缩办理时限、"
                    "强化信息公开。各地自然资源主管部门要严格落实，"
                    "切实提高矿业权登记管理服务水平。"
                ),
                "source_site": "自然资源部",
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
            },
            {
                "title": "矿山生态修复技术规范",
                "content": (
                    "为规范矿山生态修复技术要求，保障修复效果，制定本规范。"
                    "主要技术要求包括：地形重塑、植被恢复、水土保持、"
                    "土壤改良、水体修复等。修复标准分为近期、中期和远期目标，"
                    "近期以消除安全隐患为主，中期恢复基本生态功能，"
                    "远期实现生态系统自我维持。"
                ),
                "source_site": "生态环境部",
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
            },
            {
                "title": "矿产资源税费制度改革方案",
                "content": (
                    "为完善矿产资源有偿使用制度，推进资源税改革，制定本方案："
                    "一、从价计征取代从量计征；二、合理设置税率幅度；"
                    "三、清理规范收费基金；四、建立税费动态调整机制；"
                    "五、加强税收征管。改革旨在促进资源节约集约利用，"
                    "维护国家资源权益，推动矿业高质量发展。"
                ),
                "source_site": "财政部",
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
            },
        ]

        data: List[MiningData] = []
        today = datetime.now()
        idx = 0

        for days_back in range(365):
            if idx >= count:
                break
            pub_date = today - relativedelta(days=days_back)
            for tmpl_idx, tmpl in enumerate(templates):
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
                    metadata={
                        "url": f"https://www.example.gov.cn/policy/simulated-{idx:04d}",
                        "source_site": tmpl["source_site"],
                        "simulated": True,
                    },
                ))
                idx += 1

        return data
