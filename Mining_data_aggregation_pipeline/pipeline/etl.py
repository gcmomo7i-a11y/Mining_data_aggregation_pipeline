"""ETL主流程编排.

负责协调采集、去重、向量化、入库的完整数据处理流水线。
支持strict模式（强制近30天200条/类真实数据）和demo模式（允许mock补齐）。
"""

from datetime import datetime, timedelta
from typing import List, Optional

from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import chromadb

from configs import settings
from pipeline.models import MiningData, SourceType
from pipeline.utils import compute_dedup_hash
from pipeline.extractors.news import NewsExtractor
from pipeline.extractors.policy import PolicyExtractor
from pipeline.extractors.price import PriceExtractor

# 严格模式：每类最低数据量
MIN_PER_SOURCE = 200
# 严格模式：数据必须在此天数范围内
RECENT_DAYS = 30


class ETLStats:
    """ETL统计数据."""

    def __init__(self) -> None:
        self.raw_counts: dict[str, int] = {}
        self.real_counts: dict[str, int] = {}
        self.mock_counts: dict[str, int] = {}
        self.recent_real_counts: dict[str, int] = {}
        self.deduped_count: int = 0
        self.loaded_count: int = 0
        self.date_ranges: dict[str, dict] = {}

    def record_raw(self, source_type: str, data: List[MiningData]) -> None:
        """记录原始采集统计."""
        now = datetime.now()
        cutoff = now - timedelta(days=RECENT_DAYS)

        self.raw_counts[source_type] = len(data)
        self.mock_counts[source_type] = sum(1 for d in data if d.is_mock)
        self.real_counts[source_type] = sum(1 for d in data if not d.is_mock)
        self.recent_real_counts[source_type] = sum(
            1 for d in data
            if not d.is_mock and d.publish_date >= cutoff
        )

        if data:
            dates = [d.publish_date for d in data]
            self.date_ranges[source_type] = {
                "earliest": min(dates).strftime("%Y-%m-%d"),
                "latest": max(dates).strftime("%Y-%m-%d"),
            }

    def validate_strict(self) -> List[str]:
        """严格校验：每类近30天真实数据>=200条.

        Returns:
            错误列表（非空则表示校验不通过）.
        """
        errors: List[str] = []
        now = datetime.now()
        cutoff = now - timedelta(days=RECENT_DAYS)

        for source_type in [SourceType.NEWS, SourceType.POLICY, SourceType.PRICE]:
            key = source_type.value
            real = self.real_counts.get(key, 0)
            recent_real = self.recent_real_counts.get(key, 0)
            mock = self.mock_counts.get(key, 0)

            if real < MIN_PER_SOURCE:
                errors.append(
                    f"{key}: 真实数据仅{real}条，不足{MIN_PER_SOURCE}条"
                    f"（其中mock{mock}条）"
                )

            if recent_real < MIN_PER_SOURCE:
                errors.append(
                    f"{key}: 近{RECENT_DAYS}天真实数据仅{recent_real}条，"
                    f"不足{MIN_PER_SOURCE}条"
                )

        total_real = sum(self.real_counts.values())
        if total_real < MIN_PER_SOURCE * 3:
            errors.append(
                f"合计真实数据{total_real}条，不足{MIN_PER_SOURCE * 3}条"
            )

        return errors

    def validate_warnings(self) -> List[str]:
        """宽松校验：仅输出警告.

        Returns:
            警告列表.
        """
        warnings: List[str] = []
        for source_type, count in self.raw_counts.items():
            real = self.real_counts.get(source_type, 0)
            recent_real = self.recent_real_counts.get(source_type, 0)
            mock = self.mock_counts.get(source_type, 0)
            if count < MIN_PER_SOURCE:
                warnings.append(
                    f"{source_type}: 总数{count}条，不足{MIN_PER_SOURCE}条"
                )
            if real < MIN_PER_SOURCE:
                warnings.append(
                    f"{source_type}: 真实数据{real}条，mock数据{mock}条"
                )
            if recent_real < MIN_PER_SOURCE:
                warnings.append(
                    f"{source_type}: 近{RECENT_DAYS}天真实数据{recent_real}条"
                )
        return warnings

    def summary(self) -> str:
        """生成统计摘要."""
        lines = ["=" * 55, "ETL 统计报告", "=" * 55]

        for source_type in self.raw_counts:
            raw = self.raw_counts[source_type]
            real = self.real_counts.get(source_type, 0)
            mock = self.mock_counts.get(source_type, 0)
            recent_real = self.recent_real_counts.get(source_type, 0)
            date_range = self.date_ranges.get(source_type, {})

            lines.append(f"\n  [{source_type}]")
            lines.append(f"    总数: {raw}  (真实: {real}, mock: {mock})")
            lines.append(f"    近{RECENT_DAYS}天真实: {recent_real}")
            if date_range:
                lines.append(
                    f"    日期范围: {date_range.get('earliest', 'N/A')} "
                    f"~ {date_range.get('latest', 'N/A')}"
                )

        lines.append(f"\n  去重后总数: {self.deduped_count}")
        lines.append(f"  入库总数: {self.loaded_count}")

        warnings = self.validate_warnings()
        if warnings:
            lines.append("\n  质量警告:")
            for w in warnings:
                lines.append(f"    - {w}")

        lines.append("=" * 55)
        return "\n".join(lines)


def _fill_mock_for_source(
    source_type: SourceType, count: int
) -> List[MiningData]:
    """为指定数据源类型生成mock数据，确保近30天每日覆盖.

    Args:
        source_type: 数据源类型.
        count: 需要生成的数量.

    Returns:
        mock MiningData列表.
    """
    from dateutil.relativedelta import relativedelta
    from pipeline.models import Commodity, Region

    data: List[MiningData] = []
    today = datetime.now()

    if source_type == SourceType.NEWS:
        titles = [
            ("全球铜矿产量增长，智利新项目投产", Commodity.COPPER, Region.GLOBAL),
            ("澳大利亚铁矿石出口创新高", Commodity.IRON_ORE, Region.AUSTRALIA),
            ("印尼镍矿出口政策调整引发关注", Commodity.NICKEL, Region.GLOBAL),
            ("非洲锂矿开发加速", Commodity.LITHIUM, Region.GLOBAL),
            ("加拿大黄金矿业公司重大收购", Commodity.GOLD, Region.GLOBAL),
            ("秘鲁铜矿社区抗议影响生产", Commodity.COPPER, Region.GLOBAL),
            ("稀土分离技术突破降低环境成本", Commodity.RARE_EARTH, Region.CHINA),
            ("澳洲锂出口政策面临调整", Commodity.LITHIUM, Region.AUSTRALIA),
            ("全球锌矿供应紧张推动价格上行", Commodity.ZINC, Region.GLOBAL),
            ("S&P Global: 全球矿业并购交易额创新高", Commodity.OTHER, Region.GLOBAL),
        ]
        contents = {
            "全球铜矿产量增长，智利新项目投产": "据最新行业报告，全球铜矿产量预计将在2026年实现显著增长。智利国家矿业公司宣布其位于阿塔卡马沙漠的新铜矿项目已正式投产，预计年产能将达到15万吨。分析人士指出，随着电动汽车和可再生能源产业对铜需求的持续攀升，铜矿供应的增加对稳定市场至关重要。",
            "澳大利亚铁矿石出口创新高": "澳大利亚统计局最新数据显示，该国铁矿石出口量在上季度达到历史新高。皮尔巴拉地区的大型矿场持续满负荷运转，主要出口目的地为中国和日本。行业专家认为，尽管全球钢铁需求增速放缓，但澳大利亚凭借低开采成本仍保持强劲的出口竞争力。",
            "印尼镍矿出口政策调整引发关注": "印尼政府宣布将对镍矿出口政策进行调整，加强了对原矿出口的限制力度。作为全球最大的镍矿储藏国，印尼的政策变动直接影响国际镍价走势。下游电池制造商面临原材料成本上升的压力，部分企业已开始寻求替代供应渠道。",
            "非洲锂矿开发加速": "非洲多国正加速锂矿资源的勘探与开发。津巴布韦、纳米比亚和马里等国近期相继发放了新的锂矿开采许可证。中国和澳大利亚矿业公司积极布局，投资总额已超过50亿美元。锂作为电动汽车电池的核心原材料，其战略价值日益凸显。",
            "加拿大黄金矿业公司重大收购": "加拿大知名黄金矿业公司Barrick Gold宣布将收购一家中型金矿企业，交易金额预计达到32亿美元。此次收购将使Barrick在北美地区的黄金产量提升约20%。市场分析认为，这反映了大型矿企在金价高企背景下积极扩充资源储备的战略意图。",
            "秘鲁铜矿社区抗议影响生产": "秘鲁南部安第斯山脉地区的铜矿再次爆发社区抗议活动，当地居民要求矿业公司增加社区投资和就业机会。抗议已导致至少两座大型铜矿暂时停产，预计影响月产量约3万吨。秘鲁是全球第二大铜生产国，此类社会冲突频发令国际市场担忧供应稳定性。",
            "稀土分离技术突破降低环境成本": "中国科学院研究团队宣布在稀土分离技术上取得重大突破，新型萃取工艺可将废水排放量减少80%以上。该技术已在江西稀土产业园区完成中试，有望在2027年前实现工业化应用。这一进展对全球稀土产业链的绿色转型具有重要意义。",
            "澳洲锂出口政策面临调整": "澳大利亚工业科学资源部(DISR)发布最新关键矿产战略文件，提出将加强对锂等关键矿产出口的监管框架。新战略强调需要在国家安全和经济利益之间取得平衡，建议对锂精矿出口实施更严格的审查机制。澳大利亚是全球最大的锂出口国，此举引发国际市场广泛关注。",
            "全球锌矿供应紧张推动价格上行": "受多座大型锌矿产能下降影响，全球锌矿供应出现紧张局面。国际铅锌研究小组数据显示，今年全球锌矿产量同比下降3.2%，而需求则保持温和增长。供给缺口推动LME锌价持续走高，部分贸易商开始增加库存。",
            "S&P Global: 全球矿业并购交易额创新高": "S&P Global Market Intelligence最新报告显示，2026年上半年全球矿业并购交易总额达到创纪录的1250亿美元。铜和锂矿资产成为最受追捧的标的，交易占比分别达到28%和22%。大型矿企积极通过并购方式补充资源储备，以应对能源转型带来的需求增长。",
        }

        idx = 0
        for days_back in range(RECENT_DAYS):
            if idx >= count:
                break
            pub_date = today - relativedelta(days=days_back)
            for title, commodity, region in titles:
                if idx >= count:
                    break
                date_str = pub_date.strftime("%Y-%m-%d")
                full_title = f"{title} ({date_str})"
                full_content = f"{contents[title]} （发布日期: {date_str}，来源: 模拟数据）"
                data.append(MiningData(
                    source_type=SourceType.NEWS,
                    title=full_title, content=full_content,
                    publish_date=pub_date, commodity=commodity,
                    country_or_region=region, is_mock=True,
                    metadata={"url": f"https://www.mining.com/news/mock-{idx:04d}", "source_site": "mining.com"},
                ))
                idx += 1

    elif source_type == SourceType.POLICY:
        templates = [
            ("关于加强稀土行业管理的通知", "为规范稀土行业秩序，严格控制稀土开采总量，加强稀土出口管理，推进稀土产业整合，加大环保监管力度。", Commodity.RARE_EARTH, Region.CHINA, "国务院"),
            ("矿产资源开发利用方案审查办法", "规范矿产资源开发利用方案审查工作，审查矿山设计规模与资源储量匹配性、开采工艺技术先进性。", Commodity.OTHER, Region.CHINA, "自然资源部"),
            ("有色金属行业碳达峰实施方案", "推动有色金属行业绿色低碳转型，到2030年碳排放达峰，到2040年电解铝使用可再生能源比例提升至50%。", Commodity.ALUMINIUM, Region.CHINA, "工信部"),
            ("稀土产业高质量发展规划", "到2030年稀土新材料产值突破5000亿元，高端稀土功能材料自给率达到80%以上。", Commodity.RARE_EARTH, Region.CHINA, "工信部"),
            ("Australia Critical Minerals Strategy 2024-2034", "澳大利亚关键矿产战略：加强锂和镍加工能力，建立主权电池材料供应链，促进可持续采矿。", Commodity.LITHIUM, Region.AUSTRALIA, "澳洲DISR"),
            ("澳洲锂出口管制政策更新", "DISR更新锂矿出口管制政策，年产能超过5万吨的锂精矿出口商须获联邦审批。新规2026年Q3生效。", Commodity.LITHIUM, Region.AUSTRALIA, "澳洲DISR"),
            ("战略性矿产资源储备管理办法", "储备矿种包括稀土、钨、钼、锑、锡、铟、锗、镓等，储备方式分国家储备和企业储备。", Commodity.RARE_EARTH, Region.CHINA, "国家发改委"),
            ("矿业权制度改革意见", "全面推进矿业权竞争性出让，完善矿业权退出机制，规范矿业权转让行为。", Commodity.OTHER, Region.CHINA, "自然资源部"),
            ("新能源汽车矿产原材料保障方案", "加大国内勘查开发力度，推进海外资源合作开发，完善回收利用体系，到2030年关键矿产自给率提升至60%。", Commodity.LITHIUM, Region.CHINA, "国家发改委"),
            ("Australian Lithium Valley Development Plan", "西澳Lithium Valley发展计划：在Kwinana建设世界级锂精炼集群，总投资70亿澳元。", Commodity.LITHIUM, Region.AUSTRALIA, "澳洲DISR"),
        ]

        idx = 0
        for days_back in range(RECENT_DAYS):
            if idx >= count:
                break
            pub_date = today - relativedelta(days=days_back)
            for title, content, commodity, region, site in templates:
                if idx >= count:
                    break
                date_str = pub_date.strftime("%Y-%m-%d")
                full_title = f"{title} ({date_str})"
                full_content = f"{content} （发布日期: {date_str}，来源: {site}模拟数据）"
                data.append(MiningData(
                    source_type=SourceType.POLICY,
                    title=full_title, content=full_content,
                    publish_date=pub_date, commodity=commodity,
                    country_or_region=region, is_mock=True,
                    metadata={"url": f"https://example.gov.cn/policy/mock-{idx:04d}", "source_site": site},
                ))
                idx += 1

    elif source_type == SourceType.PRICE:
        import random
        metals = [
            ("铜", "LME", Commodity.COPPER, Region.GLOBAL, 72000.0, "美元/吨"),
            ("锌", "LME", Commodity.ZINC, Region.GLOBAL, 22000.0, "美元/吨"),
            ("镍", "LME", Commodity.NICKEL, Region.GLOBAL, 128000.0, "美元/吨"),
            ("碳酸锂", "SHFE", Commodity.LITHIUM, Region.CHINA, 95000.0, "元/吨"),
            ("铁矿石", "Mysteel", Commodity.IRON_ORE, Region.CHINA, 800.0, "元/吨"),
        ]

        idx = 0
        for days_back in range(60):  # 覆盖60天确保5品种×60天≥200条
            if idx >= count:
                break
            pub_date = today - relativedelta(days=days_back)
            date_str = pub_date.strftime("%Y-%m-%d")
            for name, exchange, commodity, region, base, unit in metals:
                if idx >= count:
                    break
                price = round(base * (1 + random.uniform(-0.05, 0.05)), 2)
                change = round(price - base, 2)
                direction = "上涨" if change > 0 else "下跌" if change < 0 else "持平"
                content = f"{date_str} {exchange} {name}价格{direction}至{price}{unit}，较前一交易日变动{abs(change)}{unit}。"
                data.append(MiningData(
                    source_type=SourceType.PRICE,
                    title=f"{exchange} {name}价格行情 {date_str}",
                    content=content, publish_date=pub_date,
                    commodity=commodity, country_or_region=region, is_mock=True,
                    metadata={"source_site": exchange, "metal": name, "price": price, "change": str(change), "date": date_str},
                ))
                idx += 1

    return data


class ETLPipeline:
    """ETL流水线编排器.

    Args:
        demo_mode: True时允许mock补齐（默认），False为strict模式强制真实数据.
    """

    def __init__(
        self,
        demo_mode: bool = True,
        embedding_model_name: Optional[str] = None,
        chroma_db_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        self._demo_mode = demo_mode
        self._model_name = embedding_model_name or settings.EMBEDDING_MODEL_NAME
        self._chroma_db_dir = chroma_db_dir or str(settings.CHROMA_DB_DIR)
        self._collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self._model: Optional[SentenceTransformer] = None
        self._chroma_client: Optional[chromadb.PersistentClient] = None
        self._collection = None
        self._stats = ETLStats()

    def run(self) -> int:
        """执行完整ETL流程.

        Returns:
            入库的总数据条数.
        """
        logger.info(f"ETL流水线启动 (mode={'demo' if self._demo_mode else 'strict'})")

        # 阶段1：采集
        raw_data = self._extract()
        logger.info(f"采集完成，共{len(raw_data)}条原始数据")

        if not raw_data:
            if self._demo_mode:
                logger.warning("未采集到任何数据，demo模式下生成全量mock数据")
                raw_data = self._fill_all_mock()
            else:
                logger.error("未采集到任何数据，strict模式终止")
                return 0

        # 记录统计
        for st in SourceType:
            self._stats.record_raw(
                st.value, [d for d in raw_data if d.source_type == st]
            )

        # Strict模式校验
        if not self._demo_mode:
            errors = self._stats.validate_strict()
            if errors:
                for e in errors:
                    logger.error(f"[STRICT] {e}")
                logger.error("strict模式校验未通过，流程终止")
                print(self._stats.summary())
                return 0

        # Demo模式补齐
        if self._demo_mode:
            raw_data = self._ensure_min_count(raw_data)

        # 阶段2：去重
        deduped_data = self._deduplicate(raw_data)
        self._stats.deduped_count = len(deduped_data)
        logger.info(f"去重完成，保留{len(deduped_data)}条数据")

        if not deduped_data:
            logger.warning("去重后无数据，流程终止")
            return 0

        # 阶段3：向量化
        self._vectorize(deduped_data)
        logger.info("向量化完成")

        # 阶段4：入库
        count = self._load(deduped_data)
        self._stats.loaded_count = count
        logger.info(f"入库完成，共写入{count}条数据")

        print(self._stats.summary())
        return count

    def _extract(self) -> List[MiningData]:
        """执行数据采集."""
        all_data: List[MiningData] = []
        extractors = [
            (SourceType.NEWS, NewsExtractor(max_pages=settings.MAX_PAGES_NEWS)),
            (SourceType.POLICY, PolicyExtractor(max_pages=settings.MAX_PAGES_POLICY)),
            (SourceType.PRICE, PriceExtractor()),
        ]
        for source_type, extractor in extractors:
            try:
                logger.info(f"开始采集: {extractor.source_name}")
                data = extractor.extract()
                all_data.extend(data)
                logger.info(f"{extractor.source_name} 采集完成: {len(data)}条")
            except Exception as e:
                logger.error(f"{extractor.source_name} 采集异常: {e}")
        return all_data

    def _ensure_min_count(self, data: List[MiningData]) -> List[MiningData]:
        """Demo模式：为每类数据补齐至MIN_PER_SOURCE条.

        Args:
            data: 当前数据列表.

        Returns:
            补齐后的数据列表.
        """
        result = list(data)
        for st in SourceType:
            current = [d for d in result if d.source_type == st]
            deficit = MIN_PER_SOURCE - len(current)
            if deficit > 0:
                mock = _fill_mock_for_source(st, deficit)
                result.extend(mock)
                logger.warning(
                    f"[DEMO] {st.value}不足{MIN_PER_SOURCE}条，补充{len(mock)}条mock数据"
                )
        return result

    def _fill_all_mock(self) -> List[MiningData]:
        """全量mock数据填充."""
        data: List[MiningData] = []
        for st in SourceType:
            data.extend(_fill_mock_for_source(st, MIN_PER_SOURCE))
        logger.warning(f"[DEMO] 生成全量mock数据{len(data)}条")
        return data

    @staticmethod
    def _deduplicate(data: List[MiningData]) -> List[MiningData]:
        """基于title+publish_date哈希去重."""
        seen: set[str] = set()
        deduped: List[MiningData] = []
        for item in data:
            h = compute_dedup_hash(item.title, item.publish_date.strftime("%Y-%m-%d"))
            if h in seen:
                continue
            seen.add(h)
            deduped.append(item)
        return deduped

    def _vectorize(self, data: List[MiningData]) -> None:
        """向量化."""
        if self._model is None:
            logger.info(f"加载向量化模型: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)

        texts = [f"{item.title} {item.content}" for item in data]
        logger.info(f"开始向量化，共{len(texts)}条数据")
        embeddings = self._model.encode(
            texts, show_progress_bar=True, batch_size=32,
            normalize_embeddings=True,
        )
        for item, emb in tqdm(zip(data, embeddings), total=len(data), desc="写入embedding"):
            item.embedding = emb.tolist()

    def _load(self, data: List[MiningData]) -> int:
        """入库."""
        if self._chroma_client is None:
            logger.info(f"初始化ChromaDB: {self._chroma_db_dir}")
            self._chroma_client = chromadb.PersistentClient(path=self._chroma_db_dir)

        self._collection = self._chroma_client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        total = 0
        for i in tqdm(range(0, len(data), 100), desc="写入ChromaDB"):
            batch = [d for d in data[i:i+100] if d.embedding is not None]
            if not batch:
                continue
            ids = [str(d.id) for d in batch]
            embeddings = [d.embedding for d in batch]
            documents = [d.content for d in batch]
            metadatas = [
                {
                    "source_type": d.source_type.value,
                    "title": d.title,
                    "publish_date": d.publish_date.strftime("%Y-%m-%d"),
                    "commodity": d.commodity.value,
                    "country_or_region": d.country_or_region.value,
                    "is_mock": str(d.is_mock),
                    **{k: str(v) for k, v in d.metadata.items() if isinstance(v, (str, int, float, bool))},
                }
                for d in batch
            ]
            try:
                self._collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
                total += len(batch)
            except Exception as e:
                logger.error(f"ChromaDB写入失败(batch {i}): {e}")
        return total


def main() -> None:
    """ETL管线入口函数."""
    import argparse
    parser = argparse.ArgumentParser(description="矿业数据ETL管线")
    parser.add_argument("--strict", action="store_true", help="strict模式：强制近30天200条/类真实数据，不足则报错终止")
    args = parser.parse_args()

    pipeline = ETLPipeline(demo_mode=not args.strict)
    count = pipeline.run()
    print(f"\nETL流程完成，共入库 {count} 条数据")


if __name__ == "__main__":
    main()
