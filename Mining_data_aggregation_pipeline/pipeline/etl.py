"""ETL主流程编排.

负责协调采集、去重、向量化、入库的完整数据处理流水线。
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


class ETLStats:
    """ETL统计数据.

    Attributes:
        raw_counts: 各数据源原始采集数量.
        deduped_count: 去重后数量.
        mock_counts: 各数据源模拟数据数量.
        real_counts: 各数据源真实数据数量.
        loaded_count: 入库数量.
        date_ranges: 各数据源最早/最晚日期.
    """

    def __init__(self) -> None:
        """初始化统计数据."""
        self.raw_counts: dict[str, int] = {}
        self.deduped_count: int = 0
        self.mock_counts: dict[str, int] = {}
        self.real_counts: dict[str, int] = {}
        self.loaded_count: int = 0
        self.date_ranges: dict[str, dict] = {}

    def record_raw(self, source_type: str, data: List[MiningData]) -> None:
        """记录原始采集统计.

        Args:
            source_type: 数据源类型.
            data: 采集到的数据列表.
        """
        self.raw_counts[source_type] = len(data)
        self.mock_counts[source_type] = sum(1 for d in data if d.is_mock)
        self.real_counts[source_type] = sum(1 for d in data if not d.is_mock)

        if data:
            dates = [d.publish_date for d in data]
            self.date_ranges[source_type] = {
                "earliest": min(dates).strftime("%Y-%m-%d"),
                "latest": max(dates).strftime("%Y-%m-%d"),
            }

    def validate(self) -> List[str]:
        """校验数据质量，返回警告列表.

        Returns:
            质量警告列表.
        """
        warnings: List[str] = []
        min_count = 200
        now = datetime.now()
        cutoff = now - timedelta(days=30)

        for source_type, count in self.raw_counts.items():
            if count < min_count:
                warnings.append(
                    f"{source_type}: 数据量{count}条，不足{min_count}条"
                )

            real = self.real_counts.get(source_type, 0)
            if real < min_count:
                warnings.append(
                    f"{source_type}: 真实数据仅{real}条，模拟数据"
                    f"{self.mock_counts.get(source_type, 0)}条"
                )

        return warnings

    def summary(self) -> str:
        """生成统计摘要.

        Returns:
            统计摘要文本.
        """
        lines = ["=" * 50, "ETL 统计报告", "=" * 50]

        for source_type in self.raw_counts:
            raw = self.raw_counts[source_type]
            real = self.real_counts.get(source_type, 0)
            mock = self.mock_counts.get(source_type, 0)
            date_range = self.date_ranges.get(source_type, {})

            lines.append(f"\n  [{source_type}]")
            lines.append(f"    总数: {raw}  (真实: {real}, 模拟: {mock})")
            if date_range:
                lines.append(
                    f"    日期范围: {date_range.get('earliest', 'N/A')} "
                    f"~ {date_range.get('latest', 'N/A')}"
                )

        lines.append(f"\n  去重后总数: {self.deduped_count}")
        lines.append(f"  入库总数: {self.loaded_count}")

        warnings = self.validate()
        if warnings:
            lines.append("\n  质量警告:")
            for w in warnings:
                lines.append(f"    - {w}")

        lines.append("=" * 50)
        return "\n".join(lines)


class ETLPipeline:
    """ETL流水线编排器.

    串联采集器→去重→向量化→入库四个阶段，
    使用tqdm显示处理进度，loguru记录日志。

    Example::

        pipeline = ETLPipeline()
        pipeline.run()
    """

    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        chroma_db_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        """初始化ETL管线.

        Args:
            embedding_model_name: 向量化模型名称.
            chroma_db_dir: ChromaDB持久化目录.
            collection_name: ChromaDB集合名称.
        """
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
        logger.info("ETL流水线启动")

        # 阶段1：采集
        raw_data = self._extract()
        logger.info(f"采集完成，共{len(raw_data)}条原始数据")

        if not raw_data:
            logger.warning("未采集到任何数据，流程终止")
            return 0

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

        # 打印统计报告
        print(self._stats.summary())

        return count

    def _extract(self) -> List[MiningData]:
        """执行数据采集阶段.

        Returns:
            采集到的原始数据列表.
        """
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
                self._stats.record_raw(source_type.value, data)
                logger.info(
                    f"{extractor.source_name} 采集完成: {len(data)}条"
                )
            except Exception as e:
                logger.error(f"{extractor.source_name} 采集异常: {e}")

        return all_data

    @staticmethod
    def _deduplicate(data: List[MiningData]) -> List[MiningData]:
        """基于title+publish_date哈希去重.

        Args:
            data: 待去重的数据列表.

        Returns:
            去重后的数据列表.
        """
        seen_hashes: set[str] = set()
        deduped: List[MiningData] = []

        for item in data:
            date_str = item.publish_date.strftime("%Y-%m-%d")
            hash_key = compute_dedup_hash(item.title, date_str)

            if hash_key in seen_hashes:
                continue

            seen_hashes.add(hash_key)
            deduped.append(item)

        return deduped

    def _vectorize(self, data: List[MiningData]) -> None:
        """使用sentence-transformers生成embedding.

        Args:
            data: 待向量化的数据列表（原地修改embedding字段）.
        """
        if self._model is None:
            logger.info(f"加载向量化模型: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)

        texts = [f"{item.title} {item.content}" for item in data]

        logger.info(f"开始向量化，共{len(texts)}条数据")
        embeddings = self._model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True,
        )

        for item, emb in tqdm(
            zip(data, embeddings),
            total=len(data),
            desc="写入embedding",
        ):
            item.embedding = emb.tolist()

    def _load(self, data: List[MiningData]) -> int:
        """将数据存入ChromaDB持久化数据库.

        Args:
            data: 待入库的数据列表.

        Returns:
            成功写入的数据条数.
        """
        if self._chroma_client is None:
            logger.info(f"初始化ChromaDB: {self._chroma_db_dir}")
            self._chroma_client = chromadb.PersistentClient(
                path=self._chroma_db_dir
            )

        self._collection = self._chroma_client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        batch_size = 100
        total_loaded = 0

        for i in tqdm(
            range(0, len(data), batch_size),
            desc="写入ChromaDB",
        ):
            batch = data[i : i + batch_size]

            valid_items = [
                item for item in batch if item.embedding is not None
            ]

            if not valid_items:
                continue

            ids = [str(item.id) for item in valid_items]
            embeddings = [item.embedding for item in valid_items]
            documents = [item.content for item in valid_items]
            metadatas = [
                {
                    "source_type": item.source_type.value,
                    "title": item.title,
                    "publish_date": item.publish_date.strftime("%Y-%m-%d"),
                    "commodity": item.commodity.value,
                    "country_or_region": item.country_or_region.value,
                    "is_mock": str(item.is_mock),
                    **{
                        k: str(v)
                        for k, v in item.metadata.items()
                        if isinstance(v, (str, int, float, bool))
                    },
                }
                for item in valid_items
            ]

            try:
                self._collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
                total_loaded += len(valid_items)
            except Exception as e:
                logger.error(f"ChromaDB写入失败(batch {i}): {e}")

        return total_loaded


def main() -> None:
    """ETL管线入口函数."""
    pipeline = ETLPipeline()
    count = pipeline.run()
    print(f"\nETL流程完成，共入库 {count} 条数据")


if __name__ == "__main__":
    main()
