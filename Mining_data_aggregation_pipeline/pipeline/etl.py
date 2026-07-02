"""ETL主流程编排.

负责协调采集、去重、向量化、入库的完整数据处理流水线。
"""

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
        logger.info(f"去重完成，保留{len(deduped_data)}条数据")

        if not deduped_data:
            logger.warning("去重后无数据，流程终止")
            return 0

        # 阶段3：向量化
        self._vectorize(deduped_data)
        logger.info("向量化完成")

        # 阶段4：入库
        count = self._load(deduped_data)
        logger.info(f"入库完成，共写入{count}条数据")

        return count

    def _extract(self) -> List[MiningData]:
        """执行数据采集阶段.

        依次调用三个采集器，合并采集结果。

        Returns:
            采集到的原始数据列表.
        """
        all_data: List[MiningData] = []

        extractors = [
            NewsExtractor(max_pages=settings.MAX_PAGES_NEWS),
            PolicyExtractor(max_pages=settings.MAX_PAGES_POLICY),
            PriceExtractor(),
        ]

        for extractor in extractors:
            try:
                logger.info(f"开始采集: {extractor.source_name}")
                data = extractor.extract()
                all_data.extend(data)
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

        批量处理数据，使用tqdm显示进度。

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

            ids = [str(item.id) for item in batch]
            embeddings = [
                item.embedding for item in batch if item.embedding is not None
            ]
            documents = [item.content for item in batch]
            metadatas = [
                {
                    "source_type": item.source_type.value,
                    "title": item.title,
                    "publish_date": item.publish_date.strftime("%Y-%m-%d"),
                    **{
                        k: str(v)
                        for k, v in item.metadata.items()
                        if isinstance(v, (str, int, float, bool))
                    },
                }
                for item in batch
            ]

            # 只入库有embedding的数据
            valid_items = [
                (idx, item)
                for idx, item in enumerate(batch)
                if item.embedding is not None
            ]

            if not valid_items:
                continue

            valid_ids = [str(batch[idx].id) for idx, _ in valid_items]
            valid_embeddings = [
                batch[idx].embedding for idx, _ in valid_items
            ]
            valid_documents = [batch[idx].content for idx, _ in valid_items]
            valid_metadatas = [metadatas[idx] for idx, _ in valid_items]

            try:
                self._collection.upsert(
                    ids=valid_ids,
                    embeddings=valid_embeddings,
                    documents=valid_documents,
                    metadatas=valid_metadatas,
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
