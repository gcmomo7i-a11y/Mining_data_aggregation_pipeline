# DATA_NOTES - 数据库Schema与字段映射

## ChromaDB Schema

### 集合名称

`mining_data`（cosine 相似度空间）

### 存储结构

| 字段 | 类型 | 说明 |
|---|---|---|
| id | string | UUID4，数据唯一标识 |
| embedding | list[float] | 384维向量（all-MiniLM-L6-v2 输出） |
| document | string | 原始正文内容，用于关键词检索和摘要展示 |
| metadata | dict | 元数据字段，见下方详细映射 |

### Metadata 字段映射

| Metadata Key | 类型 | 说明 | 来源 |
|---|---|---|---|
| source_type | string | 数据源类型：`news` / `policy` / `price` | MiningData.source_type |
| title | string | 数据标题 | MiningData.title |
| publish_date | string | 发布日期（YYYY-MM-DD格式） | MiningData.publish_date |
| url | string | 原始页面链接（如有） | MiningData.metadata["url"] |
| source_site | string | 采集站点名称 | MiningData.metadata["source_site"] |
| metal | string | 金属品种名称（仅price类型） | MiningData.metadata["metal"] |
| price | float | 价格数值（仅price类型） | MiningData.metadata["price"] |
| change | string | 涨跌变动值（仅price类型） | MiningData.metadata["change"] |
| simulated | string | 是否为模拟数据（"True"/缺省） | MiningData.metadata["simulated"] |

## 去重逻辑

采用 **SHA256 哈希去重**，基于 `title + publish_date` 组合键：

```
hash_input = "{title.strip()}|{publish_date.strftime('%Y-%m-%d')}"
dedup_hash = sha256(hash_input.encode('utf-8')).hexdigest()
```

- 标题与发布日期完全相同的记录视为重复
- 去重在 ETL 流程的 `_deduplicate` 阶段执行
- 使用 set 存储已见哈希，O(1) 查重

## Pydantic 模型 → ChromaDB 字段映射

```
MiningData.id            → ChromaDB.ids
MiningData.embedding     → ChromaDB.embeddings
MiningData.content       → ChromaDB.documents
MiningData.source_type   → ChromaDB.metadatas["source_type"]
MiningData.title         → ChromaDB.metadatas["title"]
MiningData.publish_date  → ChromaDB.metadatas["publish_date"]
MiningData.metadata[*]   → ChromaDB.metadatas[*]（平铺展开，值转为string）
```

## 数据流

```
采集器 → List[MiningData] (embedding=None)
    ↓ 去重（基于 title+date 哈希）
    ↓ 向量化（sentence-transformers 填充 embedding）
    ↓ 入库（ChromaDB upsert）
ChromaDB (id, embedding, document, metadata)
    ↓ API查询（/query → 向量化问题 → cosine检索Top5）
返回 QueryResult (title, source_type, snippet, score)
```

## 数据量要求

| 数据源 | 最少条数 | 采集器 |
|---|---|---|
| 新闻 (news) | 200 | NewsExtractor |
| 政策 (policy) | 200 | PolicyExtractor |
| 价格 (price) | 200 | PriceExtractor（含模拟补充） |
