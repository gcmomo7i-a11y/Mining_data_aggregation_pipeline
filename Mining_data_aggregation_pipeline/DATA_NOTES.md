# DATA_NOTES - 数据库Schema与字段映射

## ChromaDB Schema

### 集合名称

`mining_data`（cosine 相似度空间）

### 存储结构

| 字段 | 类型 | 说明 |
|---|---|---|
| id | string | UUID4，数据唯一标识（主键） |
| embedding | list[float] | 384维向量（all-MiniLM-L6-v2 输出） |
| document | string | 原始正文内容，用于关键词检索和摘要展示 |
| metadata | dict | 元数据字段，见下方详细映射 |

### Metadata 字段映射

| Metadata Key | 类型 | 说明 | 来源 |
|---|---|---|---|
| source_type | string | 数据源类型：`news` / `policy` / `price` | MiningData.source_type |
| title | string | 数据标题 | MiningData.title |
| publish_date | string | 发布日期（YYYY-MM-DD格式） | MiningData.publish_date |
| commodity | string | 矿产品种：copper/zinc/nickel/lithium/iron_ore/aluminium/lead/tin/gold/silver/rare_earth/other | MiningData.commodity |
| country_or_region | string | 国家/地区：China/Australia/Global/Other | MiningData.country_or_region |
| is_mock | string | 是否模拟数据："True"/"False" | MiningData.is_mock |
| url | string | 原始页面链接（如有） | MiningData.metadata["url"] |
| source_site | string | 采集站点名称 | MiningData.metadata["source_site"] |
| metal | string | 金属品种名称（仅price类型） | MiningData.metadata["metal"] |
| price | float | 价格数值（仅price类型） | MiningData.metadata["price"] |
| change | string | 涨跌变动值（仅price类型） | MiningData.metadata["change"] |
| simulated | string | 是否为模拟价格数据 | MiningData.metadata["simulated"] |

## 主键生成规则

主键为 UUID4，由 `uuid.uuid4()` 自动生成，全局唯一。

示例：`a1b2c3d4-e5f6-7890-abcd-ef1234567890`

## 去重策略

采用 **SHA256 哈希去重**，基于 `title + publish_date` 组合键：

```
hash_input = "{title.strip()}|{publish_date.strftime('%Y-%m-%d')}"
dedup_hash = sha256(hash_input.encode('utf-8')).hexdigest()
```

- 标题与发布日期完全相同的记录视为重复
- 去重在 ETL 流程的 `_deduplicate` 阶段执行
- 使用 set 存储已见哈希，O(1) 查重
- 去重前后数量记录在 ETLStats 中

## Pydantic 模型 → ChromaDB 字段映射

```
MiningData.id                → ChromaDB.ids
MiningData.embedding         → ChromaDB.embeddings
MiningData.content           → ChromaDB.documents
MiningData.source_type       → ChromaDB.metadatas["source_type"]
MiningData.title             → ChromaDB.metadatas["title"]
MiningData.publish_date      → ChromaDB.metadatas["publish_date"]
MiningData.commodity         → ChromaDB.metadatas["commodity"]
MiningData.country_or_region → ChromaDB.metadatas["country_or_region"]
MiningData.is_mock           → ChromaDB.metadatas["is_mock"]
MiningData.metadata[*]       → ChromaDB.metadatas[*]（平铺展开，值转为string）
```

## 数据源清单

### 新闻 (news)

| 数据源 | URL | 采集方式 | 目标区域 |
|---|---|---|---|
| mining.com | https://www.mining.com | HTML + RSS | Global |
| S&P Global Mining | https://www.spglobal.com | RSS | Global |

### 政策 (policy)

| 数据源 | URL | 采集方式 | 目标区域 |
|---|---|---|---|
| 中国稀土集团 | https://www.cregroup.com.cn | HTML | China |
| 自然资源部 | https://www.mnr.gov.cn | HTML | China |
| 工信部 | https://www.miit.gov.cn | HTML | China |
| 澳洲DISR | https://www.industry.gov.au | HTML | Australia |

### 价格 (price)

| 数据源 | 品种 | 采集方式 | 目标区域 |
|---|---|---|---|
| LME | 铜、锌、镍 | HTML | Global |
| SHFE | 碳酸锂 | JSON API | China |
| 上海钢联/Mysteel | 铁矿石 | API | China |

## 数据质量统计

### 数据量要求

| 数据源类型 | 最低条数 | 近30天要求 |
|---|---|---|
| 新闻 (news) | 200 | 是 |
| 政策 (policy) | 200 | 是 |
| 价格 (price) | 200 | 是 |
| **合计** | **600** | - |

### 真实 vs 模拟数据

- `is_mock=False`：从真实外部数据源采集
- `is_mock=True`：模拟数据兜底（当外部数据不可用时）

ETL流程运行后会输出统计报告，包含：
- 各源总数、真实数据量、模拟数据量
- 日期范围（最早/最晚日期）
- 去重前后数量
- 数据质量警告（不足200条或真实数据比例低）

### 字段缺失率

| 字段 | 缺失情况 |
|---|---|
| id | 0%（自动生成） |
| source_type | 0%（枚举必填） |
| title | 0%（必填校验） |
| content | 0%（必填校验） |
| publish_date | 0%（必填，采集失败时默认当前时间） |
| commodity | 0%（有默认值other） |
| country_or_region | 0%（有默认值Global） |
| is_mock | 0%（布尔默认False） |
| embedding | 入库前100%填充 |
| metadata.url | 部分模拟数据无真实URL |

### 价格数据品种覆盖

| 品种 | 交易所 | Commodity枚举 |
|---|---|---|
| 铜 | LME | copper |
| 锌 | LME | zinc |
| 镍 | LME | nickel |
| 碳酸锂 | SHFE | lithium |
| 铁矿石 | Mysteel | iron_ore |

## 数据流

```
采集器 → List[MiningData] (embedding=None)
    ↓ 去重（基于 title+date SHA256哈希）
    ↓ 向量化（sentence-transformers 填充 embedding）
    ↓ 入库（ChromaDB upsert，含commodity/country/is_mock元数据）
ChromaDB (id, embedding, document, metadata)
    ↓ API查询（/query → 时间解析 → 元数据过滤 → 向量化问题 → cosine检索TopK → RAG答案生成）
返回 QueryResponse (question, answer, results, filters_applied)
```
