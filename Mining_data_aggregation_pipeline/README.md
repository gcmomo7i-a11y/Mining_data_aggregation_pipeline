# 矿业数据聚合管线

从多个数据源采集矿业新闻、政策法规、价格行情，经 ETL 处理后存入向量数据库，提供自然语言语义检索 API。

## 项目结构

```
Mining_data_aggregation_pipeline/
├── pipeline/                    # ETL 核心逻辑
│   ├── models.py                # Pydantic 数据模型
│   ├── etl.py                   # ETL 主流程编排
│   ├── utils.py                 # 公共工具（HTML清洗、去重哈希）
│   └── extractors/              # 数据采集器
│       ├── base.py              # 采集器抽象基类
│       ├── news.py              # 矿业新闻采集器
│       ├── policy.py            # 政策法规采集器
│       └── price.py             # 价格行情采集器
├── serve/                       # API 服务层
│   └── main.py                  # FastAPI 应用
├── eval/                        # 评估与测试
│   ├── ground_truth.json        # 20条测试问答对
│   └── run_eval.py              # 自动化评估脚本
├── tests/                       # 单元测试
├── configs/                     # 配置文件
│   └── settings.py              # 全局配置
├── data/                        # 运行时数据（ChromaDB持久化）
├── requirements.txt             # 依赖清单
├── DATA_NOTES.md                # 数据库Schema与字段映射
└── README.md
```

## 环境要求

- Python 3.10+
- pip

## 安装

```bash
# 克隆项目
cd Mining_data_aggregation_pipeline

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

## 运行

### 1. ETL 流水线（采集→去重→向量化→入库）

```bash
python -m pipeline.etl
```

该命令将依次：
- 从 mining.com 采集矿业新闻
- 从中国稀土集团/自然资源部/工信部采集政策法规
- 从 LME/SHFE 采集价格行情
- 基于 title+publish_date 哈希去重
- 使用 all-MiniLM-L6-v2 生成向量
- 存入 ChromaDB（`data/chroma_db/`）

### 2. 启动 API 服务

```bash
python -m serve.main
```

服务启动后访问：
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

### 3. 运行评估

先确保 API 服务已启动，然后执行：

```bash
python -m eval.run_eval
```

评估脚本将调用 `/query` 接口，使用 20 条测试用例计算 Recall@5 分数。

## API 接口

### POST /query

自然语言语义检索。

**请求：**

```json
{
  "question": "最近铜价走势如何？",
  "top_k": 5
}
```

**响应：**

```json
{
  "question": "最近铜价走势如何？",
  "results": [
    {
      "title": "SHFE 铜期货 价格行情 2026-07-01",
      "source_type": "price",
      "snippet": "2026-07-01 SHFE 铜价格上涨至72000.0...",
      "score": 0.8523
    }
  ],
  "total": 5
}
```

### GET /health

健康检查，返回服务状态和数据库数据量。

## 技术栈

| 组件 | 技术 |
|---|---|
| 语言 | Python 3.10+ |
| Web框架 | FastAPI |
| 向量数据库 | ChromaDB (PersistentClient) |
| 向量化模型 | sentence-transformers (all-MiniLM-L6-v2) |
| 爬虫 | requests, BeautifulSoup4 |
| 数据校验 | Pydantic |
| 进度条 | tqdm |
| 日志 | loguru |
| 代码规范 | PEP8, Type Hints |
