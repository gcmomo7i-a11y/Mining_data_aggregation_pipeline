"""全局配置模块.

集中管理项目所有配置项，包括路径、模型参数、采集参数等。
"""

from pathlib import Path

# 项目根目录
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# 数据目录
DATA_DIR: Path = PROJECT_ROOT / "data"
CHROMA_DB_DIR: Path = DATA_DIR / "chroma_db"

# ChromaDB 配置
CHROMA_COLLECTION_NAME: str = "mining_data"

# 向量化模型
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

# 采集配置
MAX_PAGES_NEWS: int = 10
MAX_PAGES_POLICY: int = 5
MAX_PAGES_PRICE: int = 5

# 采集延迟（秒），避免被封
CRAWL_DELAY: float = 1.0

# API 服务配置
API_HOST: str = "0.0.0.0"
API_PORT: int = 8000

# 检索配置
QUERY_TOP_K: int = 5

# User-Agent 池
USER_AGENTS: list[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) "
    "Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
]
