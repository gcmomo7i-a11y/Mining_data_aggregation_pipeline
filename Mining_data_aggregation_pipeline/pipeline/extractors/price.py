"""价格行情采集器.

从 LME（铜/锌/镍）、SHFE（锂）、上海钢联/Mysteel（铁矿石）采集价格数据。
"""

import json
import random
import re
import time
from datetime import datetime
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from loguru import logger

from configs import settings
from pipeline.extractors.base import BaseExtractor
from pipeline.models import Commodity, MiningData, Region, SourceType
from pipeline.utils import HTMLCleaner, get_request_headers

# 题图要求的品种
METAL_PRODUCTS: List[dict] = [
    # LME: 铜、锌、镍
    {"name": "铜", "name_en": "Copper", "exchange": "LME", "commodity": Commodity.COPPER, "base_price": 72000.0},
    {"name": "锌", "name_en": "Zinc", "exchange": "LME", "commodity": Commodity.ZINC, "base_price": 22000.0},
    {"name": "镍", "name_en": "Nickel", "exchange": "LME", "commodity": Commodity.NICKEL, "base_price": 128000.0},
    # SHFE: 锂
    {"name": "碳酸锂", "name_en": "Lithium Carbonate", "exchange": "SHFE", "commodity": Commodity.LITHIUM, "base_price": 95000.0},
    # 上海钢联: 铁矿石
    {"name": "铁矿石", "name_en": "Iron Ore", "exchange": "Mysteel", "commodity": Commodity.IRON_ORE, "base_price": 800.0},
]


class PriceExtractor(BaseExtractor):
    """价格行情采集器.

    数据源:
    - LME: 铜、锌、镍
    - SHFE: 碳酸锂
    - 上海钢联/Mysteel: 铁矿石

    Example::

        extractor = PriceExtractor()
        data = extractor.extract()
    """

    LME_PRICES_URL: str = "https://www.lme.com/en/metals"
    SHFE_DATA_URL: str = "https://www.shfe.com.cn/data/dailydata/kx/kx{}.dat"
    MYSTEEL_IRON_ORE_URL: str = "https://www.mysteel.com/newapi/zx/intention/search"

    def __init__(self) -> None:
        """初始化价格采集器."""
        self._cleaner = HTMLCleaner()
        self._session = requests.Session()

    @property
    def source_name(self) -> str:
        """数据源名称."""
        return "MiningPrice"

    def extract(self) -> List[MiningData]:
        """执行价格数据采集.

        Returns:
            采集到的MiningData列表.
        """
        all_data: List[MiningData] = []

        # 采集LME数据
        try:
            lme_data = self._extract_lme()
            all_data.extend(lme_data)
        except Exception as e:
            logger.error(f"采集LME价格失败: {e}")

        # 采集SHFE数据
        try:
            shfe_data = self._extract_shfe()
            all_data.extend(shfe_data)
        except Exception as e:
            logger.error(f"采集SHFE价格失败: {e}")

        # 采集Mysteel铁矿石数据
        try:
            mysteel_data = self._extract_mysteel()
            all_data.extend(mysteel_data)
        except Exception as e:
            logger.error(f"采集Mysteel价格失败: {e}")

        # 如果外部数据不足，生成模拟数据补充（确保近30天每日每品种一条）
        if len(all_data) < 200:
            simulated = self._generate_simulated_prices(200 - len(all_data))
            all_data.extend(simulated)
            self.log_progress(
                f"外部数据不足，补充模拟数据{len(simulated)}条"
            )

        self.log_progress(f"共采集价格数据{len(all_data)}条")
        return all_data

    def _extract_lme(self) -> List[MiningData]:
        """从LME采集铜/锌/镍价格数据.

        Returns:
            价格数据列表.
        """
        data: List[MiningData] = []
        headers = get_request_headers()

        try:
            resp = self._session.get(
                self.LME_PRICES_URL, headers=headers, timeout=30
            )
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")
            tables = soup.select("table")

            for table in tables:
                rows = table.select("tr")
                for row in rows[1:]:
                    cells = row.select("td")
                    if len(cells) >= 3:
                        metal_name = cells[0].get_text(strip=True)
                        price_val = self._parse_price(cells[1].get_text(strip=True))
                        change = cells[2].get_text(strip=True) if len(cells) > 2 else ""

                        commodity = self._match_commodity(metal_name)
                        if price_val and commodity in (
                            Commodity.COPPER, Commodity.ZINC, Commodity.NICKEL,
                        ):
                            content = self._build_price_text(
                                metal_name, price_val, change, "LME"
                            )
                            data.append(MiningData(
                                source_type=SourceType.PRICE,
                                title=f"LME {metal_name} 价格行情",
                                content=content,
                                publish_date=datetime.now(),
                                commodity=commodity,
                                country_or_region=Region.GLOBAL,
                                is_mock=False,
                                metadata={
                                    "source_site": "LME",
                                    "metal": metal_name,
                                    "price": price_val,
                                    "change": change,
                                },
                            ))

            time.sleep(settings.CRAWL_DELAY)

        except Exception as e:
            logger.warning(f"LME页面解析失败: {e}")

        self.log_progress(f"LME采集{len(data)}条")
        return data

    def _extract_shfe(self) -> List[MiningData]:
        """从SHFE采集碳酸锂期货价格数据.

        Returns:
            价格数据列表.
        """
        data: List[MiningData] = []
        headers = get_request_headers()
        today = datetime.now()

        for days_ago in range(30):
            target_date = datetime(today.year, today.month, today.day)
            try:
                date_str = target_date.strftime("%Y%m%d")
                url = self.SHFE_DATA_URL.format(date_str)

                resp = self._session.get(url, headers=headers, timeout=30)
                if resp.status_code != 200:
                    continue

                try:
                    json_data = resp.json()
                except (json.JSONDecodeError, ValueError):
                    continue

                # 查找锂相关合约
                for item in self._parse_shfe_items(json_data):
                    if "锂" in item.get("name", "") or "lc" in item.get("code", "").lower():
                        content = self._build_price_text(
                            "碳酸锂", item["price"],
                            item.get("change", ""), "SHFE", target_date,
                        )
                        data.append(MiningData(
                            source_type=SourceType.PRICE,
                            title=f"SHFE 碳酸锂期货 价格行情 {target_date.strftime('%Y-%m-%d')}",
                            content=content,
                            publish_date=target_date,
                            commodity=Commodity.LITHIUM,
                            country_or_region=Region.CHINA,
                            is_mock=False,
                            metadata={
                                "source_site": "SHFE",
                                "metal": "碳酸锂",
                                "price": item["price"],
                                "change": item.get("change", ""),
                                "date": target_date.strftime("%Y-%m-%d"),
                            },
                        ))

                time.sleep(settings.CRAWL_DELAY)

            except Exception as e:
                logger.debug(f"SHFE {date_str}数据获取失败: {e}")
                continue

        self.log_progress(f"SHFE采集{len(data)}条")
        return data

    @staticmethod
    def _parse_shfe_items(json_data: dict) -> List[dict]:
        """解析SHFE返回的JSON数据.

        Args:
            json_data: SHFE JSON数据.

        Returns:
            提取的品种列表.
        """
        items: List[dict] = []
        try:
            o_cur = json_data.get("o_cur", {})
            for item in o_cur.get("data", []):
                code = item.get("INSTRUMENTID", "")
                close = item.get("CLOSEPRICE", 0)
                pre_settle = item.get("PRESETTLEPRICE", 0)
                if close:
                    change = round(float(close) - float(pre_settle), 2)
                    items.append({
                        "code": code,
                        "name": code,
                        "price": float(close),
                        "change": str(change),
                    })
        except (KeyError, TypeError, ValueError) as e:
            logger.debug(f"SHFE数据解析异常: {e}")
        return items

    def _extract_mysteel(self) -> List[MiningData]:
        """从上海钢联/Mysteel采集铁矿石价格数据.

        Returns:
            价格数据列表.
        """
        data: List[MiningData] = []
        headers = get_request_headers()
        headers["Content-Type"] = "application/json"

        try:
            payload = {
                "keyword": "铁矿石价格",
                "page": 1,
                "pageSize": 50,
            }
            resp = self._session.post(
                self.MYSTEEL_IRON_ORE_URL,
                json=payload,
                headers=headers,
                timeout=30,
            )

            if resp.status_code == 200:
                result = resp.json()
                for item in result.get("data", {}).get("list", []):
                    title = item.get("title", "")
                    content = item.get("content", title)
                    date_str = item.get("publishDate", "")

                    publish_date = self._parse_date(date_str)

                    data.append(MiningData(
                        source_type=SourceType.PRICE,
                        title=title,
                        content=content,
                        publish_date=publish_date,
                        commodity=Commodity.IRON_ORE,
                        country_or_region=Region.CHINA,
                        is_mock=False,
                        metadata={
                            "source_site": "Mysteel",
                            "metal": "铁矿石",
                        },
                    ))

        except Exception as e:
            logger.warning(f"Mysteel数据获取失败: {e}")

        self.log_progress(f"Mysteel采集{len(data)}条")
        return data

    @staticmethod
    def _generate_simulated_prices(count: int) -> List[MiningData]:
        """生成模拟价格数据，确保近30天每日每品种一条.

        Args:
            count: 需要生成的数据条数.

        Returns:
            模拟的MiningData列表.
        """
        from dateutil.relativedelta import relativedelta

        data: List[MiningData] = []
        today = datetime.now()
        idx = 0

        # 优先填充近30天
        for days_back in range(30):
            if idx >= count:
                break
            pub_date = today - relativedelta(days=days_back)
            for metal in METAL_PRODUCTS:
                if idx >= count:
                    break
                base = metal["base_price"]
                price = round(base * (1 + random.uniform(-0.05, 0.05)), 2)
                change = round(price - base, 2)
                exchange = metal["exchange"]
                date_str = pub_date.strftime("%Y-%m-%d")
                direction = "上涨" if change > 0 else "下跌" if change < 0 else "持平"

                unit = "元/吨" if exchange in ("SHFE", "Mysteel") else "美元/吨"
                content = (
                    f"{date_str} {exchange} "
                    f"{metal['name']}价格{direction}至{price}{unit}，"
                    f"较前一交易日变动{abs(change)}{unit}。"
                )

                region = Region.CHINA if exchange in ("SHFE", "Mysteel") else Region.GLOBAL

                data.append(MiningData(
                    source_type=SourceType.PRICE,
                    title=f"{exchange} {metal['name']}价格行情 {date_str}",
                    content=content,
                    publish_date=pub_date,
                    commodity=metal["commodity"],
                    country_or_region=region,
                    is_mock=True,
                    metadata={
                        "source_site": exchange,
                        "metal": metal["name"],
                        "price": price,
                        "change": str(change),
                        "simulated": True,
                        "date": date_str,
                    },
                ))
                idx += 1

        # 如果还需要更多数据，继续往前填充
        for days_back in range(30, 365):
            if idx >= count:
                break
            pub_date = today - relativedelta(days=days_back)
            for metal in METAL_PRODUCTS:
                if idx >= count:
                    break
                base = metal["base_price"]
                price = round(base * (1 + random.uniform(-0.05, 0.05)), 2)
                change = round(price - base, 2)
                exchange = metal["exchange"]
                date_str = pub_date.strftime("%Y-%m-%d")
                direction = "上涨" if change > 0 else "下跌" if change < 0 else "持平"

                unit = "元/吨" if exchange in ("SHFE", "Mysteel") else "美元/吨"
                content = (
                    f"{date_str} {exchange} "
                    f"{metal['name']}价格{direction}至{price}{unit}，"
                    f"较前一交易日变动{abs(change)}{unit}。"
                )

                region = Region.CHINA if exchange in ("SHFE", "Mysteel") else Region.GLOBAL

                data.append(MiningData(
                    source_type=SourceType.PRICE,
                    title=f"{exchange} {metal['name']}价格行情 {date_str}",
                    content=content,
                    publish_date=pub_date,
                    commodity=metal["commodity"],
                    country_or_region=region,
                    is_mock=True,
                    metadata={
                        "source_site": exchange,
                        "metal": metal["name"],
                        "price": price,
                        "change": str(change),
                        "simulated": True,
                        "date": date_str,
                    },
                ))
                idx += 1

        return data

    @staticmethod
    def _match_commodity(name: str) -> Commodity:
        """从金属名称匹配Commodity枚举.

        Args:
            name: 金属名称.

        Returns:
            Commodity枚举值.
        """
        name_lower = name.lower()
        mapping = {
            Commodity.COPPER: ["copper", "铜"],
            Commodity.ZINC: ["zinc", "锌"],
            Commodity.NICKEL: ["nickel", "镍"],
            Commodity.LITHIUM: ["lithium", "锂"],
            Commodity.IRON_ORE: ["iron ore", "iron-ore", "铁矿石"],
        }
        for commodity, keywords in mapping.items():
            for kw in keywords:
                if kw in name_lower:
                    return commodity
        return Commodity.OTHER

    @staticmethod
    def _parse_price(text: str) -> Optional[float]:
        """从文本中解析价格数值.

        Args:
            text: 包含价格的文本.

        Returns:
            价格浮点数或None.
        """
        cleaned = re.sub(r"[^\d.\-]", "", text)
        try:
            return float(cleaned)
        except ValueError:
            return None

    @staticmethod
    def _build_price_text(
        metal_name: str,
        price: float,
        change: str,
        exchange: str,
        date: Optional[datetime] = None,
    ) -> str:
        """构建价格文本描述.

        Args:
            metal_name: 金属名称.
            price: 价格.
            change: 涨跌值文本.
            exchange: 交易所.
            date: 日期.

        Returns:
            价格文本描述.
        """
        date_str = (date or datetime.now()).strftime("%Y-%m-%d")
        change_val = 0.0
        try:
            change_val = float(re.sub(r"[^\d.\-]", "", change))
        except ValueError:
            pass

        direction = "上涨" if change_val > 0 else "下跌" if change_val < 0 else "持平"

        return (
            f"{date_str} {exchange} {metal_name}价格{direction}至{price}，"
            f"较前一交易日变动{abs(change_val)}。"
        )

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
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y.%m.%d"):
            try:
                return datetime.strptime(date_str[:19], fmt)
            except (ValueError, TypeError):
                continue
        return datetime.now()
