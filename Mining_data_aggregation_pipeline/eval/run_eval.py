"""自动化评估脚本.

调用本地API接口，使用ground_truth中的问答对计算Recall@5指标。
"""

import json
import sys
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from configs import settings

# 评估配置
GROUND_TRUTH_PATH: Path = Path(__file__).parent / "ground_truth.json"
API_BASE_URL: str = "http://127.0.0.1:8000"
QUERY_ENDPOINT: str = f"{API_BASE_URL}/query"
HEALTH_ENDPOINT: str = f"{API_BASE_URL}/health"
TOP_K: int = 5
TIMEOUT: float = 30.0


def load_ground_truth() -> list[dict[str, Any]]:
    """加载ground_truth测试问答对.

    Returns:
        问答对列表.

    Raises:
        FileNotFoundError: ground_truth.json文件不存在.
    """
    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(
            f"ground_truth文件不存在: {GROUND_TRUTH_PATH}"
        )

    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def check_service_available() -> bool:
    """检查API服务是否可用.

    Returns:
        服务可用返回True，否则False.
    """
    try:
        resp = httpx.get(HEALTH_ENDPOINT, timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            logger.info(
                f"服务可用，数据库数据量: {data.get('collection_count', 0)}"
            )
            return True
    except httpx.ConnectError:
        logger.error("无法连接API服务，请确认服务已启动")
    except Exception as e:
        logger.error(f"健康检查失败: {e}")

    return False


def query_api(question: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
    """调用查询API.

    Args:
        question: 查询问题.
        top_k: 返回结果数量.

    Returns:
        查询结果列表.

    Raises:
        httpx.HTTPError: 请求失败.
    """
    resp = httpx.post(
        QUERY_ENDPOINT,
        json={"question": question, "top_k": top_k},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("results", [])


def compute_recall_at_k(
    results: list[dict[str, Any]],
    expected_keywords: list[str],
) -> float:
    """计算单条查询的Recall@K.

    判断返回结果中命中了多少个期望关键词，
    Recall = 命中关键词数 / 期望关键词总数.

    Args:
        results: API返回的查询结果列表.
        expected_keywords: 期望出现的关键词列表.

    Returns:
        Recall值（0.0 ~ 1.0）.
    """
    if not expected_keywords:
        return 0.0

    # 将所有返回结果的文本合并
    combined_text = " ".join(
        r.get("snippet", "") + " " + r.get("title", "")
        for r in results
    ).lower()

    hits = 0
    for keyword in expected_keywords:
        if keyword.lower() in combined_text:
            hits += 1

    return hits / len(expected_keywords)


def run_evaluation() -> dict[str, Any]:
    """执行完整评估流程.

    Returns:
        评估结果字典，包含各条目的Recall和总体指标.
    """
    logger.info("开始评估流程")

    if not check_service_available():
        logger.error("API服务不可用，评估终止")
        sys.exit(1)

    ground_truth = load_ground_truth()
    logger.info(f"加载{len(ground_truth)}条测试用例")

    item_recalls: list[dict[str, Any]] = []
    total_recall: float = 0.0
    success_count: int = 0

    for item in ground_truth:
        qid = item["id"]
        question = item["question"]
        expected_keywords = item["expected_keywords"]

        try:
            results = query_api(question, TOP_K)
            recall = compute_recall_at_k(results, expected_keywords)
            total_recall += recall
            success_count += 1

            item_recalls.append({
                "id": qid,
                "question": question,
                "recall": round(recall, 4),
                "result_count": len(results),
                "hit_keywords": [
                    kw for kw in expected_keywords
                    if kw.lower() in " ".join(
                        r.get("snippet", "") + " " + r.get("title", "")
                        for r in results
                    ).lower()
                ],
                "miss_keywords": [
                    kw for kw in expected_keywords
                    if kw.lower() not in " ".join(
                        r.get("snippet", "") + " " + r.get("title", "")
                        for r in results
                    ).lower()
                ],
            })

            logger.info(
                f"[{qid}] Q: {question[:30]}... | "
                f"Recall@{TOP_K}: {recall:.4f}"
            )

        except Exception as e:
            logger.error(f"[{qid}] 查询失败: {e}")
            item_recalls.append({
                "id": qid,
                "question": question,
                "recall": 0.0,
                "error": str(e),
            })

    # 计算总体指标
    avg_recall = total_recall / success_count if success_count else 0.0

    report = {
        "total_questions": len(ground_truth),
        "successful_queries": success_count,
        "failed_queries": len(ground_truth) - success_count,
        "top_k": TOP_K,
        "avg_recall_at_k": round(avg_recall, 4),
        "items": item_recalls,
    }

    return report


def print_report(report: dict[str, Any]) -> None:
    """打印评估报告.

    Args:
        report: 评估结果字典.
    """
    print("\n" + "=" * 60)
    print("         矿业数据检索系统 - 评估报告")
    print("=" * 60)
    print(f"  测试问题总数:   {report['total_questions']}")
    print(f"  成功查询数:     {report['successful_queries']}")
    print(f"  失败查询数:     {report['failed_queries']}")
    print(f"  Top-K:          {report['top_k']}")
    print(f"  Avg Recall@{report['top_k']}:  {report['avg_recall_at_k']:.4f}")
    print("-" * 60)

    for item in report["items"]:
        status = "OK" if "error" not in item else "FAIL"
        print(
            f"  [{status}] #{item['id']:>2d} | "
            f"Recall: {item['recall']:.4f} | "
            f"Q: {item['question'][:35]}..."
        )

    print("=" * 60)
    print(f"  最终得分: Recall@{report['top_k']} = {report['avg_recall_at_k']:.4f}")
    print("=" * 60 + "\n")


def main() -> None:
    """评估脚本入口."""
    report = run_evaluation()
    print_report(report)

    # 保存详细报告到文件
    report_path = Path(__file__).parent / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"详细报告已保存至: {report_path}")


if __name__ == "__main__":
    main()
