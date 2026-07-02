"""自动化评估脚本.

调用本地API接口，使用ground_truth中的问答对计算Recall@5和Answer Faithfulness指标。
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
    """
    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(
            f"ground_truth文件不存在: {GROUND_TRUTH_PATH}"
        )

    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def check_service_available() -> bool:
    """检查API服务是否可用."""
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


def query_api(question: str, top_k: int = TOP_K) -> dict[str, Any]:
    """调用查询API.

    Args:
        question: 查询问题.
        top_k: 返回结果数量.

    Returns:
        完整查询响应（含answer和results）.
    """
    resp = httpx.post(
        QUERY_ENDPOINT,
        json={"question": question, "top_k": top_k},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def compute_recall_at_k(
    results: list[dict[str, Any]],
    expected_keywords: list[str],
) -> float:
    """计算单条查询的Recall@K.

    Args:
        results: API返回的查询结果列表.
        expected_keywords: 期望出现的关键词列表.

    Returns:
        Recall值（0.0 ~ 1.0）.
    """
    if not expected_keywords:
        return 0.0

    combined_text = " ".join(
        r.get("snippet", "") + " " + r.get("title", "")
        for r in results
    ).lower()

    hits = sum(1 for kw in expected_keywords if kw.lower() in combined_text)
    return hits / len(expected_keywords)


def compute_answer_faithfulness(
    answer: str,
    results: list[dict[str, Any]],
) -> float:
    """计算Answer Faithfulness.

    检查生成的答案是否被检索到的证据所支持。
    方法：提取答案中的关键陈述，检查每条陈述是否能在证据片段中找到支撑。

    Args:
        answer: RAG生成的答案.
        results: 检索到的证据列表.

    Returns:
        Faithfulness值（0.0 ~ 1.0）.
    """
    if not answer or not results:
        return 0.0

    # 将所有证据文本合并
    evidence_text = " ".join(
        r.get("snippet", "") + " " + r.get("title", "")
        for r in results
    ).lower()

    # 从答案中提取关键片段（分句）
    sentences = [
        s.strip()
        for s in answer.replace("。", "。|").replace("，", "，|").replace(".", ".|").split("|")
        if len(s.strip()) > 5
    ]

    if not sentences:
        return 0.0

    supported = 0
    for sentence in sentences:
        # 提取句子中的关键词（去除停用词）
        words = set(sentence.lower().split())
        # 检查关键词是否在证据中出现
        keyword_hits = sum(1 for w in words if w in evidence_text)
        if keyword_hits >= max(1, len(words) * 0.3):
            supported += 1

    return supported / len(sentences) if sentences else 0.0


def compute_answer_relevance(
    answer: str,
    expected_answer: str,
    expected_keywords: list[str],
) -> float:
    """计算Answer Relevance.

    检查答案是否包含期望答案中的关键信息。

    Args:
        answer: RAG生成的答案.
        expected_answer: 期望答案.
        expected_keywords: 期望关键词列表.

    Returns:
        Relevance值（0.0 ~ 1.0）.
    """
    if not answer:
        return 0.0

    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords) if expected_keywords else 0.0


def run_evaluation() -> dict[str, Any]:
    """执行完整评估流程.

    Returns:
        评估结果字典.
    """
    logger.info("开始评估流程")

    if not check_service_available():
        logger.error("API服务不可用，评估终止")
        sys.exit(1)

    ground_truth = load_ground_truth()
    logger.info(f"加载{len(ground_truth)}条测试用例")

    item_results: list[dict[str, Any]] = []
    total_recall: float = 0.0
    total_faithfulness: float = 0.0
    total_relevance: float = 0.0
    success_count: int = 0

    for item in ground_truth:
        qid = item["id"]
        question = item["question"]
        expected_keywords = item.get("expected_keywords", [])
        expected_answer = item.get("expected_answer", "")

        try:
            response = query_api(question, TOP_K)
            results = response.get("results", [])
            answer = response.get("answer", "")

            recall = compute_recall_at_k(results, expected_keywords)
            faithfulness = compute_answer_faithfulness(answer, results)
            relevance = compute_answer_relevance(
                answer, expected_answer, expected_keywords
            )

            total_recall += recall
            total_faithfulness += faithfulness
            total_relevance += relevance
            success_count += 1

            item_results.append({
                "id": qid,
                "question": question,
                "recall": round(recall, 4),
                "faithfulness": round(faithfulness, 4),
                "relevance": round(relevance, 4),
                "answer_preview": answer[:100] + "..." if len(answer) > 100 else answer,
                "result_count": len(results),
            })

            logger.info(
                f"[{qid}] Recall: {recall:.4f} | "
                f"Faith: {faithfulness:.4f} | "
                f"Rel: {relevance:.4f} | "
                f"Q: {question[:30]}..."
            )

        except Exception as e:
            logger.error(f"[{qid}] 查询失败: {e}")
            item_results.append({
                "id": qid,
                "question": question,
                "recall": 0.0,
                "faithfulness": 0.0,
                "relevance": 0.0,
                "error": str(e),
            })

    n = success_count if success_count else 1
    report = {
        "total_questions": len(ground_truth),
        "successful_queries": success_count,
        "failed_queries": len(ground_truth) - success_count,
        "top_k": TOP_K,
        "avg_recall_at_k": round(total_recall / n, 4),
        "avg_answer_faithfulness": round(total_faithfulness / n, 4),
        "avg_answer_relevance": round(total_relevance / n, 4),
        "items": item_results,
    }

    return report


def print_report(report: dict[str, Any]) -> None:
    """打印评估报告.

    Args:
        report: 评估结果字典.
    """
    print("\n" + "=" * 70)
    print("         矿业数据检索系统 - 评估报告")
    print("=" * 70)
    print(f"  测试问题总数:       {report['total_questions']}")
    print(f"  成功查询数:         {report['successful_queries']}")
    print(f"  失败查询数:         {report['failed_queries']}")
    print(f"  Top-K:              {report['top_k']}")
    print("-" * 70)
    print(f"  Avg Recall@5:       {report['avg_recall_at_k']:.4f}")
    print(f"  Avg Faithfulness:   {report['avg_answer_faithfulness']:.4f}")
    print(f"  Avg Relevance:      {report['avg_answer_relevance']:.4f}")
    print("-" * 70)

    for item in report["items"]:
        status = "OK" if "error" not in item else "FAIL"
        print(
            f"  [{status}] #{item['id']:>2d} | "
            f"R: {item['recall']:.2f} | "
            f"F: {item['faithfulness']:.2f} | "
            f"Rel: {item['relevance']:.2f} | "
            f"Q: {item['question'][:30]}..."
        )

    print("=" * 70)
    print(
        f"  最终得分: Recall@5={report['avg_recall_at_k']:.4f}  "
        f"Faithfulness={report['avg_answer_faithfulness']:.4f}  "
        f"Relevance={report['avg_answer_relevance']:.4f}"
    )
    print("=" * 70 + "\n")


def main() -> None:
    """评估脚本入口."""
    report = run_evaluation()
    print_report(report)

    report_path = Path(__file__).parent / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"详细报告已保存至: {report_path}")


if __name__ == "__main__":
    main()
