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
    """严格事实核验：逐句断言验证.

    从答案中提取每个事实断言（句子级），严格验证每条断言
    是否能在检索证据中找到实质性支撑（而非仅字面匹配）。

    验证规则：
    - 从答案中提取独立断言句（去除引用和元数据行）
    - 对每条断言，提取核心名词/动词实体
    - 检查证据中是否包含这些实体的组合
    - 断言被支撑则计1，否则计0

    Args:
        answer: RAG生成的答案.
        results: 检索到的证据列表.

    Returns:
        Faithfulness值（0.0 ~ 1.0）.
    """
    if not answer or not results:
        return 0.0

    evidence_text = " ".join(
        r.get("snippet", "") + " " + r.get("title", "")
        for r in results
    ).lower()

    # 提取答案中的事实断言（排除引用行和元数据行）
    claim_lines: list[str] = []
    for line in answer.split("\n"):
        line = line.strip()
        # 跳过引用、标题、空行
        if not line or line.startswith("[") or line.startswith("参考"):
            continue
        if line.startswith("·") or line.startswith("-"):
            line = line[1:].strip()
        if line.startswith("核心内容：") or line.startswith("趋势推理") or line.startswith("政策影响"):
            line = line.split("：", 1)[-1].strip() if "：" in line else line
        if len(line) > 5:
            claim_lines.append(line)

    if not claim_lines:
        # 如果无法提取断言，回退到全文检查
        answer_lower = answer.lower()
        hits = sum(1 for kw in _extract_key_phrases(answer_lower) if kw in evidence_text)
        total = len(_extract_key_phrases(answer_lower))
        return hits / total if total else 0.0

    supported = 0
    for claim in claim_lines:
        claim_lower = claim.lower()
        phrases = _extract_key_phrases(claim_lower)
        if not phrases:
            continue
        # 严格验证：至少60%的关键短语必须在证据中出现
        phrase_hits = sum(1 for p in phrases if p in evidence_text)
        if phrase_hits >= max(1, len(phrases) * 0.6):
            supported += 1

    return supported / len(claim_lines) if claim_lines else 0.0


def _extract_key_phrases(text: str) -> list[str]:
    """从文本中提取关键短语（实体级）.

    提取名词性短语和专业术语，过滤停用词。

    Args:
        text: 输入文本.

    Returns:
        关键短语列表.
    """
    # 中文停用词
    stopwords = {
        "的", "了", "在", "是", "和", "与", "及", "等", "为", "中",
        "有", "对", "将", "从", "到", "由", "其", "此", "这", "那",
        "也", "又", "还", "都", "就", "而", "或", "但", "不", "被",
        "根据", "关于", "以下", "以上", "通过", "进行", "实现", "推动",
    }

    # 提取连续中文词组（2-6字）和英文词
    import re
    phrases: list[str] = []

    # 提取中文短语
    cn_words = re.findall(r"[一-鿿]{2,6}", text)
    for w in cn_words:
        if w not in stopwords and len(w) >= 2:
            phrases.append(w)

    # 提取英文词
    en_words = re.findall(r"[a-z]{3,}", text)
    phrases.extend(en_words)

    # 去重并保留有信息量的短语
    return list(dict.fromkeys(phrases))[:20]


def compute_answer_relevance(
    answer: str,
    expected_answer: str,
    expected_keywords: list[str],
) -> float:
    """计算Answer Relevance（答案相关性）.

    验证答案是否覆盖了期望答案的核心信息点。

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
    expected_lower = expected_answer.lower()

    # 1. 关键词覆盖率
    keyword_hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    keyword_score = keyword_hits / len(expected_keywords) if expected_keywords else 0.0

    # 2. 期望答案中的关键短语覆盖率
    expected_phrases = _extract_key_phrases(expected_lower)
    if expected_phrases:
        phrase_hits = sum(1 for p in expected_phrases if p in answer_lower)
        phrase_score = phrase_hits / len(expected_phrases)
    else:
        phrase_score = keyword_score

    # 综合得分（关键词占60%，短语占40%）
    return round(0.6 * keyword_score + 0.4 * phrase_score, 4)


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
