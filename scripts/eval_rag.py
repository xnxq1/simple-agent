#!/usr/bin/env python3
"""RAG Evaluation Script via HTTP.

Sends questions to the running agent server and collects evaluation metrics
from the response (already computed by the graph).

Usage:
    python scripts/eval_rag.py --base-url http://localhost:8000
    python scripts/eval_rag.py --concurrency 3 --output results.json
    python scripts/eval_rag.py --min-context-score 0.6 --min-faithfulness 0.7 --junit-xml reports/eval.xml
"""

import argparse
import asyncio
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

ABSTAIN_MARKERS = [
    "информация не найдена",
    "не нашел",
    "нет информации",
    "не содержит",
    "не могу",
    "невозможно",
]


@dataclass
class QuestionResult:
    """Single question evaluation result."""

    question: str
    difficulty: str
    is_positive: bool
    context_score: float | None = None
    faithfulness_score: float | None = None
    answer_relevance_score: float | None = None
    answer: str | None = None
    abstained: bool = False
    error: str | None = None


async def query_agent(client: httpx.AsyncClient, base_url: str, question: str) -> dict:
    """Send a question to the agent and get the response."""
    resp = await client.post(
        f"{base_url}/agent/query",
        params={"payload": question},
        headers={"Content-Type": "text/plain"},
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()


def extract_metrics(
    response: dict,
) -> tuple[float | None, float | None, float | None, str | None]:
    """Extract evaluation metrics from agent response."""
    ctx_score = None
    faith_score = None
    ans_score = None
    answer = response.get("answer")

    if ctx_result := response.get("context_relevance_result"):
        ctx_score = ctx_result.get("context_score")

    if gnd_result := response.get("groundness_result"):
        faith_score = gnd_result.get("faithfulness_score")

    if ans_result := response.get("answer_relevance_result"):
        ans_score = ans_result.get("score")

    return ctx_score, faith_score, ans_score, answer


def check_abstain(answer: str | None) -> bool:
    """Check if the answer contains abstain markers."""
    if not answer:
        return False
    answer_lower = answer.lower()
    return any(marker in answer_lower for marker in ABSTAIN_MARKERS)


async def evaluate_question(
    client: httpx.AsyncClient,
    base_url: str,
    question: str,
    difficulty: str,
    is_positive: bool,
) -> QuestionResult:
    """Evaluate a single question."""
    try:
        response = await query_agent(client, base_url, question)
        ctx_score, faith_score, ans_score, answer = extract_metrics(response)

        result = QuestionResult(
            question=question,
            difficulty=difficulty,
            is_positive=is_positive,
            context_score=ctx_score,
            faithfulness_score=faith_score,
            answer_relevance_score=ans_score,
            answer=answer,
        )

        if not is_positive:
            result.abstained = check_abstain(answer)

        return result
    except Exception as e:
        return QuestionResult(
            question=question,
            difficulty=difficulty,
            is_positive=is_positive,
            error=str(e),
        )


async def evaluate_dataset(
    client: httpx.AsyncClient,
    base_url: str,
    questions_data: list[dict],
    is_positive: bool,
    concurrency: int = 1,
) -> list[QuestionResult]:
    """Evaluate all questions in a dataset with concurrency control."""
    semaphore = asyncio.Semaphore(concurrency)

    async def limited_evaluate(q_item: dict) -> QuestionResult:
        async with semaphore:
            question_text = q_item["question"]
            difficulty = q_item.get("difficulty", "unknown")
            return await evaluate_question(
                client, base_url, question_text, difficulty, is_positive
            )

    tasks = [limited_evaluate(q) for q in questions_data]
    results = await asyncio.gather(*tasks)
    return results


def load_dataset(path: Path) -> list[dict]:
    """Load a dataset from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Flatten: extract all questions from all documents
    questions = []
    for doc in data:
        for q in doc.get("questions", []):
            questions.append(q)

    return questions


def compute_stats(results: list[QuestionResult], is_positive: bool) -> dict:
    """Compute statistics grouped by difficulty."""
    by_difficulty = defaultdict(list)

    for result in results:
        if result.error:
            continue  # Skip failed evaluations

        by_difficulty[result.difficulty].append(result)

    stats = {}
    for difficulty in ["easy", "medium", "hard"]:
        items = by_difficulty.get(difficulty, [])
        if not items:
            continue

        n = len(items)
        if is_positive:
            ctx_scores = [r.context_score for r in items if r.context_score is not None]
            faith_scores = [
                r.faithfulness_score for r in items if r.faithfulness_score is not None
            ]
            ans_scores = [
                r.answer_relevance_score
                for r in items
                if r.answer_relevance_score is not None
            ]

            stats[difficulty] = {
                "n": n,
                "context_score": sum(ctx_scores) / len(ctx_scores)
                if ctx_scores
                else None,
                "faithfulness_score": sum(faith_scores) / len(faith_scores)
                if faith_scores
                else None,
                "answer_relevance_score": sum(ans_scores) / len(ans_scores)
                if ans_scores
                else None,
            }
        else:
            abstain_count = sum(1 for r in items if r.abstained)
            ctx_scores = [r.context_score for r in items if r.context_score is not None]

            stats[difficulty] = {
                "n": n,
                "abstain_rate": abstain_count / n if n > 0 else 0,
                "context_score": sum(ctx_scores) / len(ctx_scores)
                if ctx_scores
                else None,
            }

    return stats


def print_summary_positive(stats: dict, total: int):
    """Print summary table for positive dataset."""
    print("\n=== POSITIVE DATASET ===\n")

    print(
        f"{'difficulty':<12} │ {'context_score':>13} │ {'faithfulness':>12} │ {'answer_rel':>10} │ {'n':>3}"
    )
    print("-" * 65)

    all_ctx = []
    all_faith = []
    all_ans = []

    for difficulty in ["easy", "medium", "hard"]:
        if difficulty not in stats:
            continue

        d = stats[difficulty]
        ctx = d.get("context_score", 0) or 0
        faith = d.get("faithfulness_score", 0) or 0
        ans = d.get("answer_relevance_score", 0) or 0
        n = d.get("n", 0)

        all_ctx.append(ctx) if ctx > 0 else None
        all_faith.append(faith) if faith > 0 else None
        all_ans.append(ans) if ans > 0 else None

        print(
            f"{difficulty:<12} │ {ctx:>13.2f} │ {faith:>12.2f} │ {ans:>10.2f} │ {n:>3}"
        )

    print("-" * 65)
    overall_ctx = sum(all_ctx) / len(all_ctx) if all_ctx else 0
    overall_faith = sum(all_faith) / len(all_faith) if all_faith else 0
    overall_ans = sum(all_ans) / len(all_ans) if all_ans else 0

    print(
        f"{'OVERALL':<12} │ {overall_ctx:>13.2f} │ {overall_faith:>12.2f} │ {overall_ans:>10.2f} │ {total:>3}"
    )


def print_summary_negative(stats: dict, total: int):
    """Print summary table for negative dataset."""
    print("\n=== NEGATIVE DATASET ===\n")

    print(
        f"{'difficulty':<12} │ {'abstain_rate':>12} │ {'context_score':>13} │ {'n':>3}"
    )
    print("-" * 55)

    all_abstain = []
    all_ctx = []

    for difficulty in ["easy", "medium", "hard"]:
        if difficulty not in stats:
            continue

        d = stats[difficulty]
        abstain_rate = d.get("abstain_rate", 0) or 0
        ctx = d.get("context_score", 0) or 0
        n = d.get("n", 0)

        all_abstain.append(abstain_rate) if abstain_rate > 0 else None
        all_ctx.append(ctx) if ctx > 0 else None

        print(f"{difficulty:<12} │ {abstain_rate:>11.0%} │ {ctx:>13.2f} │ {n:>3}")

    print("-" * 55)
    overall_abstain = sum(all_abstain) / len(all_abstain) if all_abstain else 0
    overall_ctx = sum(all_ctx) / len(all_ctx) if all_ctx else 0

    print(
        f"{'OVERALL':<12} │ {overall_abstain:>11.0%} │ {overall_ctx:>13.2f} │ {total:>3}"
    )


def write_junit_xml(
    positive_results: list[QuestionResult],
    negative_results: list[QuestionResult],
    output_path: Path,
):
    """Write JUnit XML report for CI/CD integration."""
    root = ET.Element("testsuites")

    # Positive test suite
    pos_failures = sum(1 for r in positive_results if r.error is not None)
    pos_suite = ET.SubElement(
        root,
        "testsuite",
        name="positive",
        tests=str(len(positive_results)),
        failures=str(pos_failures),
    )

    for result in positive_results:
        testcase = ET.SubElement(
            pos_suite,
            "testcase",
            name=f"[{result.difficulty}] {result.question[:60]}",
            classname="positive",
        )
        if result.error is not None:
            ET.SubElement(testcase, "failure", message=result.error)

    # Negative test suite
    neg_failures = sum(1 for r in negative_results if not r.abstained)
    neg_suite = ET.SubElement(
        root,
        "testsuite",
        name="negative",
        tests=str(len(negative_results)),
        failures=str(neg_failures),
    )

    for result in negative_results:
        testcase = ET.SubElement(
            neg_suite,
            "testcase",
            name=f"[{result.difficulty}] {result.question[:60]}",
            classname="negative",
        )
        if not result.abstained:
            answer_preview = (result.answer or "")[:100]
            failure_msg = f"Did not abstain: {answer_preview}"
            ET.SubElement(testcase, "failure", message=failure_msg)

    # Write XML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"JUnit XML report saved to {output_path}")


async def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG system via HTTP")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Server base URL",
    )
    parser.add_argument(
        "--positive",
        default="scripts/datasets/json/positive_dataset.json",
        help="Positive dataset path",
    )
    parser.add_argument(
        "--negative",
        default="scripts/datasets/json/negative_dataset.json",
        help="Negative dataset path",
    )
    parser.add_argument(
        "--skip-negative",
        action="store_true",
        help="Skip negative evaluation",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Parallel requests",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON path for raw results",
    )
    parser.add_argument(
        "--min-context-score",
        type=float,
        default=0.0,
        help="Minimum context_score for positive dataset",
    )
    parser.add_argument(
        "--min-faithfulness",
        type=float,
        default=0.0,
        help="Minimum faithfulness_score for positive dataset",
    )
    parser.add_argument(
        "--min-answer-relevance",
        type=float,
        default=0.0,
        help="Minimum answer_relevance_score for positive dataset",
    )
    parser.add_argument(
        "--min-abstain-rate",
        type=float,
        default=0.0,
        help="Minimum abstain rate for negative dataset",
    )
    parser.add_argument(
        "--junit-xml",
        help="Optional path for JUnit XML report",
    )

    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    positive_path = Path(args.positive)
    negative_path = Path(args.negative)
    concurrency = max(1, args.concurrency)

    print(f"Loading datasets...")
    try:
        positive_questions = load_dataset(positive_path)
        print(f"  Positive: {len(positive_questions)} questions")
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        positive_questions = []

    negative_questions = []
    if not args.skip_negative:
        try:
            negative_questions = load_dataset(negative_path)
            print(f"  Negative: {len(negative_questions)} questions")
        except FileNotFoundError as e:
            print(f"  Error: {e}")

    # Exit if both datasets are empty
    if not positive_questions and not negative_questions:
        print("Error: Both positive and negative datasets are empty")
        sys.exit(1)

    print(f"\nConnecting to {base_url}...")
    async with httpx.AsyncClient() as client:
        try:
            # Health check
            resp = await client.get(f"{base_url}/docs")
            print(f"Server is running (HTTP {resp.status_code})")
        except Exception as e:
            print(f"Error: Could not connect to server: {e}")
            sys.exit(1)

        all_results = []

        # Evaluate positive dataset
        positive_results = []
        if positive_questions:
            print(
                f"\nEvaluating positive dataset ({len(positive_questions)} questions)..."
            )
            positive_results = await evaluate_dataset(
                client,
                base_url,
                positive_questions,
                is_positive=True,
                concurrency=concurrency,
            )
            all_results.extend(positive_results)
            errors = sum(1 for r in positive_results if r.error)
            print(
                f"  Completed: {len(positive_results) - errors}/{len(positive_results)}"
            )

        # Evaluate negative dataset
        negative_results = []
        if negative_questions:
            print(
                f"\nEvaluating negative dataset ({len(negative_questions)} questions)..."
            )
            negative_results = await evaluate_dataset(
                client,
                base_url,
                negative_questions,
                is_positive=False,
                concurrency=concurrency,
            )
            all_results.extend(negative_results)
            errors = sum(1 for r in negative_results if r.error)
            print(
                f"  Completed: {len(negative_results) - errors}/{len(negative_results)}"
            )

        # Compute and print statistics
        positive_stats = {}
        if positive_results:
            positive_stats = compute_stats(positive_results, is_positive=True)
            print_summary_positive(positive_stats, len(positive_results))

        negative_stats = {}
        if negative_results:
            negative_stats = compute_stats(negative_results, is_positive=False)
            print_summary_negative(negative_stats, len(negative_results))

        # Compute overall metrics
        overall_ctx = 0
        overall_faith = 0
        overall_ans = 0
        overall_abstain = 0

        if positive_stats:
            ctx_scores = []
            faith_scores = []
            ans_scores = []
            for d in positive_stats.values():
                if d.get("context_score") is not None:
                    ctx_scores.append(d["context_score"])
                if d.get("faithfulness_score") is not None:
                    faith_scores.append(d["faithfulness_score"])
                if d.get("answer_relevance_score") is not None:
                    ans_scores.append(d["answer_relevance_score"])

            overall_ctx = sum(ctx_scores) / len(ctx_scores) if ctx_scores else 0
            overall_faith = sum(faith_scores) / len(faith_scores) if faith_scores else 0
            overall_ans = sum(ans_scores) / len(ans_scores) if ans_scores else 0

        if negative_stats:
            abstain_rates = []
            for d in negative_stats.values():
                if d.get("abstain_rate") is not None:
                    abstain_rates.append(d["abstain_rate"])
            overall_abstain = (
                sum(abstain_rates) / len(abstain_rates) if abstain_rates else 0
            )

        # Check thresholds
        violations = []
        if args.min_context_score > 0 and overall_ctx < args.min_context_score:
            violations.append(
                f"context_score {overall_ctx:.2f} < {args.min_context_score}"
            )
        if args.min_faithfulness > 0 and overall_faith < args.min_faithfulness:
            violations.append(
                f"faithfulness {overall_faith:.2f} < {args.min_faithfulness}"
            )
        if args.min_answer_relevance > 0 and overall_ans < args.min_answer_relevance:
            violations.append(
                f"answer_relevance {overall_ans:.2f} < {args.min_answer_relevance}"
            )
        if args.min_abstain_rate > 0 and overall_abstain < args.min_abstain_rate:
            violations.append(
                f"abstain_rate {overall_abstain:.2%} < {args.min_abstain_rate:.2%}"
            )

        # Save raw results if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Build overall stats
            overall_stats = {
                "positive": positive_stats,
                "negative": negative_stats,
            }

            # Add OVERALL key to positive stats
            if positive_stats:
                overall_stats["positive"]["OVERALL"] = {
                    "context_score": overall_ctx,
                    "faithfulness_score": overall_faith,
                    "answer_relevance_score": overall_ans,
                    "n": len(positive_results),
                }

            # Add OVERALL key to negative stats
            if negative_stats:
                overall_stats["negative"]["OVERALL"] = {
                    "abstain_rate": overall_abstain,
                    "context_score": sum(
                        d.get("context_score", 0)
                        for d in negative_stats.values()
                        if d.get("context_score")
                    ) / len([d for d in negative_stats.values() if d.get("context_score")])
                    if any(d.get("context_score") for d in negative_stats.values())
                    else None,
                    "n": len(negative_results),
                }

            output_data = {
                "questions": [asdict(r) for r in all_results],
                "stats": overall_stats,
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"\nRaw results saved to {output_path}")

        # Write JUnit XML if requested
        if args.junit_xml:
            write_junit_xml(positive_results, negative_results, Path(args.junit_xml))

        # Print verdict and exit
        print()
        if violations:
            print("[FAIL]")
            for v in violations:
                print(f"  - {v}")
            sys.exit(1)
        else:
            print("[PASS]")


if __name__ == "__main__":
    asyncio.run(main())
