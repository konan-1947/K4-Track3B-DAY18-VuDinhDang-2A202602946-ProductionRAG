from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import json
import os
import re
import sys
from dataclasses import asdict, dataclass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(tb), 1)


def _fallback_eval(questions: list[str], answers: list[str], contexts: list[list[str]], ground_truths: list[str]) -> dict:
    per_question: list[EvalResult] = []
    for q, a, ctxs, gt in zip(questions, answers, contexts, ground_truths):
        context_text = "\n".join(ctxs)
        faith = max(_overlap(a, c) for c in ctxs) if ctxs else 0.0
        relevancy = max(_overlap(a, q), _overlap(a, gt))
        precision = sum(1 for c in ctxs if _overlap(c, q) > 0 or _overlap(c, gt) > 0) / max(len(ctxs), 1)
        recall = _overlap(context_text, gt)
        per_question.append(EvalResult(q, a, ctxs, gt, faith, relevancy, precision, recall))

    def avg(name: str) -> float:
        return sum(getattr(r, name) for r in per_question) / max(len(per_question), 1)

    return {
        "faithfulness": avg("faithfulness"),
        "answer_relevancy": avg("answer_relevancy"),
        "context_precision": avg("context_precision"),
        "context_recall": avg("context_recall"),
        "per_question": per_question,
    }


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation; fallback to lexical metrics when unavailable."""
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
        df = result.to_pandas()
        per_question = [
            EvalResult(
                question=row["question"],
                answer=row["answer"],
                contexts=row["contexts"],
                ground_truth=row["ground_truth"],
                faithfulness=float(row.get("faithfulness", 0.0) or 0.0),
                answer_relevancy=float(row.get("answer_relevancy", 0.0) or 0.0),
                context_precision=float(row.get("context_precision", 0.0) or 0.0),
                context_recall=float(row.get("context_recall", 0.0) or 0.0),
            )
            for _, row in df.iterrows()
        ]
        fallback = _fallback_eval(questions, answers, contexts, ground_truths)
        fallback["per_question"] = per_question
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            fallback[metric] = sum(getattr(r, metric) for r in per_question) / max(len(per_question), 1)
        return fallback
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation unavailable, using lexical fallback: {e}")
        return _fallback_eval(questions, answers, contexts, ground_truths)


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating or answer not fully grounded", "Tighten prompt, cite context, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking, add BM25 terms, increase top_k"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filters"),
        "answer_relevancy": ("Answer does not match the question intent", "Improve prompt template and query rewriting"),
    }
    rows = []
    for r in eval_results:
        metrics = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        avg_score = sum(metrics.values()) / 4
        worst_metric = min(metrics, key=metrics.get)
        diagnosis, fix = diagnostic_tree[worst_metric]
        rows.append({
            "question": r.question,
            "ground_truth": r.ground_truth,
            "answer": r.answer,
            "worst_metric": worst_metric,
            "score": float(metrics[worst_metric]),
            "avg_score": float(avg_score),
            "diagnosis": diagnosis,
            "suggested_fix": fix,
            "error_tree": "Output đúng? → Context đủ? → Retrieval precision? → Prompt generation?",
        })
    rows.sort(key=lambda x: x["avg_score"])
    return rows[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "per_question": [asdict(r) if hasattr(r, "__dataclass_fields__") else r for r in results.get("per_question", [])],
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
