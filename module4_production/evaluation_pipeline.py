"""
Module 4 — CI/CD Evaluation Pipeline: Faithfulness & Context Precision

Automated evaluation gate that runs before any new embedding model or LLM
is promoted to production. Uses DeepEval (primary) and Ragas (secondary).

Pipeline trigger:
  - New embedding model candidate → full retrieval + generation eval
  - New LLM version → generation-only eval (faithfulness + answer relevancy)
  - Weekly scheduled run → regression check against golden dataset

Metrics tracked:
  Faithfulness        — every claim in the answer is supported by a retrieved chunk
                        Formula: supported_claims / total_claims  (want >= 0.95)
  Context Precision   — relevant chunks are ranked higher than irrelevant ones
                        Formula: precision@K weighted by rank  (want >= 0.80)
  Context Recall      — all information needed to answer is present in retrieved chunks
                        Formula: covered_ground_truths / total_ground_truths  (want >= 0.75)
  Answer Relevancy    — the answer actually addresses the question (no topic drift)
                        Formula: cosine(answer_embedding, question_embedding)  (want >= 0.85)

Thresholds are intentionally strict for a zero-hallucination legal system.
A Faithfulness drop from 0.97 to 0.94 in staging blocks the promotion.
"""

import json
from dataclasses import dataclass, field
from typing import Optional

from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import (AnswerRelevancyMetric, ContextualPrecisionMetric,
                              ContextualRecallMetric, FaithfulnessMetric)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

# ---------------------------------------------------------------------------
# 1. Thresholds — strict for legal/fiscal domain
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "faithfulness": 0.95,
    "context_precision": 0.80,
    "context_recall": 0.75,
    "answer_relevancy": 0.85,
}

# Minimum acceptable regression: if current score drops more than this
# below the production baseline, the promotion is blocked.
MAX_REGRESSION = 0.03


# ---------------------------------------------------------------------------
# 2. Golden evaluation dataset structure
# ---------------------------------------------------------------------------


@dataclass
class GoldenSample:
    """
    A single evaluation sample with ground-truth annotations.
    Created by tax law experts; stored in version-controlled JSONL file.
    """

    question: str
    expected_answer: str
    expected_citations: list[str]
    expected_context_chunks: list[str]
    doc_type: str = "legislation"
    difficulty: str = "medium"  # "easy" | "medium" | "hard"


def load_golden_dataset(path: str = "eval/golden_dataset.jsonl") -> list[GoldenSample]:
    """Loads the expert-annotated evaluation set from JSONL."""
    samples = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            samples.append(GoldenSample(**data))
    return samples


# ---------------------------------------------------------------------------
# 3. Test case builder — converts RAG outputs into DeepEval format
# ---------------------------------------------------------------------------


def build_test_case(
    sample: GoldenSample,
    actual_answer: str,
    actual_contexts: list[str],
) -> LLMTestCase:
    """
    Wraps a golden sample + system output into a DeepEval LLMTestCase.
    actual_contexts should be the raw text of the retrieved + reranked chunks.
    """
    return LLMTestCase(
        input=sample.question,
        actual_output=actual_answer,
        expected_output=sample.expected_answer,
        retrieval_context=actual_contexts,
        context=sample.expected_context_chunks,
    )


# ---------------------------------------------------------------------------
# 4. Metrics configuration
# ---------------------------------------------------------------------------


def build_metrics(model: str = "gpt-4o") -> list:
    """
    Instantiates DeepEval metrics with the evaluation LLM.
    Note: the evaluation LLM (judge) should be different from the system LLM
    to avoid self-grading bias. Use GPT-4o as judge even if the system uses
    a different model.
    """
    return [
        FaithfulnessMetric(
            threshold=THRESHOLDS["faithfulness"],
            model=model,
            include_reason=True,
        ),
        ContextualPrecisionMetric(
            threshold=THRESHOLDS["context_precision"],
            model=model,
            include_reason=True,
        ),
        ContextualRecallMetric(
            threshold=THRESHOLDS["context_recall"],
            model=model,
            include_reason=True,
        ),
        AnswerRelevancyMetric(
            threshold=THRESHOLDS["answer_relevancy"],
            model=model,
            include_reason=True,
        ),
    ]


# ---------------------------------------------------------------------------
# 5. Evaluation runner
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    faithfulness: float
    context_precision: float
    context_recall: float
    answer_relevancy: float
    passed: bool
    failures: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  Faithfulness:      {self.faithfulness:.3f}  (threshold {THRESHOLDS['faithfulness']})",
            f"  Context Precision: {self.context_precision:.3f}  (threshold {THRESHOLDS['context_precision']})",
            f"  Context Recall:    {self.context_recall:.3f}  (threshold {THRESHOLDS['context_recall']})",
            f"  Answer Relevancy:  {self.answer_relevancy:.3f}  (threshold {THRESHOLDS['answer_relevancy']})",
            f"  Result: {'PASS' if self.passed else 'FAIL'}",
        ]
        if self.failures:
            lines.append(f"  Failures: {', '.join(self.failures)}")
        return "\n".join(lines)


def run_evaluation(
    test_cases: list[LLMTestCase],
    production_baseline: Optional[dict] = None,
) -> EvalResult:
    """
    Runs the full evaluation suite and returns a structured result.
    If production_baseline is provided, also checks for regressions.

    Args:
        test_cases: built from build_test_case() for each golden sample
        production_baseline: dict of {metric_name: score} from last prod deployment

    Returns:
        EvalResult with pass/fail verdict — used by CI gate to block promotion.
    """
    metrics = build_metrics()
    dataset = EvaluationDataset(test_cases=test_cases)

    results = evaluate(dataset, metrics)

    scores = {
        "faithfulness": _avg_metric_score(results, "FaithfulnessMetric"),
        "context_precision": _avg_metric_score(results, "ContextualPrecisionMetric"),
        "context_recall": _avg_metric_score(results, "ContextualRecallMetric"),
        "answer_relevancy": _avg_metric_score(results, "AnswerRelevancyMetric"),
    }

    failures = []

    # Threshold check
    for metric, score in scores.items():
        if score < THRESHOLDS[metric]:
            failures.append(f"{metric}={score:.3f} < {THRESHOLDS[metric]}")

    # Regression check against production baseline
    if production_baseline:
        for metric, score in scores.items():
            baseline = production_baseline.get(metric, 0.0)
            if score < baseline - MAX_REGRESSION:
                failures.append(
                    f"{metric} regressed: {score:.3f} vs baseline {baseline:.3f} "
                    f"(delta {score - baseline:.3f} < -{MAX_REGRESSION})"
                )

    return EvalResult(
        faithfulness=scores["faithfulness"],
        context_precision=scores["context_precision"],
        context_recall=scores["context_recall"],
        answer_relevancy=scores["answer_relevancy"],
        passed=len(failures) == 0,
        failures=failures,
    )


def _avg_metric_score(results, metric_class_name: str) -> float:
    """Extracts the average score for a named metric across all test cases."""
    scores = []
    for test_result in results.test_results:
        for metric_result in test_result.metrics_data:
            if metric_result.name == metric_class_name:
                scores.append(metric_result.score or 0.0)
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# 6. CI/CD gate — called from GitHub Actions / GitLab CI
# ---------------------------------------------------------------------------


def ci_promotion_gate(
    candidate_test_cases: list[LLMTestCase],
    baseline_scores_path: str = "eval/production_baseline.json",
) -> bool:
    """
    Entry point for the CI promotion gate.
    Returns True (promote) or False (block).

    Usage in CI:
        python -c "
        from evaluation_pipeline import ci_promotion_gate, build_test_case, load_golden_dataset
        # ... build test cases from candidate model outputs ...
        ok = ci_promotion_gate(test_cases)
        exit(0 if ok else 1)
        "
    """
    production_baseline = None
    try:
        with open(baseline_scores_path) as f:
            production_baseline = json.load(f)
    except FileNotFoundError:
        print("No production baseline found — skipping regression check.")

    result = run_evaluation(candidate_test_cases, production_baseline)
    print("\n=== Evaluation Results ===")
    print(result.summary())

    if result.passed:
        # Save new baseline on successful promotion
        new_baseline = {
            "faithfulness": result.faithfulness,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
            "answer_relevancy": result.answer_relevancy,
        }
        with open(baseline_scores_path, "w") as f:
            json.dump(new_baseline, f, indent=2)
        print("\nNew baseline saved. Promotion approved.")
    else:
        print("\nPromotion BLOCKED. Fix the failures above before deploying.")

    return result.passed


# ---------------------------------------------------------------------------
# 7. Ragas cross-check (secondary framework)
# ---------------------------------------------------------------------------
#
# Ragas is used as a secondary check on faithfulness using its own LLM judge.
# Running two independent judges reduces false positives in either direction.
#
# from ragas import evaluate as ragas_evaluate
# from ragas.metrics import faithfulness, context_precision, context_recall
# from datasets import Dataset
#
# def ragas_check(questions, answers, contexts, ground_truths):
#     data = {
#         "question":    questions,
#         "answer":      answers,
#         "contexts":    contexts,      # list of list of str
#         "ground_truth": ground_truths,
#     }
#     dataset = Dataset.from_dict(data)
#     result = ragas_evaluate(
#         dataset,
#         metrics=[faithfulness, context_precision, context_recall],
#     )
#     return result.to_pandas()
