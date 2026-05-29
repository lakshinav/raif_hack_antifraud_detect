"""Validation pipeline with macro F1 scoring for risk detection model."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
import time
import typing
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.client import OpenRouterClient
from app.models import (
    CLEAN_CATEGORY,
    PossibleRiskCategory,
    RiskClassifierDecision,
    process_risk_detection,
)


@typing.final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ValidationSession:
    """Single session from validation dataset."""

    session_id: str
    messages: list[dict[str, str]]
    expected_category: PossibleRiskCategory


@typing.final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ValidationResult:
    """Result of single session validation."""

    session_id: str
    expected_category: PossibleRiskCategory
    predicted_category: PossibleRiskCategory
    processing_time_ms: int
    is_correct: bool
    explanation: str | None = None


@typing.final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ClassMetrics:
    """Metrics for a single class."""

    class_name: str
    true_positives: int
    false_positives: int
    false_negatives: int
    support: int

    @property
    def precision(self) -> float:
        """Calculate precision for this class."""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        """Calculate recall for this class."""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1_score(self) -> float:
        """Calculate F1 score for this class."""
        precision_value = self.precision
        recall_value = self.recall
        if precision_value + recall_value == 0:
            return 0.0
        return 2 * (precision_value * recall_value) / (precision_value + recall_value)


@typing.final
@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ValidationReport:
    """Complete validation report with all metrics."""

    total_sessions: int
    correct_predictions: int
    accuracy: float
    macro_f1: float
    per_class_metrics: dict[str, ClassMetrics]
    results: list[ValidationResult]
    total_processing_time_ms: int


def format_dialogue_from_messages(messages: list[dict[str, str]]) -> str:
    """Format dialogue messages into single text block."""
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def extract_expected_category(expected_red_flags: list[dict[str, str]]) -> PossibleRiskCategory:
    """Extract expected category from ground truth, defaulting to clean."""
    if not expected_red_flags:
        return CLEAN_CATEGORY
    return typing.cast("PossibleRiskCategory", expected_red_flags[0]["category"])


def load_validation_dataset(dataset_path: Path) -> list[ValidationSession]:
    """Load and parse validation dataset from JSON file."""
    with dataset_path.open(encoding="utf-8") as dataset_file:
        raw_sessions: list[dict[str, typing.Any]] = json.load(dataset_file)

    validation_sessions: list[ValidationSession] = []
    for raw_session in raw_sessions:
        session = ValidationSession(
            session_id=raw_session["session_id"],
            messages=raw_session["messages"],
            expected_category=extract_expected_category(raw_session.get("expected_red_flags", [])),
        )
        validation_sessions.append(session)

    return validation_sessions


async def validate_single_session(
    llm_client: OpenRouterClient,
    session: ValidationSession,
) -> ValidationResult:
    """Process single session and return validation result with explanation."""
    dialogue_text = format_dialogue_from_messages(session.messages)
    start_time = time.perf_counter()

    # Use full pipeline including local rules
    risk_result = await process_risk_detection(
        llm_client,
        dialogue_text,
    )

    processing_time_ms = int((time.perf_counter() - start_time) * 1000)

    # Convert RiskDetectionResult to category
    predicted_category: PossibleRiskCategory = (
        risk_result["category"] if risk_result else CLEAN_CATEGORY
    )
    # Local rules don't provide explanations, so we set it to None
    explanation_value: str | None = None

    return ValidationResult(
        session_id=session.session_id,
        expected_category=session.expected_category,
        predicted_category=predicted_category,
        processing_time_ms=processing_time_ms,
        is_correct=predicted_category == session.expected_category,
        explanation=explanation_value,
    )


async def validate_all_sessions(
    llm_client: OpenRouterClient,
    sessions: list[ValidationSession],
) -> list[ValidationResult]:
    """Process all sessions concurrently and return results."""
    validation_tasks = [
        validate_single_session(llm_client, session)
        for session in sessions
    ]
    return await asyncio.gather(*validation_tasks)


def compute_class_metrics(
    all_categories: set[str],
    results: list[ValidationResult],
) -> dict[str, ClassMetrics]:
    """Compute per-class metrics from validation results."""
    per_class_metrics: dict[str, ClassMetrics] = {}

    for category in all_categories:
        true_positives = sum(
            1
            for result in results
            if result.expected_category == category and result.predicted_category == category
        )
        false_positives = sum(
            1
            for result in results
            if result.expected_category != category and result.predicted_category == category
        )
        false_negatives = sum(
            1
            for result in results
            if result.expected_category == category and result.predicted_category != category
        )
        support = sum(1 for result in results if result.expected_category == category)

        per_class_metrics[category] = ClassMetrics(
            class_name=category,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            support=support,
        )

    return per_class_metrics


def build_validation_report(results: list[ValidationResult]) -> ValidationReport:
    """Build complete validation report from individual results."""
    all_categories: set[str] = set()
    for result in results:
        all_categories.add(result.expected_category)
        all_categories.add(result.predicted_category)

    per_class_metrics = compute_class_metrics(all_categories, results)

    correct_predictions = sum(1 for result in results if result.is_correct)
    total_processing_time_ms = sum(result.processing_time_ms for result in results)

    macro_f1 = (
        sum(metrics.f1_score for metrics in per_class_metrics.values()) / len(per_class_metrics)
        if per_class_metrics
        else 0.0
    )

    return ValidationReport(
        total_sessions=len(results),
        correct_predictions=correct_predictions,
        accuracy=correct_predictions / len(results) if results else 0.0,
        macro_f1=macro_f1,
        per_class_metrics=per_class_metrics,
        results=results,
        total_processing_time_ms=total_processing_time_ms,
    )


def print_validation_report(report: ValidationReport) -> None:  # noqa: T201
    """Print formatted validation report to console."""
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    print(f"\nTotal sessions: {report.total_sessions}")
    print(f"Correct predictions: {report.correct_predictions} ({report.accuracy:.1%})")
    print(f"Macro F1 Score: {report.macro_f1:.4f}")
    print(f"Total processing time: {report.total_processing_time_ms}ms")
    print(f"Average processing time: {report.total_processing_time_ms / report.total_sessions:.0f}ms")

    print("\n" + "-" * 60)
    print("PER-CLASS METRICS")
    print("-" * 60)
    print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 60)

    for class_name in sorted(report.per_class_metrics.keys()):
        metrics = report.per_class_metrics[class_name]
        print(
            f"{class_name:<25} "
            f"{metrics.precision:>10.4f} "
            f"{metrics.recall:>10.4f} "
            f"{metrics.f1_score:>10.4f} "
            f"{metrics.support:>10}"
        )

    print("\n" + "-" * 60)
    print("CONFUSION MATRIX")
    print("-" * 60)

    sorted_categories = sorted(report.per_class_metrics.keys())
    max_category_len = max(len(cat) for cat in sorted_categories)

    header = " " * (max_category_len + 2)
    for category in sorted_categories:
        header += f"{category[:8]:>10}"
    print(header)

    for true_category in sorted_categories:
        row = f"{true_category:<{max_category_len}} |"
        for predicted_category in sorted_categories:
            count = sum(
                1
                for result in report.results
                if result.expected_category == true_category
                and result.predicted_category == predicted_category
            )
            row += f"{count:>10}"
        print(row)

    misclassified = [result for result in report.results if not result.is_correct]
    if misclassified:
        print("\n" + "-" * 60)
        print("MISCLASSIFIED SESSIONS")
        print("-" * 60)
        for result in misclassified:
            print(
                f"  {result.session_id}: "
                f"expected={result.expected_category}, "
                f"predicted={result.predicted_category} "
                f"({result.processing_time_ms}ms)"
            )
            if result.explanation:
                print(f"    Explanation: {result.explanation}")

    print("\n" + "-" * 60)
    print("ALL SESSION EXPLANATIONS")
    print("-" * 60)
    for result in report.results:
        status_marker = "✓" if result.is_correct else "✗"
        print(
            f"  [{status_marker}] {result.session_id}: "
            f"predicted={result.predicted_category}"
        )
        if result.explanation:
            print(f"      {result.explanation}")

    print("\n" + "=" * 60)


def export_report_to_json(report: ValidationReport, output_path: Path) -> None:
    """Export validation report to JSON file."""
    report_data = {
        "total_sessions": report.total_sessions,
        "correct_predictions": report.correct_predictions,
        "accuracy": report.accuracy,
        "macro_f1": report.macro_f1,
        "total_processing_time_ms": report.total_processing_time_ms,
        "per_class_metrics": {
            class_name: {
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "support": metrics.support,
                "true_positives": metrics.true_positives,
                "false_positives": metrics.false_positives,
                "false_negatives": metrics.false_negatives,
            }
            for class_name, metrics in report.per_class_metrics.items()
        },
        "results": [
            {
                "session_id": result.session_id,
                "expected_category": result.expected_category,
                "predicted_category": result.predicted_category,
                "is_correct": result.is_correct,
                "processing_time_ms": result.processing_time_ms,
                "explanation": result.explanation,
            }
            for result in report.results
        ],
    }

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(report_data, output_file, indent=2, ensure_ascii=False)


async def run_validation_pipeline(
    dataset_path: Path,
    output_path: Path | None,
) -> int:
    """Run complete validation pipeline and return exit code."""
    print(f"Loading dataset from {dataset_path}...")
    validation_sessions = load_validation_dataset(dataset_path)
    print(f"Loaded {len(validation_sessions)} sessions")

    print("Initializing LLM client...")
    llm_client = OpenRouterClient()

    print("Processing sessions...")
    validation_results = await validate_all_sessions(llm_client, validation_sessions)

    print("Computing metrics...")
    validation_report = build_validation_report(validation_results)

    print_validation_report(validation_report)

    if output_path:
        export_report_to_json(validation_report, output_path)
        print(f"\nReport exported to: {output_path}")

    return 0 if validation_report.accuracy == 1.0 else 1


def parse_command_line_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    argument_parser = argparse.ArgumentParser(
        description="Validate risk detection model against labeled dataset with macro F1 scoring",
    )
    argument_parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/train.json"),
        help="Path to validation dataset JSON file (default: data/train.json)",
    )
    argument_parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Path to export JSON report (optional)",
    )
    return argument_parser.parse_args()


async def main() -> int:
    """Run main entry point for validation pipeline."""
    arguments = parse_command_line_arguments()
    return await run_validation_pipeline(arguments.dataset_path, arguments.output_path)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
