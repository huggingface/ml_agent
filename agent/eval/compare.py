"""Comparison logic for baseline-vs-candidate evaluation results."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelResult:
    model_id: str
    metrics: dict[str, float]


@dataclass(frozen=True)
class ComparisonResult:
    task_id: str
    primary_metric: str
    baseline: ModelResult
    candidate: ModelResult

    @property
    def baseline_score(self) -> float:
        return self.baseline.metrics[self.primary_metric]

    @property
    def candidate_score(self) -> float:
        return self.candidate.metrics[self.primary_metric]

    @property
    def delta(self) -> float:
        return self.candidate_score - self.baseline_score


def compare_results(
    task_id: str,
    primary_metric: str,
    baseline: ModelResult,
    candidate: ModelResult,
) -> ComparisonResult:
    return ComparisonResult(
        task_id=task_id,
        primary_metric=primary_metric,
        baseline=baseline,
        candidate=candidate,
    )
