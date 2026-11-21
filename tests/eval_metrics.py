"""Pure metric functions shared by the evaluation runners (no I/O, no model calls)."""

from __future__ import annotations

import math
import re
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Retrieval: page-level relevance labels
# ---------------------------------------------------------------------------
# A case lists `relevant_sources`: URL substrings. A retrieved chunk is relevant
# when its URL contains any of them. This is page-level, not chunk-level, so it
# says "the right page was retrieved", not "the right sentence was".

def is_relevant(url: str, relevant_sources: Iterable[str]) -> bool:
    return any(marker and marker in url for marker in relevant_sources)


def precision_at_k(urls: list[str], relevant_sources: list[str]) -> float | None:
    """Share of retrieved chunks that come from a relevant page."""
    if not urls:
        return None
    return sum(is_relevant(url, relevant_sources) for url in urls) / len(urls)


def recall_at_k(
    urls: list[str], relevant_sources: list[str], index_urls: Iterable[str], k: int
) -> float | None:
    """Distinct relevant pages retrieved / relevant pages that could be retrieved.

    The denominator is capped at k: with k chunks you cannot retrieve more than
    k distinct pages, so a perfect result is 1.0.
    """
    available = {url for url in set(index_urls) if is_relevant(url, relevant_sources)}
    if not available:
        return None
    found = {url for url in urls if url in available}
    return len(found) / min(k, len(available))


def reciprocal_rank(urls: list[str], relevant_sources: list[str]) -> float:
    for rank, url in enumerate(urls, start=1):
        if is_relevant(url, relevant_sources):
            return 1.0 / rank
    return 0.0


def hit_at_k(urls: list[str], relevant_sources: list[str]) -> bool:
    return reciprocal_rank(urls, relevant_sources) > 0.0


# ---------------------------------------------------------------------------
# Agentic behaviour
# ---------------------------------------------------------------------------

def same_agents(actual: Iterable[str], expected: Iterable[str]) -> bool:
    return set(actual) == set(expected)


def same_order(actual: list[str], expected: list[str]) -> bool:
    return list(actual) == list(expected)


def tool_scores(called: Iterable[str], expected: Iterable[str]) -> dict[str, float | None]:
    """Recall: were the expected tools called? Precision: were only they called?"""
    called_set, expected_set = set(called), set(expected)
    recall = len(called_set & expected_set) / len(expected_set) if expected_set else None
    precision = len(called_set & expected_set) / len(called_set) if called_set else None
    return {"recall": recall, "precision": precision}


def fact_coverage(answer: str, required_facts: list[str]) -> dict[str, Any] | None:
    """Each required fact is a case-insensitive regex that must match the answer."""
    if not required_facts:
        return None
    matched = [bool(re.search(fact, answer, re.IGNORECASE)) for fact in required_facts]
    return {
        "coverage": sum(matched) / len(matched),
        "missing": [fact for fact, ok in zip(required_facts, matched) if not ok],
    }


# ---------------------------------------------------------------------------
# Aggregation and judge calibration
# ---------------------------------------------------------------------------

def mean(values: Iterable[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def judge_agreement(
    judged: dict[str, dict[str, float]], human: dict[str, dict[str, float]]
) -> dict[str, Any]:
    """Compare model-judge scores with human scores on the same cases.

    Reports, per criterion, the mean absolute difference and the share of cases
    where the two are within one point. This is how you check whether the judge
    can be trusted before reading its averages.
    """
    report: dict[str, Any] = {}
    shared = sorted(set(judged) & set(human))
    criteria = sorted({c for case in shared for c in human[case] if c in judged[case]})
    for criterion in criteria:
        diffs = [
            abs(judged[case][criterion] - human[case][criterion])
            for case in shared
            if criterion in judged[case] and criterion in human[case]
            and human[case][criterion] is not None
        ]
        if diffs:
            report[criterion] = {
                "cases": len(diffs),
                "mean_abs_diff": round(sum(diffs) / len(diffs), 3),
                "within_one_point": round(sum(d <= 1 for d in diffs) / len(diffs), 3),
            }
    return report
