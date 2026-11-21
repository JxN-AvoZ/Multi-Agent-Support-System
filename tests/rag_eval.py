"""Offline evaluation runner for the Getnet BM25-style RAG retriever.

Run from the project root:
    python tests/rag_eval.py --dataset evaluation/rag_eval.json --top-k 5

This script uses the local index and main.search(). It does not call Anthropic
or Tavily. If the index is missing/stale, main.load_index() may crawl the
configured Getnet seed URLs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import eval_metrics  # noqa: E402  (shared metric functions, same folder)
import main  # noqa: E402  (project path is added above)


DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "rag_eval.json"
DEFAULT_REPORT = PROJECT_ROOT / "evaluation" / "reports" / "rag_eval_report.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Getnet RAG retrieval against a JSON dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the JSON evaluation dataset (default: evaluation/rag_eval.json).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve for each question (default: 5).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Where to write the JSON evaluation report.",
    )
    parser.add_argument(
        "--show-text",
        action="store_true",
        help="Print a short preview of every retrieved chunk.",
    )
    parser.add_argument(
        "--min-mrr",
        type=float,
        default=None,
        help="Exit with status 1 if MRR over labeled cases is below this value (CI gate).",
    )
    parser.add_argument(
        "--min-hit-rate",
        type=float,
        default=None,
        help="Exit with status 1 if Hit@K over labeled cases is below this value (CI gate).",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    """Resolve relative paths from the project root, not the current directory."""
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _display_path(path: Path) -> str:
    """Show paths relative to the project root so reports hold no local paths."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def load_dataset(path: Path) -> dict[str, Any]:
    path = resolve_project_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, dict) or not isinstance(dataset.get("cases"), list):
        raise ValueError("Invalid dataset: expected a JSON object containing a 'cases' list.")

    for index, case in enumerate(dataset["cases"]):
        if not isinstance(case, dict):
            raise ValueError(f"Invalid case at position {index}: expected an object.")
        if not case.get("id"):
            raise ValueError(f"Invalid case at position {index}: missing 'id'.")
        if not (case.get("message") or case.get("question")):
            raise ValueError(f"Case {case['id']!r} is missing 'message' or 'question'.")

    return dataset


def initialize_retriever() -> int:
    """Load the local index and initialize the retrieval statistics used by main.search."""
    chunks = main.load_index()
    if not isinstance(chunks, list):
        raise TypeError("main.load_index() must return a list of chunks.")

    main.CHUNKS.clear()
    main.CHUNKS.extend(chunks)
    main.build_retrieval_stats(main.CHUNKS)
    return len(main.CHUNKS)


def phrase_tokens(phrase: str) -> list[str]:
    return main.words(phrase)


def phrase_present(phrase: str, document_tokens: set[str]) -> bool:
    """Return True when every normalized term in a keyword phrase is present."""
    tokens = phrase_tokens(phrase)
    return bool(tokens) and all(token in document_tokens for token in tokens)


def source_hint_overlap(hint: str, retrieved_text: str) -> dict[str, Any]:
    """Approximate source-hint overlap; useful for inspection, not a formal metric.

    Natural-language source_hints are not stable document IDs. This heuristic
    reports lexical overlap only and should not be treated as proof of relevance.
    """
    hint_tokens = [token for token in phrase_tokens(hint) if len(token) > 3]
    if not hint_tokens:
        return {"hint": hint, "matched": False, "overlap": 0.0}

    retrieved_tokens = set(main.words(retrieved_text))
    matched = [token for token in hint_tokens if token in retrieved_tokens]
    overlap = len(matched) / len(set(hint_tokens))
    return {
        "hint": hint,
        "matched": overlap >= 0.5,
        "overlap": round(overlap, 3),
        "matched_terms": matched,
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def evaluate_case(case: dict[str, Any], top_k: int, show_text: bool) -> dict[str, Any]:
    case_id = case["id"]
    message = str(case.get("message") or case.get("question") or "")

    # Optional manual query expansions let you test English questions against
    # a Portuguese knowledge base without using a paid LLM query-rewriter.
    retrieval_queries = case.get("retrieval_queries")
    if isinstance(retrieval_queries, list) and retrieval_queries:
        queries = [str(query) for query in retrieval_queries if str(query).strip()]
    else:
        queries = [message]

    retrieved = main.search(queries, k=top_k)
    combined_text = "\n".join(
        f"{item.get('url', '')}\n{item.get('text', '')}" for item in retrieved
    )
    retrieved_tokens = set(main.words(combined_text))

    expected_keywords = case.get("expected_keywords", [])
    keyword_results = [
        {
            "keyword": str(keyword),
            "matched": phrase_present(str(keyword), retrieved_tokens),
        }
        for keyword in expected_keywords
    ]
    keyword_total = len(keyword_results)
    keyword_matched = sum(item["matched"] for item in keyword_results)
    keyword_coverage = keyword_matched / keyword_total if keyword_total else None

    source_hints = case.get("source_hints", case.get("expected_sources", []))
    source_checks = [source_hint_overlap(str(hint), combined_text) for hint in source_hints]

    retrieved_documents = []
    for item in retrieved:
        document = {
            "url": item.get("url", ""),
            "text_preview": str(item.get("text", ""))[:500],
        }
        retrieved_documents.append(document)

    # Page-level ranking metrics, only for cases that carry relevance labels.
    relevant_sources = case.get("relevant_sources")
    urls = [str(item.get("url", "")) for item in retrieved]
    index_urls = {chunk["url"] for chunk in main.CHUNKS}
    labeled = bool(relevant_sources)
    ranking: dict[str, Any] = {"labeled": labeled}
    if labeled:
        ranking.update(
            {
                "precision_at_k": eval_metrics.precision_at_k(urls, relevant_sources),
                "recall_at_k": eval_metrics.recall_at_k(
                    urls, relevant_sources, index_urls, top_k
                ),
                "reciprocal_rank": eval_metrics.reciprocal_rank(urls, relevant_sources),
                "hit_at_k": eval_metrics.hit_at_k(urls, relevant_sources),
            }
        )

    result: dict[str, Any] = {
        "id": case_id,
        "message": message,
        "retrieval_queries": queries,
        "expected_topic": case.get("expected_topic"),
        "split": case.get("split", "dev"),
        "retrieved_count": len(retrieved),
        "retrieved_any": bool(retrieved),
        "keyword_matches": keyword_results,
        "keyword_coverage": round(keyword_coverage, 3) if keyword_coverage is not None else None,
        "source_hint_checks": source_checks,
        "ranking": ranking,
        "retrieved_documents": retrieved_documents,
    }

    print(f"\n[{ 'HIT' if retrieved else 'MISS' }] {case_id}")
    print(f"  query: {message}")
    if len(queries) > 1 or queries[0] != message:
        print(f"  retrieval queries: {queries}")
    print(f"  retrieved chunks: {len(retrieved)}")
    if keyword_total:
        print(f"  keyword coverage: {keyword_matched}/{keyword_total} ({keyword_coverage:.1%})")
        missing = [item["keyword"] for item in keyword_results if not item["matched"]]
        if missing:
            print(f"  missing keywords: {', '.join(missing)}")
    if labeled:
        print(
            f"  P@{top_k}={_fmt(ranking['precision_at_k'])}  "
            f"R@{top_k}={_fmt(ranking['recall_at_k'])}  "
            f"RR={_fmt(ranking['reciprocal_rank'])}"
        )
    else:
        print("  ranking metrics: n/a (no relevant page in the index for this case)")
    for rank, document in enumerate(retrieved_documents, start=1):
        print(f"  {rank}. {document['url']}")
        if show_text:
            print(f"     {document['text_preview'][:300].replace(chr(10), ' ')}")

    return result


def main_cli() -> int:
    args = parse_arguments()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    dataset_path = resolve_project_path(args.dataset)
    dataset = load_dataset(dataset_path)

    cases = dataset["cases"]
    rag_cases = [
        case for case in cases
        if case.get("expected_knowledge_mode") == "rag"
        and case.get("should_be_answerable_from_rag", True) is not False
    ]
    skipped_cases = [case for case in cases if case not in rag_cases]

    print(f"Dataset: {dataset.get('dataset_name', 'Unnamed dataset')}")
    print(f"Version: {dataset.get('version', 'unknown')}")
    print(f"Dataset cases: {len(cases)}")
    print(f"RAG cases: {len(rag_cases)}")
    print(f"Skipped non-RAG cases: {len(skipped_cases)}")
    print(f"Top-K: {args.top_k}")

    chunk_count = initialize_retriever()
    print(f"Indexed chunks available: {chunk_count}")

    if not chunk_count:
        print(
            "\nNo indexed chunks were loaded. Check SEED_URLS, INDEX_FILE, "
            "crawler logs, and network access."
        )
        return 2

    results = [evaluate_case(case, args.top_k, args.show_text) for case in rag_cases]

    total_cases = len(results)
    non_empty = sum(item["retrieved_any"] for item in results)
    keyword_values = [
        item["keyword_coverage"] for item in results
        if item["keyword_coverage"] is not None
    ]
    average_keyword_coverage = (
        sum(keyword_values) / len(keyword_values) if keyword_values else None
    )

    def summarize_ranking(items: list[dict[str, Any]]) -> dict[str, Any]:
        labeled = [item["ranking"] for item in items if item["ranking"]["labeled"]]
        return {
            "labeled_cases": len(labeled),
            "unlabeled_cases": len(items) - len(labeled),
            "mean_precision_at_k": eval_metrics.mean(r["precision_at_k"] for r in labeled),
            "mean_recall_at_k": eval_metrics.mean(r["recall_at_k"] for r in labeled),
            "mrr": eval_metrics.mean(r["reciprocal_rank"] for r in labeled),
            "hit_rate_at_k": eval_metrics.mean(float(r["hit_at_k"]) for r in labeled),
        }

    ranking_summary = {
        "label_level": "page (URL substring), written by the project author",
        "all": summarize_ranking(results),
        "by_split": {
            split: summarize_ranking([r for r in results if r["split"] == split])
            for split in sorted({r["split"] for r in results})
        },
    }

    summary = {
        "dataset_name": dataset.get("dataset_name"),
        "dataset_version": dataset.get("version"),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": _display_path(dataset_path),
        "index_chunks": chunk_count,
        "top_k": args.top_k,
        "total_dataset_cases": len(cases),
        "rag_cases_evaluated": total_cases,
        "non_rag_cases_skipped": len(skipped_cases),
        "non_empty_retrieval_rate": non_empty / total_cases if total_cases else 0.0,
        "average_keyword_coverage": average_keyword_coverage,
        "ranking": ranking_summary,
        "results": results,
    }

    report_path = resolve_project_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(summary, report_file, ensure_ascii=False, indent=2)

    print("\n" + "=" * 54)
    print("RAG EVALUATION SUMMARY")
    print("=" * 54)
    print(f"Cases evaluated: {total_cases}")
    print(f"Non-empty retrieval: {non_empty}/{total_cases} ({non_empty / total_cases:.1%})" if total_cases else "Non-empty retrieval: n/a")
    if average_keyword_coverage is not None:
        print(f"Average keyword coverage: {average_keyword_coverage:.1%}")
    else:
        print("Average keyword coverage: n/a (no expected keywords)")
    for label, block in (("all", ranking_summary["all"]), *ranking_summary["by_split"].items()):
        print(
            f"Ranking [{label}] over {block['labeled_cases']} labeled cases "
            f"({block['unlabeled_cases']} with no relevant page in the index): "
            f"P@{args.top_k}={_fmt(block['mean_precision_at_k'])} "
            f"R@{args.top_k}={_fmt(block['mean_recall_at_k'])} "
            f"MRR={_fmt(block['mrr'])} Hit@{args.top_k}={_fmt(block['hit_rate_at_k'])}"
        )
    print(f"Report: {_display_path(report_path)}")
    print("\nNotes: keyword coverage is a lexical diagnostic. Precision/Recall/MRR use")
    print("page-level relevance labels (URL substrings) written by the project author,")
    print("not an independent annotator, so they measure 'right page', not 'right sentence'.")


    overall = ranking_summary["all"]
    failures = []
    if args.min_mrr is not None and (overall["mrr"] or 0.0) < args.min_mrr:
        failures.append(f"MRR {_fmt(overall['mrr'])} < {args.min_mrr}")
    if args.min_hit_rate is not None and (overall["hit_rate_at_k"] or 0.0) < args.min_hit_rate:
        failures.append(f"Hit@{args.top_k} {_fmt(overall['hit_rate_at_k'])} < {args.min_hit_rate}")
    if failures:
        print("RETRIEVAL REGRESSION: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
