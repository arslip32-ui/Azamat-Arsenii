"""
process_data.py
================

Batch-processing script for the Trilingual Readability Assessment System
(English / Russian / Kazakh). Runs the full pipeline end to end over the
sample data:

    raw sentence files -> parse -> extract_features -> compute_readability
    -> data/processed/readability_scores.csv

Usage
-----
    python -m src.process_data

Python: 3.9+
"""

from __future__ import annotations

import csv
import os
import statistics
import sys
from typing import Dict, List

from .features import extract_features
from .normalize import compute_readability
from .parsers import ParserManager

LANGUAGES: List[str] = ["en", "ru", "kk"]

CSV_FIELDNAMES: List[str] = [
    "lang",
    "sentence",
    "readability_score",
    "interpretation",
    "tree_depth",
    "char_density",
    "avg_syllables",
    "pos_ratio",
    "token_count",
]


def _warn(message: str) -> None:
    """Print a warning to stderr, prefixed for easy grepping in script output."""
    print(f"[WARN] {message}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 1. load_raw_sentences
# ---------------------------------------------------------------------------
def load_raw_sentences(lang: str) -> List[str]:
    """Load sample sentences for one language from ``data/raw/{lang}/sample_sentences.txt``.

    Parameters
    ----------
    lang : str
        Language code (``"en"``, ``"ru"``, or ``"kk"``); used only to build
        the file path, so any code with a matching data directory works.

    Returns
    -------
    List[str]
        Non-empty, whitespace-stripped lines from the file, in file order.
        Blank lines are dropped. If the file doesn't exist, a warning is
        printed to stderr and an empty list is returned rather than
        raising -- a missing language's data shouldn't stop the other two
        languages from being processed.
    """
    path = os.path.join("data", "raw", lang, "sample_sentences.txt")
    if not os.path.isfile(path):
        _warn(f"Raw sentence file not found for lang={lang!r}: {path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    return [line for line in lines if line]


# ---------------------------------------------------------------------------
# 2. process_language
# ---------------------------------------------------------------------------
def process_language(parser_manager: ParserManager, lang: str) -> List[Dict]:
    """Run the full parse -> features -> readability pipeline for one language.

    For each raw sentence, ``parser_manager.parse`` produces a ``Document``,
    ``src.features.extract_features`` turns it into one feature row per
    syntactic sentence (usually just one, since each input line is already
    a single sentence), and ``src.normalize.compute_readability`` combines
    those rows into one document-level readability score.

    Parameters
    ----------
    parser_manager : ParserManager
        A single shared parser instance (loading NLP models is expensive,
        so this is created once in ``main`` and passed in).
    lang : str
        Language code (``"en"``, ``"ru"``, or ``"kk"``).

    Returns
    -------
    List[Dict]
        One dict per successfully processed sentence, with keys ``lang``,
        ``sentence`` (the original raw text), ``readability_score``,
        ``interpretation``, ``tree_depth``, ``char_density``,
        ``avg_syllables``, ``pos_ratio``, and ``token_count``. The four
        feature values and ``token_count`` come from
        ``compute_readability``'s ``aggregated_features`` (means across
        that sentence's syntactic sub-sentences, if there were more than
        one); ``token_count`` is rounded to the nearest int for a natural
        "number of tokens" reading.

        If a sentence fails to parse or extract features (any exception),
        it is skipped with a warning printed to stderr rather than
        aborting the whole batch -- one malformed input shouldn't lose
        every other result.
    """
    sentences = load_raw_sentences(lang)
    results: List[Dict] = []

    for text in sentences:
        try:
            document = parser_manager.parse(text, lang)
            sentence_features = extract_features(document, lang)
            readability = compute_readability(sentence_features, lang)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: one bad
            # sentence must not crash the whole batch.
            _warn(f"Skipping sentence (lang={lang!r}) due to error: {text!r} -> {exc}")
            continue

        aggregated = readability["aggregated_features"]
        results.append(
            {
                "lang": lang,
                "sentence": text,
                "readability_score": readability["readability_score"],
                "interpretation": readability["interpretation"],
                "tree_depth": aggregated.get("tree_depth", 0.0),
                "char_density": aggregated.get("char_density", 0.0),
                "avg_syllables": aggregated.get("avg_syllables", 0.0),
                "pos_ratio": aggregated.get("pos_ratio", 0.0),
                "token_count": int(round(aggregated.get("token_count", 0.0))),
            }
        )

    return results


# ---------------------------------------------------------------------------
# 3. save_results_csv
# ---------------------------------------------------------------------------
def save_results_csv(results: List[Dict], output_path: str = "data/processed/readability_scores.csv") -> None:
    """Write per-sentence results to a CSV file, creating the output directory if needed.

    Prefers pandas (cleaner Unicode/quoting handling) when it's installed,
    and falls back to the standard-library ``csv`` module otherwise, so
    this script has no hard dependency on pandas being present.

    Parameters
    ----------
    results : List[Dict]
        Rows to write, each with the keys in ``CSV_FIELDNAMES`` (this is
        exactly what ``process_language`` returns).
    output_path : str
        Destination CSV path. Defaults to
        ``data/processed/readability_scores.csv``.

    Returns
    -------
    None
        Writes the file and prints a one-line summary
        (``"Processed X sentences, wrote to {output_path}"``). An empty
        ``results`` list still produces a valid header-only CSV (X = 0)
        instead of raising.
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        import pandas as pd  # type: ignore

        # Even with zero rows, build the DataFrame with the right columns
        # so the header line is still written correctly.
        df = pd.DataFrame(results, columns=CSV_FIELDNAMES)
        df.to_csv(output_path, index=False, encoding="utf-8")
    except ImportError:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(results)

    print(f"Processed {len(results)} sentences, wrote to {output_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def _summarize_by_language(results: List[Dict]) -> List[Dict]:
    """Group results by language and compute per-language mean statistics."""
    by_lang: Dict[str, List[Dict]] = {lang: [] for lang in LANGUAGES}
    for row in results:
        by_lang.setdefault(row["lang"], []).append(row)

    summary = []
    for lang in LANGUAGES:
        rows = by_lang.get(lang, [])
        if not rows:
            continue
        summary.append(
            {
                "lang": lang,
                "sentence_count": len(rows),
                "mean_readability": round(statistics.mean(r["readability_score"] for r in rows), 2),
                "mean_tree_depth": round(statistics.mean(r["tree_depth"] for r in rows), 2),
                "mean_char_density": round(statistics.mean(r["char_density"] for r in rows), 2),
                "mean_avg_syllables": round(statistics.mean(r["avg_syllables"] for r in rows), 2),
                "mean_pos_ratio": round(statistics.mean(r["pos_ratio"] for r in rows), 2),
            }
        )
    return summary


def _print_summary_table(summary_rows: List[Dict]) -> None:
    """Print the per-language summary, using `tabulate` if installed."""
    if not summary_rows:
        print("No results to summarize.")
        return

    headers = list(summary_rows[0].keys())
    try:
        from tabulate import tabulate  # type: ignore

        table = [[r[h] for h in headers] for r in summary_rows]
        print(tabulate(table, headers=headers, tablefmt="grid"))
        return
    except ImportError:
        pass  # fall back to manual aligned columns

    str_rows = [[str(r[h]) for h in headers] for r in summary_rows]
    widths = [max(len(headers[i]), max(len(r[i]) for r in str_rows)) for i in range(len(headers))]

    def _fmt(cells: List[str]) -> str:
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    print(_fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for r in str_rows:
        print(_fmt(r))


# ---------------------------------------------------------------------------
# 4. main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the full pipeline for all three languages and report results.

    Initializes a single ``ParserManager``, processes ``"en"``, ``"ru"``,
    and ``"kk"`` in turn via ``process_language``, writes the combined
    results with ``save_results_csv``, then prints a per-language summary
    table (sentence count and mean of each feature + readability score).

    A language with no data (missing file) or zero successfully processed
    sentences is simply omitted from the summary table rather than shown
    with meaningless empty statistics; if every language comes back empty,
    the CSV is still written (header-only) and a "No results to
    summarize." message is printed instead of crashing on an empty
    average.
    """
    parser_manager = ParserManager()

    all_results: List[Dict] = []
    for lang in LANGUAGES:
        all_results.extend(process_language(parser_manager, lang))

    save_results_csv(all_results)
    _print_summary_table(_summarize_by_language(all_results))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - top-level catch-all so the
        # script reports a clean error instead of a raw traceback, while
        # still exiting non-zero for any calling shell script/CI job.
        print(f"[FATAL] process_data.py failed: {exc}", file=sys.stderr)
        sys.exit(1)
