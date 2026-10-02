"""
normalize.py
=============

Turns per-sentence linguistic features from ``src/features.py`` into a
single 0-100 readability score per document, for the Trilingual
Readability Assessment System (English / Russian / Kazakh).

Pipeline: ``aggregate_features`` (sentence -> document means) ->
``normalize_feature`` (min-max scale to [0, 1]) ->
``calculate_readability_score`` (normalize + invert + average all 4
features) -> ``compute_readability`` (end-to-end wrapper + label).

Design notes
------------
* All four features (tree_depth, char_density, avg_syllables, pos_ratio)
  increase with text difficulty, so each is inverted
  (``1.0 - normalized_value``) before averaging: scoring "hard" on every
  feature pushes the result toward 0, "easy" on every feature toward 100.
* Features live on very different raw scales (tree_depth: small integers;
  char_density: character counts; pos_ratio: a 0-1 proportion), so each
  is min-max normalized to [0, 1] *before* combining -- otherwise a
  wide-range feature like char_density would dominate the average.
* Reference ranges are fixed, language-specific literals (below) rather
  than computed from the current batch, so a document's score doesn't
  depend on whatever else happens to be processed alongside it.
* Weighting: for this science-fair scope, all four features are weighted
  equally (25% each) -- a simple, defensible baseline. A future version
  could learn weights from the Phase 4 user-study data instead.

Python: 3.9+
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Language-specific reference ranges for min-max normalization.
# ---------------------------------------------------------------------------
REFERENCE_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    'en': {
        'char_density': (3.5, 7.3),
        'avg_syllables': (1.0, 2.9),
        'pos_ratio': (0.40, 0.85),
        'tree_depth': (3.0, 8.8),
    },
    'ru': {
        'char_density': (3.967, 8.406),
        'avg_syllables': (1.619, 3.906),
        'pos_ratio': (0.444, 0.763),
        'tree_depth': (3.0, 10.05),
    },
    'kk': {
        'char_density': (4.737, 8.024),
        'avg_syllables': (2.0, 3.531),
        'pos_ratio': (0.329, 0.827),
        'tree_depth': (2.0, 7.3),
    },
}
FEATURE_WEIGHTS: Dict[str, Dict[str, float]] = {
    'en': {'char_density': 0.15, 'avg_syllables': 0.10, 'pos_ratio': 0.20, 'tree_depth': 0.55},
    'ru': {'char_density': 0.25, 'avg_syllables': 0.25, 'pos_ratio': 0.25, 'tree_depth': 0.25},
    'kk': {'char_density': 0.25, 'avg_syllables': 0.25, 'pos_ratio': 0.25, 'tree_depth': 0.25},
}
MEAN_ERRORS: Dict[str, float] = {
    "en": 10.9,
    "ru": 20.7,
    "kk": 25.6,
}
# The 4 features combined into the readability score; all "higher = harder".
_SCORE_FEATURE_KEYS: Tuple[str, ...] = ("tree_depth", "char_density", "avg_syllables", "pos_ratio")

# Everything aggregate_features() averages. token_count is kept for
# reporting even though it isn't part of the score itself.
_AGGREGATE_KEYS: Tuple[str, ...] = ("token_count",) + _SCORE_FEATURE_KEYS

# Fixed score -> label thresholds, checked from highest to lowest.
_INTERPRETATION_BANDS: Tuple[Tuple[float, str], ...] = (
    (90, "Very Easy"),
    (75, "Easy"),
    (60, "Moderate"),
    (40, "Difficult"),
    (0, "Very Difficult"),
)


# ---------------------------------------------------------------------------
# 1. aggregate_features
# ---------------------------------------------------------------------------
def aggregate_features(features_list: List[Dict]) -> Dict:
    """Collapse sentence-level feature dicts into one document-level dict.

    Each numeric feature (``token_count``, ``tree_depth``, ``char_density``,
    ``avg_syllables``, ``pos_ratio``) is reduced to its arithmetic mean
    across all sentences. Non-numeric fields (``sentence`` text, ``lang``)
    describe individual sentences, not the document, and are dropped.

    Parameters
    ----------
    features_list : List[Dict]
        Sentence-level rows, typically from ``src.features.extract_features``.

    Returns
    -------
    Dict
        The same numeric keys, each mapped to its mean. If
        ``features_list`` is empty, or a key is missing from every row,
        that key defaults to 0.0 rather than raising -- callers (notably
        ``compute_readability``) decide what an empty document should score.
    """
    if not features_list:
        return {key: 0.0 for key in _AGGREGATE_KEYS}

    aggregated: Dict[str, float] = {}
    for key in _AGGREGATE_KEYS:
        values = [row[key] for row in features_list if key in row]
        aggregated[key] = statistics.mean(values) if values else 0.0
    return aggregated


# ---------------------------------------------------------------------------
# 2. normalize_feature
# ---------------------------------------------------------------------------
def normalize_feature(value: float, min_val: float, max_val: float) -> float:
    """Min-max normalize a value to [0, 1].

    Parameters
    ----------
    value : float
        Raw feature value to normalize.
    min_val, max_val : float
        Reference range; ``min_val`` maps to 0.0, ``max_val`` to 1.0.

    Returns
    -------
    float
        ``(value - min_val) / (max_val - min_val)``, always in [0, 1].

    Edge cases
    ----------
    * ``value`` outside ``[min_val, max_val]`` is clamped to the nearer
      bound first, so the result never goes below 0 or above 1.
    * ``min_val == max_val`` (a collapsed range, or a dataset where every
      value is identical) would otherwise divide by zero; this returns a
      neutral 0.5 instead, since a zero-width range carries no information
      to tell "easy" from "hard".
    """
    if max_val == min_val:
        return 0.5
    clamped_value = max(min_val, min(value, max_val))
    return (clamped_value - min_val) / (max_val - min_val)


# ---------------------------------------------------------------------------
# 3. calculate_readability_score
# ---------------------------------------------------------------------------
def calculate_readability_score(aggregated_features: Dict, lang: str) -> float:
    """Combine aggregated features into a single 0-100 readability score.

    For each of the 4 scoring features: min-max normalize it to [0, 1]
    using the language's reference range, then invert it
    (``1.0 - normalized``) since a higher raw value always means harder
    text (deeper syntax, longer words, more syllables, denser content
    load). The 4 inverted scores are averaged (equal 25% weights -- see
    module docstring) and scaled from [0, 1] to [0, 100].

    Parameters
    ----------
    aggregated_features : Dict
        Document-level feature means from ``aggregate_features`` (ideally
        containing ``tree_depth``, ``char_density``, ``avg_syllables``,
        ``pos_ratio``).
    lang : str
        Language code selecting the reference ranges: ``"en"``, ``"ru"``,
        or ``"kk"``.

    Returns
    -------
    float
        Readability score in [0, 100]; 100 = easiest, 0 = hardest.

    Raises
    ------
    ValueError
        If ``lang`` isn't one of the three supported languages.

    Notes
    -----
    A missing scoring key defaults to the *midpoint* of its reference
    range -- i.e. it contributes a neutral 0.5 rather than skewing the
    score or raising a KeyError.
    """
    if lang not in REFERENCE_RANGES:
        raise ValueError(f"Unsupported language {lang!r}; expected one of {sorted(REFERENCE_RANGES)}.")

    ranges = REFERENCE_RANGES[lang]
    weights = FEATURE_WEIGHTS[lang]
    score = 0.0
    for key in _SCORE_FEATURE_KEYS:
        min_val, max_val = ranges[key]
        default_midpoint = (min_val + max_val) / 2
        raw_value = aggregated_features.get(key, default_midpoint)
        normalized = normalize_feature(raw_value, min_val, max_val)
        score += ((1.0 - normalized) * weights[key])
    
    score = score * 100.0 + MEAN_ERRORS[lang]
    return max(0.0, min(score, 100.0))  # defensive clamp; already in range by construction


def get_interpretation(score: float) -> str:
    """Map a 0-100 readability score to a difficulty label.

    Bands: Very Easy [90-100], Easy [75-89], Moderate [60-74],
    Difficult [40-59], Very Difficult [0-39].
    """
    for threshold, label in _INTERPRETATION_BANDS:
        if score >= threshold:
            return label
    return _INTERPRETATION_BANDS[-1][1]  # unreachable given the 0 threshold; safety net


# ---------------------------------------------------------------------------
# 4. compute_readability
# ---------------------------------------------------------------------------
def compute_readability(doc_features: List[Dict], lang: str) -> Dict:
    """End-to-end: sentence-level features in, readability score + label out.

    Parameters
    ----------
    doc_features : List[Dict]
        Sentence-level rows for one document, typically from
        ``src.features.extract_features``.
    lang : str
        Language code: ``"en"``, ``"ru"``, or ``"kk"``.

    Returns
    -------
    Dict
        - ``readability_score`` (float) -- 0-100, rounded to 1 decimal.
        - ``interpretation`` (str) -- see ``get_interpretation``.
        - ``lang`` (str) -- echoes the input.
        - ``sentence_count`` (int) -- sentences aggregated.
        - ``aggregated_features`` (Dict) -- see ``aggregate_features``.

    Raises
    ------
    ValueError
        If ``lang`` isn't one of the three supported languages.

    Edge case: empty document
    --------------------------
    An empty ``doc_features`` would naively aggregate to all zeros, and
    every one of those zeros sits *below* every reference range's
    minimum -- which ``normalize_feature`` clamps to 0, and inversion then
    turns into 1.0 for every feature, silently producing a "perfect" 100
    ("Very Easy") score for a document with no actual content. To avoid
    that misleading result, an empty document short-circuits to a fixed,
    neutral score of 50.0 instead of running through
    ``calculate_readability_score``.
    """
    if lang not in REFERENCE_RANGES:
        raise ValueError(f"Unsupported language {lang!r}; expected one of {sorted(REFERENCE_RANGES)}.")

    aggregated = aggregate_features(doc_features)
    score = 50.0 if not doc_features else calculate_readability_score(aggregated, lang)

    return {
        "readability_score": round(score, 1),
        "interpretation": get_interpretation(score),
        "lang": lang,
        "sentence_count": len(doc_features),
        "aggregated_features": aggregated,
    }


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------
def _print_results_table(rows: List[Dict]) -> None:
    """Pretty-print compute_readability() results, using `tabulate` if installed."""
    if not rows:
        print("(no rows to display)")
        return

    flat_rows = []
    for row in rows:
        agg = row["aggregated_features"]
        flat_rows.append(
            {
                "lang": row["lang"],
                "sentences": row["sentence_count"],
                "score": row["readability_score"],
                "level": row["interpretation"],
                "tree_depth": round(agg.get("tree_depth", 0.0), 2),
                "char_density": round(agg.get("char_density", 0.0), 2),
                "avg_syllables": round(agg.get("avg_syllables", 0.0), 2),
                "pos_ratio": round(agg.get("pos_ratio", 0.0), 2),
            }
        )

    headers = list(flat_rows[0].keys())
    try:
        from tabulate import tabulate  # type: ignore

        table = [[r[h] for h in headers] for r in flat_rows]
        print(tabulate(table, headers=headers, tablefmt="grid"))
        return
    except ImportError:
        pass  # fall back to manual aligned columns

    str_rows = [[str(r[h]) for h in headers] for r in flat_rows]
    widths = [max(len(headers[i]), max(len(r[i]) for r in str_rows)) for i in range(len(headers))]

    def _fmt(cells: List[str]) -> str:
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    print(_fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for r in str_rows:
        print(_fmt(r))


def _run_demo() -> None:
    """Parse 2 sample sentences per language and score each language's text.

    Each language's 2 sentences are treated as one short document, so
    ``compute_readability`` has more than one sentence to aggregate over.
    Assumes ``ParserManager().parse(text, lang)`` returns a ``Document`` --
    see ``src/features.py`` for the same assumption and how to adjust it
    if your ``ParserManager``'s real API differs.
    """
    from .features import extract_features
    from .parsers import ParserManager

    sample_texts = {
        "en": "The quick brown fox jumps over the lazy dog. "
        "Reading comprehension improves with consistent daily practice.",
        "ru": "Быстрая лиса прыгает через ленивую собаку. "
        "Чтение улучшает словарный запас и грамотность.",
        "kk": "Жылдам түлкі жалқау иттің үстінен секіреді. "
        "Кітап оқу сөздік қорды дамытады.",
    }

    try:
        parser_manager = ParserManager()
    except Exception as exc:  # pragma: no cover - demo-only defensive path
        print(f"[demo] Could not initialize ParserManager ({exc}).")
        print("[demo] See _run_demo()'s docstring for the assumed parse() API.")
        return

    results = []
    for lang, text in sample_texts.items():
        try:
            document = parser_manager.parse(text, lang)  # <-- assumed API
        except Exception as exc:  # pragma: no cover - demo-only defensive path
            print(f"[demo] Failed to parse ({lang}) text: {exc}")
            continue
        sentence_features = extract_features(document, lang)
        results.append(compute_readability(sentence_features, lang))

    # Also demonstrate the empty-document edge case explicitly.
    results.append(compute_readability([], "en"))

    _print_results_table(results)


if __name__ == "__main__":
    _run_demo()
