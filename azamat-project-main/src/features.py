"""
features.py
============

Feature extraction for the Trilingual Readability Assessment System
(English / Russian / Kazakh).

Consumes the ``Token`` / ``Sentence`` / ``Document`` data structures
produced by ``src/parsers.py`` (spaCy for English, Stanza for Russian and
Kazakh) and turns them into numeric linguistic features for a downstream
readability model:

1. ``tree_depth``       -- dependency-tree depth (syntactic complexity)
2. ``char_density``     -- average word length in characters
3. ``syllable_count``   -- language-specific syllable estimation
4. ``pos_ratio``        -- proportion of content words (lexical density)
5. ``extract_features`` -- combines the above into one row per sentence

Python: 3.9+
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Set

# Documented interface relied upon from src/parsers.py:
#   Token:    .text (str), .upos (str, Universal POS tag),
#             .head_idx (int, index of syntactic head in sentence.tokens;
#             -1 means this token is the root)
#   Sentence: .tokens (List[Token]), .text (str, raw sentence text)
#   Document: .sentences (List[Sentence])
#   ParserManager: unified spaCy/Stanza wrapper -> returns a Document
from .parsers import Document, ParserManager, Sentence, Token  # noqa: F401

# ---------------------------------------------------------------------------
# Language-specific vowel sets (all Cyrillic comparisons upper-cased first,
# so the functions are case-insensitive with a single set each).
# ---------------------------------------------------------------------------
_ENGLISH_VOWELS: Set[str] = set("aeiouy")

# Й is always a consonant; Ь/Ъ carry no sound -- none of the three appear
# in this set, so they're correctly excluded with no special-casing.
_RUSSIAN_VOWELS: Set[str] = set("АЕЁИОУЫЭЮЯ")

# Kazakh letters that are vowels regardless of context. У and И are handled
# separately below since they're reused for vowel AND glide/consonant sounds.
_KAZAKH_STANDARD_VOWELS: Set[str] = set("АӘЕОӨҰҮЫІЭЮЯ")
_KAZAKH_AMBIGUOUS_LETTERS: Set[str] = {"У", "И"}

# Universal POS tags counted as "content words".
_CONTENT_POS_TAGS: Set[str] = {"NOUN", "VERB", "ADJ", "ADV"}

_SUPPORTED_LANGS: Set[str] = {"en", "ru", "kk"}


# ---------------------------------------------------------------------------
# 1. tree_depth
# ---------------------------------------------------------------------------
def tree_depth(sentence: Sentence) -> int:
    """Return the maximum root-to-leaf depth of a sentence's dependency tree.

    Deeper trees generally mean more syntactically complex sentences, so
    this is used as a syntactic-complexity feature.

    Rules
    -----
    * The root token (``head_idx == -1``) has depth 1.
    * Every other token's depth is ``parent_depth + 1``.
    * Returns the max depth found across all tokens (0 for an empty
      sentence).

    Depths are computed with memoized recursion (O(n) for n tokens). A
    malformed head index (out of range, self-loop, or a cycle from an
    upstream parsing error) is treated as a root rather than raising an
    exception, so one bad sentence can't crash the whole pipeline.

    Parameters
    ----------
    sentence : Sentence
        Parsed sentence; each ``Token`` must expose ``head_idx``.

    Returns
    -------
    int
        Maximum root-to-leaf path length.

    Example
    -------
    "The quick fox jumps": jumps (root, depth 1) <- fox (depth 2)
    <- quick (depth 3). ``tree_depth(sentence)`` returns 3.
    """
    tokens = sentence.tokens
    n = len(tokens)
    if n == 0:
        return 0

    depth_cache: Dict[int, int] = {}

    def _get_depth(idx: int, visiting: set) -> int:
        if idx in depth_cache:
            return depth_cache[idx]
        
        head_idx = tokens[idx].head_idx
        malformed = (
            head_idx is None
            or head_idx == -1
            or head_idx == idx
            or idx in visiting
            or not (0 <= head_idx < n)
        )
        if malformed:
            depth = 1
        else:
            visiting.add(idx)
            depth = _get_depth(head_idx, visiting) + 1
            visiting.remove(idx)

        depth_cache[idx] = depth
        return depth

    return max(_get_depth(i, set()) for i in range(n))


# ---------------------------------------------------------------------------
# 2. char_density
# ---------------------------------------------------------------------------
def char_density(sentence: Sentence) -> float:
    """Return the average number of characters per word in a sentence.

    Only alphabetic tokens count as words (``token.text.isalpha()``, which
    is Unicode-aware and works for Cyrillic and Latin scripts alike);
    punctuation tokens are excluded from both the character and word
    counts.

    Parameters
    ----------
    sentence : Sentence
        Parsed sentence containing a list of ``Token`` objects.

    Returns
    -------
    float
        ``total_characters / word_count``. Returns 0.0 if the sentence has
        no alphabetic tokens.

    Example
    -------
    "The cat sat" -> (3 + 3 + 3) / 3 = 3.0
    """
    words = [t for t in sentence.tokens if t.text.isalpha()]
    if not words:
        return 0.0
    return sum(len(t.text) for t in words) / len(words)


# ---------------------------------------------------------------------------
# 3. syllable_count and language-specific helpers
# ---------------------------------------------------------------------------
def _syllable_count_english(word: str) -> int:
    """Count English syllables as the number of vowel clusters.

    Each maximal run of consecutive vowels (a, e, i, o, u, y --
    case-insensitive) counts as one syllable. Edge case: 0 vowels -> 1.

    Example: "beautiful" -> [eau][i][u] -> 3 syllables.
    """
    word = word.lower()
    syllables = 0
    prev_was_vowel = False
    for char in word:
        is_vowel = char in _ENGLISH_VOWELS
        if is_vowel and not prev_was_vowel:
            syllables += 1
        prev_was_vowel = is_vowel
    return syllables if syllables > 0 else 1


def _syllable_count_russian(word: str) -> int:
    """Count Russian syllables by direct orthographic vowel counting.

    Counts occurrences of the 10 vowel letters {А, Е, Ё, И, О, У, Ы, Э,
    Ю, Я}. Й (always a consonant) and Ь/Ъ (silent) are simply absent from
    the set and need no special-casing. Edge case: 0 vowels -> 1.

    Example: "красивая" -> а, и, а, я -> 4 syllables.
    """
    word = word.upper()
    syllables = sum(1 for char in word if char in _RUSSIAN_VOWELS)
    return syllables if syllables > 0 else 1


def _syllable_count_kazakh(word: str) -> int:
    """Count Kazakh syllables with context-sensitive vowel detection.

    Standard vowels {А, Ә, Е, О, Ө, Ұ, Ү, Ы, І, Э, Ю, Я} always count. У
    and И are ambiguous and resolved by position:

    * Preceded by a vowel      -> consonant/glide (not counted).
    * Preceded by a consonant  -> vowel (counted).
    * At the start of the word:
        - single-letter word (e.g. "у") -> vowel.
        - followed by a vowel   -> consonant, like "й" (not counted).
        - followed by a consonant -> vowel (counted).

    Edge case: 0 vowels -> 1. The word is scanned left to right, resolving
    each letter's status as it goes, so a preceding ambiguous letter's
    *resolved* status (not just raw membership) is used for later "preceded
    by a vowel" checks.

    Examples
    --------
    "сұлу" -> ұ (standard vowel) + у (preceded by consonant л -> vowel)
    -> 2 syllables.
    "у" -> single-letter word -> 1 syllable.
    "ұйы" -> ұ (standard) + ы (standard); й is always a consonant
    -> 2 syllables.
    """
    word = word.upper()
    n = len(word)
    if n == 0:
        return 1

    vowel_flags = [False] * n
    for i, char in enumerate(word):
        if char in _KAZAKH_STANDARD_VOWELS:
            vowel_flags[i] = True
        elif char in _KAZAKH_AMBIGUOUS_LETTERS:
            if i == 0:
                if n == 1:
                    vowel_flags[i] = True
                else:
                    next_char = word[i + 1]
                    vowel_flags[i] = next_char not in _KAZAKH_STANDARD_VOWELS
            else:
                vowel_flags[i] = not vowel_flags[i - 1]
        # else: ordinary consonant / Й / Ь / Ъ -> stays False

    syllables = sum(vowel_flags)
    return syllables if syllables > 0 else 1


def syllable_count(word: str, lang: str) -> int:
    """Estimate the number of syllables in a single word.

    Dispatches to a language-specific heuristic (see the ``_syllable_
    count_*`` helpers' docstrings for the exact rules).

    Parameters
    ----------
    word : str
        A single word/token, ideally free of punctuation. Case-insensitive.
    lang : str
        One of ``"en"`` (English), ``"ru"`` (Russian), ``"kk"`` (Kazakh).

    Returns
    -------
    int
        Estimated syllable count. Always >= 1 for any input (including an
        empty string), so downstream averaging never divides by / sums a
        meaningless zero.

    Raises
    ------
    ValueError
        If ``lang`` is not one of the three supported codes.
    """
    if lang not in _SUPPORTED_LANGS:
        raise ValueError(f"Unsupported language {lang!r}; expected one of {sorted(_SUPPORTED_LANGS)}.")
    if not word:
        return 1
    if lang == "en":
        return _syllable_count_english(word)
    if lang == "ru":
        return _syllable_count_russian(word)
    return _syllable_count_kazakh(word)  # lang == "kk"


# ---------------------------------------------------------------------------
# 4. pos_ratio
# ---------------------------------------------------------------------------
def pos_ratio(sentence: Sentence) -> float:
    """Return the proportion of content words (NOUN/VERB/ADJ/ADV) in a sentence.

    Content words carry most of a sentence's meaning; a higher ratio
    (higher lexical density) tends to make text harder to read.

    Parameters
    ----------
    sentence : Sentence
        Parsed sentence; each ``Token`` must expose a Universal POS tag
        via ``token.upos``.

    Returns
    -------
    float
        ``content_word_count / total_token_count`` (punctuation and all
        other tokens count toward the denominator). Returns 0.0 for an
        empty sentence.

    Example
    -------
    "The quick fox jumps over dog." (7 tokens incl. the period): content
    words quick/fox/jumps/dog = 4 -> 4 / 7 = 0.571.
    """
    tokens = sentence.tokens
    if not tokens:
        return 0.0
    content_count = sum(1 for t in tokens if t.upos in _CONTENT_POS_TAGS)
    return content_count / len(tokens)


# ---------------------------------------------------------------------------
# 5. extract_features
# ---------------------------------------------------------------------------
def extract_features(doc: Document, lang: str) -> List[Dict]:
    """Extract the full readability feature set for every sentence in a Document.

    Main entry point for the rest of the pipeline (``normalize.py`` / the
    Streamlit app): turns a parsed ``Document`` into a list of plain
    dictionaries, one per sentence, ready for a pandas DataFrame or a
    CSV/JSON export.

    Parameters
    ----------
    doc : Document
        Parsed document with a list of ``Sentence`` objects.
    lang : str
        Language code for the whole document (``"en"``/``"ru"``/``"kk"``);
        tags each row and selects the syllable-counting rules.

    Returns
    -------
    List[Dict]
        One dict per sentence with keys: ``sentence`` (str), ``lang``
        (str), ``token_count`` (int), ``tree_depth`` (int),
        ``char_density`` (float), ``avg_syllables`` (float, mean syllable
        count across non-punctuation tokens), ``pos_ratio`` (float).
        Empty list if the document has no sentences.

    Notes
    -----
    If a sentence has zero alphabetic tokens (e.g. punctuation/numerals
    only), ``avg_syllables`` falls back to 0.0 instead of raising on
    ``statistics.mean([])``.
    """
    rows: List[Dict] = []
    for sentence in doc.sentences:
        words = [t for t in sentence.tokens if t.text.isalpha()]
        if words:
            avg_syllables = statistics.mean(syllable_count(t.text, lang) for t in words)
        else:
            avg_syllables = 0.0

        rows.append(
            {
                "sentence": sentence.text,
                "lang": lang,
                "token_count": len(sentence.tokens),
                "tree_depth": tree_depth(sentence),
                "char_density": char_density(sentence),
                "avg_syllables": avg_syllables,
                "pos_ratio": pos_ratio(sentence),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------
def _print_feature_table(rows: List[Dict]) -> None:
    """Pretty-print feature dicts as a table, using `tabulate` if installed."""
    if not rows:
        print("(no rows to display)")
        return

    headers = list(rows[0].keys())
    try:
        from tabulate import tabulate  # type: ignore

        table = [[row[h] for h in headers] for row in rows]
        print(tabulate(table, headers=headers, floatfmt=".3f", tablefmt="grid"))
        return
    except ImportError:
        pass  # fall back to manual aligned columns

    str_rows = []
    for row in rows:
        cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row.values()]
        str_rows.append(cells)

    sent_col = headers.index("sentence")
    for r in str_rows:
        if len(r[sent_col]) > 40:
            r[sent_col] = r[sent_col][:37] + "..."

    widths = [max(len(headers[i]), max((len(r[i]) for r in str_rows), default=0)) for i in range(len(headers))]

    def _fmt(cells: List[str]) -> str:
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    print(_fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for r in str_rows:
        print(_fmt(r))


def _run_demo() -> None:
    """Parse sample sentences (2 per language) and print their features.

    Assumes ``ParserManager().parse(text, lang)`` returns a ``Document``.
    If the real ``ParserManager`` exposes a different API (e.g. separate
    ``parse_english`` / ``parse_russian`` / ``parse_kazakh`` methods),
    update the ``parser_manager.parse(...)`` call below -- the feature
    functions above don't need to change either way.
    """
    sample_sentences = {
        "en": [
            "The quick brown fox jumps over the lazy dog.",
            "Reading comprehension improves with consistent daily practice.",
        ],
        "ru": [
            "Быстрая лиса прыгает через ленивую собаку.",
            "Чтение улучшает словарный запас и грамотность.",
        ],
        "kk": [
            "Жылдам түлкі жалқау иттің үстінен секіреді.",
            "Кітап оқу сөздік қорды дамытады.",
        ],
    }

    try:
        parser_manager = ParserManager()
    except Exception as exc:  # pragma: no cover - demo-only defensive path
        print(f"[demo] Could not initialize ParserManager ({exc}).")
        print("[demo] See _run_demo()'s docstring for the assumed parse() API.")
        return

    all_rows: List[Dict] = []
    for lang, sentences in sample_sentences.items():
        for text in sentences:
            try:
                document = parser_manager.parse(text, lang)  # <-- assumed API
            except Exception as exc:  # pragma: no cover - demo-only defensive path
                print(f"[demo] Failed to parse ({lang}) {text!r}: {exc}")
                continue
            all_rows.extend(extract_features(document, lang))

    _print_feature_table(all_rows)


if __name__ == "__main__":
    _run_demo()
