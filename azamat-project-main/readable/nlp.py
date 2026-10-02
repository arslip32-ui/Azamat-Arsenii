"""Adapter for the user's EXISTING parser, features and normalization modules."""
from __future__ import annotations

import threading

from .core import Analysis, AnalysisError, LANGUAGES, MAX_TEXT_CHARS, align_sentences


class NLPAnalyzer:
    def __init__(self):
        # Lazy import: the UI remains available even if a model isn't installed.
        from src.features import extract_features
        from src.normalize import calculate_readability_score
        from src.parsers import ParserManager

        self._extract = extract_features
        self._calculate = calculate_readability_score
        self._manager = ParserManager()
        # A cached ParserManager is shared by Streamlit sessions. Serialize both
        # lazy model loading and inference because Stanza may mutate state.
        self._lock = threading.RLock()

    def analyze(self, text: str, lang: str) -> Analysis:
        if lang not in LANGUAGES:
            raise AnalysisError("Выберите русский, английский или казахский язык.")
        if not text.strip():
            raise AnalysisError("Сначала введите текст.")
        if len(text) > MAX_TEXT_CHARS:
            raise AnalysisError(f"Максимум {MAX_TEXT_CHARS:,} символов за один анализ.")
        with self._lock:
            doc = self._manager.parse(text, lang)
            features = list(self._extract(doc, lang))
            if len(features) != len(doc.sentences):
                raise AnalysisError(
                    "extract_features должен возвращать по одной записи на предложение "
                    "в том же порядке, что и doc.sentences."
                )
            texts, scores = [], []
            for sentence, row in zip(doc.sentences, features):
                if "sentence" in row and row["sentence"].strip() != sentence.text.strip():
                    raise AnalysisError("Порядок предложений в extract_features изменился.")
                texts.append(sentence.text)
                # Per-sentence score, NOT one score averaged over the document.
                scores.append(self._calculate(row, lang))
        if not texts:
            raise AnalysisError("В тексте не найдено предложений.")
        return Analysis(text, lang, align_sentences(text, texts, scores))

    def score_rewrite(self, text: str, lang: str) -> float:
        # If a rewrite splits a sentence, every resulting sentence must improve.
        return min(s.score for s in self.analyze(text, lang).sentences)
