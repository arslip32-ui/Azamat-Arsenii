"""Pure editing logic. Nothing here depends on Streamlit or makes implicit API calls."""
from __future__ import annotations

import hashlib
import html
import json
import math
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol, Sequence

LANGUAGES = {"ru": "Russian", "en": "English", "kk": "Kazakh"}
MAX_TEXT_CHARS = 30_000
MAX_BATCH_CHARS = 6_000
MAX_BATCH_SENTENCES = 6
PROMPT_VERSION = "selective-v1"


class AnalysisError(ValueError):
    """Input cannot be analysed or aligned safely."""


class ProviderError(RuntimeError):
    """A safe, user-facing error; never contains keys or provider request bodies."""


@dataclass(frozen=True)
class SentenceScore:
    id: int
    text: str
    start: int
    end: int
    score: float


@dataclass(frozen=True)
class Analysis:
    text: str
    lang: str
    sentences: tuple[SentenceScore, ...]

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256((self.lang + "\0" + self.text).encode()).hexdigest()


@dataclass(frozen=True)
class Rewrite:
    text: str
    score: float
    target: float
    model: str


@dataclass
class Result:
    fingerprint: str
    threshold: int
    selected_ids: tuple[int, ...]
    changes: dict[int, Rewrite] = field(default_factory=dict)
    skipped: dict[int, str] = field(default_factory=dict)
    api_calls: int = 0
    cached: int = 0


class Rewriter(Protocol):
    def rewrite(self, sentences: Sequence[SentenceScore], lang: str,
                target: float) -> Mapping[int, str]: ...


def selected_sentences(analysis: Analysis, threshold: int) -> list[SentenceScore]:
    """Inclusive boundary is intentional: score == X MUST be selected."""
    if isinstance(threshold, bool) or not isinstance(threshold, int) or not 0 <= threshold <= 100:
        raise ValueError("X должен быть целым числом от 0 до 100.")
    return sorted((s for s in analysis.sentences if s.score <= threshold),
                  key=lambda s: (s.score, s.id))


def align_sentences(text: str, sentence_texts: Sequence[str],
                    scores: Sequence[float]) -> tuple[SentenceScore, ...]:
    """Find occurrences in order, not str.replace(): repeated sentences are distinct.

    Only exact source slices are editable. All gaps are retained byte-for-byte
    after UTF-8 decoding, including CRLF, tabs, repeated spaces and paragraphs.
    A parser that normalises its sentence text is rejected instead of guessed.
    """
    if len(sentence_texts) != len(scores):
        raise AnalysisError("Количество предложений и оценок не совпало.")
    cursor, result = 0, []
    for sid, (sentence, value) in enumerate(zip(sentence_texts, scores)):
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise AnalysisError("Модель вернула оценку вне диапазона 0–100.")
        # Boundary whitespace belongs to the immutable gap, never to a rewrite.
        sentence = sentence.strip()
        if not sentence:
            raise AnalysisError("Парсер вернул пустое предложение.")
        start = text.find(sentence, cursor)
        if start < 0:
            raise AnalysisError("Не удалось точно сопоставить предложения с исходным текстом.")
        end = start + len(sentence)
        result.append(SentenceScore(sid, text[start:end], start, end, value))
        cursor = end
    return tuple(result)


def split_batches(sentences: Sequence[SentenceScore]) -> list[list[SentenceScore]]:
    batches, current, size = [], [], 0
    for sentence in sentences:
        if len(sentence.text) > MAX_BATCH_CHARS:
            raise ValueError("Слишком длинное предложение для одного запроса.")
        if current and (size + len(sentence.text) > MAX_BATCH_CHARS
                        or len(current) >= MAX_BATCH_SENTENCES):
            batches.append(current)
            current, size = [], 0
        current.append(sentence)
        size += len(sentence.text)
    if current:
        batches.append(current)
    return batches


def decode_response(content: str, expected_ids: set[int]) -> dict[int, str]:
    """JSON mode still requires validation: unknown/duplicate IDs are unsafe."""
    if len(content) > MAX_BATCH_CHARS * 8:
        raise ValueError("Ответ превышает допустимую длину.")
    data = json.loads(content)
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise ValueError("Ожидался JSON с массивом items.")
    result = {}
    for item in data["items"]:
        if not isinstance(item, dict):
            raise ValueError("Некорректный элемент ответа.")
        sid, text = item.get("id"), item.get("text")
        if type(sid) is not int or sid not in expected_ids or sid in result:
            raise ValueError("Неизвестный или повторяющийся ID предложения.")
        if not isinstance(text, str):
            raise ValueError("Перефраз должен быть строкой.")
        result[sid] = text.strip()
    # Missing IDs are deliberately handled per sentence by simplify().
    return result


def protected_content(text: str) -> tuple[Counter, Counter, Counter]:
    """Conservative checks, NOT a claim of full semantic equivalence."""
    numbers = Counter(re.findall(r"[+−-]?\d+(?:[.,]\d+)*%?", text))
    citations = Counter(re.findall(r"\[\d+(?:\s*[,;–-]\s*\d+)*\]", text))
    urls = Counter(re.findall(r"https?://[^\s<>\"']+", text))
    return numbers, citations, urls


def validate_candidate(original: str, candidate: str) -> str | None:
    if not candidate.strip():
        return "Модель вернула пустой ответ."
    if candidate == original:
        return "Модель оставила предложение без изменений."
    if len(candidate) > max(300, len(original) * 3):
        return "Ответ оказался слишком длинным."
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", candidate):
        return "В ответе обнаружены служебные символы."
    if protected_content(original) != protected_content(candidate):
        return "В ответе изменились числа, ссылки или цитирования."
    return None


def simplify(analysis: Analysis, threshold: int, rewriter: Rewriter,
             score_text: Callable[[str, str], float],
             cache: OrderedDict, model: str,
             progress: Callable[[int, int], None] | None = None) -> Result:
    """One bounded pass, always from the original text. Easy text never reaches API.

    A candidate is accepted only if its LOCAL score improves. score_text should
    return the minimum score when one original becomes several sentences.
    Model responses can miss the requested target; no endless regeneration loop.
    Cache belongs to one UI session, never to a global cross-user cache.
    """
    selected = selected_sentences(analysis, threshold)
    result = Result(analysis.fingerprint, threshold, tuple(s.id for s in selected))
    if not selected:
        return result
    target = min(100, threshold + 15)
    pending = []
    prefix = (PROMPT_VERSION, analysis.fingerprint, model)
    for s in selected:
        key = (*prefix, s.id)
        previous = cache.get(key)
        if previous and previous.score > s.score and (
            previous.target >= target or previous.score >= target
        ):
            result.changes[s.id] = previous
            result.cached += 1
            cache.move_to_end(key)
        elif len(s.text) > MAX_BATCH_CHARS:
            result.skipped[s.id] = "Предложение длиннее 6 000 символов; разделите его вручную."
        else:
            pending.append(s)
    total = len(selected)
    done = total - len(pending)
    if progress:
        progress(done, total)
    batches = split_batches(pending)
    for batch_index, batch in enumerate(batches):
        try:
            result.api_calls += 1
            proposals = rewriter.rewrite(batch, analysis.lang, target)
        except ProviderError as exc:
            # Authentication/rate-limit/network error: stop, keep completed work.
            for rest in batches[batch_index:]:
                for s in rest:
                    result.skipped[s.id] = str(exc)
            if progress:
                progress(total, total)
            break
        except (ValueError, TypeError, KeyError):
            for s in batch:
                result.skipped[s.id] = "Некорректный ответ модели; оригинал сохранён."
            done += len(batch)
            if progress:
                progress(done, total)
            continue
        for s in batch:
            candidate = proposals.get(s.id, "")
            reason = validate_candidate(s.text, candidate)
            if reason is None:
                try:
                    candidate_score = float(score_text(candidate, analysis.lang))
                    if not math.isfinite(candidate_score) or not 0 <= candidate_score <= 100:
                        raise ValueError("Invalid score")
                except Exception:
                    reason = "Не удалось проверить читаемость нового предложения."
                else:
                    if candidate_score <= s.score:
                        reason = "По локальной оценке предложение не стало проще."
            if reason:
                result.skipped[s.id] = reason
            else:
                rewrite = Rewrite(candidate, candidate_score, target, model)
                result.changes[s.id] = rewrite
                cache[(*prefix, s.id)] = rewrite
                cache.move_to_end((*prefix, s.id))
                while len(cache) > 512:
                    cache.popitem(last=False)
            done += 1
            if progress:
                progress(done, total)
    return result


def assemble_text(analysis: Analysis, changes: Mapping[int, Rewrite]) -> str:
    parts, cursor = [], 0
    for sentence in analysis.sentences:
        parts.append(analysis.text[cursor:sentence.start])
        replacement = changes.get(sentence.id)
        parts.append(replacement.text if replacement else sentence.text)
        cursor = sentence.end
    parts.append(analysis.text[cursor:])
    return "".join(parts)


def render_text(analysis: Analysis, threshold: int,
                changes: Mapping[int, Rewrite] | None = None) -> str:
    """HTML-escape EVERY user/provider string. No Markdown interpretation."""
    original_view = changes is None
    changes = changes or {}
    parts, cursor = [], 0
    for sentence in analysis.sentences:
        parts.append(html.escape(analysis.text[cursor:sentence.start]))
        replacement = changes.get(sentence.id)
        display = html.escape(replacement.text if replacement else sentence.text)
        if original_view and sentence.score <= threshold:
            title = f"Предложение {sentence.id + 1} · {sentence.score:.2f}/100 · выбрано для упрощения"
            display = f'<mark class="hard" tabindex="0" title="{title}">{display}</mark>'
        elif replacement:
            title = f"Предложение {sentence.id + 1} · {sentence.score:.2f} → {replacement.score:.2f}/100"
            display = f'<mark class="changed" tabindex="0" title="{title}">{display}</mark>'
        parts.append(display)
        cursor = sentence.end
    parts.append(html.escape(analysis.text[cursor:]))
    return '<div class="document-text" dir="auto">' + "".join(parts) + "</div>"
