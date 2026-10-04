"""Run from the existing project root: python -m streamlit run app.py"""
from __future__ import annotations
import gc
import torch
import html
import os
import statistics
from collections import OrderedDict
from pathlib import Path

import streamlit as st

from readable.core import (
    AnalysisError, MAX_TEXT_CHARS, Result, assemble_text, render_text,
    selected_sentences, simplify,
)
from readable.examples import EXAMPLES, LEVEL_NAMES, SAMPLE_TEXTS
from readable.nlp import NLPAnalyzer
from readable.provider import DEFAULT_MODEL, GroqRewriter

st.set_page_config(page_title="READABLE. — текст проще", page_icon="📖",
                   layout="wide", initial_sidebar_state="expanded")

LANG_LABELS = {"ru": "Русский", "en": "English", "kk": "Қазақша"}
ASSETS = Path(__file__).parent / "readable" / "assets"


@st.cache_resource(show_spinner=False)
def get_analyzer() -> NLPAnalyzer:
    return NLPAnalyzer()


def setting(name: str, fallback: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value
    try:
        return str(st.secrets.get(name, fallback))
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return fallback


def init_state() -> None:
    defaults = {
        "source_text": "", "language": "ru", "threshold": 40,
        "example_level": 60, "view": "Редактировать", "api_key": "",
        "analysis": None, "result": None, "notice": None,
        "rewrite_cache": OrderedDict(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def request_action(action: str) -> None:
    st.session_state["pending_action"] = action


def load_example() -> None:
    st.session_state.source_text = SAMPLE_TEXTS[st.session_state.language]
    st.session_state.analysis = None
    st.session_state.result = None
    st.session_state.notice = None
    st.session_state.view = "Редактировать"


def clear_text() -> None:
    st.session_state.source_text = ""
    st.session_state.analysis = None
    st.session_state.result = None
    st.session_state.notice = None
    st.session_state.view = "Редактировать"
    st.session_state.rewrite_cache.clear()
    gc.collect()


def run_pending_action() -> None:
    state = st.session_state
    current = state.analysis
    if current and (current.text != state.source_text or current.lang != state.language):
        state.analysis, state.result, state.notice = None, None, None
        state.view = "Редактировать"
    action = state.pop("pending_action", None)
    if not action:
        return
    state.notice = None
    try:
        if state.analysis is None:
            with st.spinner("Разбираем текст. Первый запуск языка может занять больше времени…"):
                state.analysis = get_analyzer().analyze(state.source_text, state.language)
        state.view = "Подсветка"
        if action == "analyze":
            return
        selected = selected_sentences(state.analysis, state.threshold)
        if not selected:
            state.result = Result(state.analysis.fingerprint, state.threshold, ())
            state.notice = ("success", "Предложений с оценкой ≤ X нет. Текст уже подходит под выбранный порог.")
            return
        api_key = state.api_key.strip() or setting("GROQ_API_KEY").strip()
        if not api_key:
            state.notice = ("warning", "Для упрощения укажите Groq API key в настройках ИИ слева. Анализ уже готов.")
            return
        from groq import Groq

        model = setting("GROQ_MODEL", DEFAULT_MODEL)
        bar = st.progress(0.0, text=f"Упрощаем выбранные предложения: {len(selected)}")
        try:
            # max_retries=5 позволяет автоматом переждать лимиты Groq (Rate Limits)
            with Groq(api_key=api_key, timeout=60.0, max_retries=5) as client:
                state.result = simplify(
                    state.analysis, state.threshold, GroqRewriter(client, model),
                    get_analyzer().score_rewrite, state.rewrite_cache, model,
                    progress=lambda done, total: bar.progress(
                        done / total, text=f"Обработано {done} из {total} выбранных предложений"
                    ),
                )
        finally:
            bar.empty()
            gc.collect()
    except AnalysisError as exc:
        state.notice = ("warning", str(exc))
    except (ModuleNotFoundError, ImportError):
        state.notice = ("error", "Не найден модуль проекта или зависимость. Поместите app.py рядом с вашей папкой src и установите requirements-web.txt. Подробности — в README_RU.md.")
    except RuntimeError:
        state.notice = ("error", "Не удалось загрузить языковую модель. Проверьте spaCy/Stanza и скачайте модель выбранного языка по инструкции README_RU.md.")
    except Exception as exc:
        state.notice = ("error", f"Ошибка обработки: {str(exc)}. Если превышен лимит API Groq, подождите 1 минуту.")
    finally:
        gc.collect()


def markup(value: str) -> None:
    st.html(value)


def render_sidebar() -> None:
    with st.sidebar:
        markup('<div class="side-brand">READABLE<span>.</span></div>'
               '<div class="side-kicker">НАСТРОЙТЕ ПОД СЕБЯ</div>')
        st.selectbox("Язык текста", list(LANG_LABELS), format_func=LANG_LABELS.get,
                     key="language")
        markup('<div class="side-section"><span>01 / ПОРОГ ЧИТАЕМОСТИ</span></div>')
        markup(f'<div class="threshold-value">{st.session_state.threshold}<span>/ 100</span></div>')
        st.slider("Порог X", 0, 100, key="threshold", label_visibility="collapsed",
                  help="Будут выбраны предложения с оценкой меньше или равной X.")
        markup('<div class="scale-labels"><span>0 · сложно</span><span>100 · легко</span></div>')
        st.caption("Упростим предложения с оценкой ≤ X. Чем выше порог, тем больше предложений может измениться.")
        markup('<div class="side-section"><span>02 / ПРИМЕРЫ УРОВНЕЙ</span></div>')
        st.radio("Уровень примера", [0, 20, 40, 60, 80, 100], key="example_level",
                  horizontal=True, label_visibility="collapsed")
        level, lang = st.session_state.example_level, st.session_state.language
        markup(f'<div class="example-card"><div class="example-label">{level} / 100 · {LEVEL_NAMES[level]}</div>'
               f'<p>{html.escape(EXAMPLES[lang][level])}</p></div>')
        st.caption("Одна мысль, разная подача. Уровни условные: это ориентиры, а не измерения вашей модели.")
        with st.expander("Настройки ИИ", expanded=False):
            st.text_input("Groq API key", type="password", key="api_key", placeholder="gsk_…")
            if setting("GROQ_API_KEY"):
                st.caption("Ключ сервера настроен. Поле выше можно оставить пустым.")
            st.markdown("[Получить ключ Groq](https://console.groq.com/keys)")
            st.caption("Анализ выполняется локально. В Groq отправляются только выбранные предложения после нажатия «Упростить».")
        markup('<div class="sidebar-foot">0 — очень сложно<br>100 — очень легко</div>')


def render_summary(analysis, result) -> None:
    if analysis:
        mean = f"{statistics.mean(s.score for s in analysis.sentences):.1f}"
        count = len(analysis.sentences)
        chosen = len(selected_sentences(analysis, st.session_state.threshold))
    else:
        mean, count, chosen = "—", "—", "—"
    changed = len(result.changes) if result else "—"
    markup('<div class="summary-strip">' + "".join(
        f'<div><strong>{value}</strong><span>{label}</span></div>' for value, label in
        [(mean, "Средняя оценка"), (count, "Предложений"), (chosen, "Выбрано по X"), (changed, "Изменено")]
    ) + '</div>')


def main() -> None:
    init_state()
    st.html('<style>' + (ASSETS / "style.css").read_text(encoding="utf-8") + '</style>')
    run_pending_action()
    render_sidebar()
    markup('<header class="masthead"><span>РЕДАКТОР ЧИТАЕМОСТИ</span>'
           '<span>RU / EN / KK</span></header>'
           '<div class="intro"><h1>Сложное — <em>проще.</em></h1>'
           '<p>Ваш текст. Ваш порог. Только нужные изменения.</p></div>')

    state = st.session_state
    analysis = state.analysis
    result = state.result if (state.result and analysis
        and state.result.fingerprint == analysis.fingerprint
        and state.result.threshold == state.threshold) else None
    render_summary(analysis, result)

    action_cols = st.columns([1.2, 1.65, 1, 0.7], gap="small")
    action_cols[0].button("Анализировать", on_click=request_action, args=("analyze",),
                          disabled=not state.source_text.strip(), width="stretch")
    selected_count = len(selected_sentences(analysis, state.threshold)) if analysis else None
    action_cols[1].button(
        f"Упростить выбранные · {selected_count}" if selected_count is not None else "Упростить по порогу",
        type="primary", on_click=request_action, args=("simplify",),
        disabled=not state.source_text.strip(), width="stretch",
    )
    action_cols[2].button("Вставить пример", on_click=load_example, width="stretch")
    action_cols[3].button("Очистить", on_click=clear_text, width="stretch",
                          disabled=not state.source_text)

    if state.notice:
        kind, message = state.notice
        getattr(st, kind)(message)
    if analysis and state.result and not result:
        st.info("Порог изменён. Выделение обновлено; нажмите «Упростить», чтобы собрать результат для нового X.")
    if result:
        if result.skipped:
            st.warning(f"Изменено: {len(result.changes)}. Сохранено без изменений: {len(result.skipped)}. Причины — в деталях ниже.")
        elif result.changes:
            st.success(f"Готово: изменено {len(result.changes)} из {len(analysis.sentences)} предложений.")

    left, right = st.columns(2, gap="medium")
    with left, st.container(key="original_panel"):
        markup('<div class="panel-heading"><div><span class="eyebrow">01 / ORIGINAL</span>'
               '<h2>Исходный текст</h2></div><span class="panel-tag tag-original">ОРИГИНАЛ</span></div>')
        st.radio("Режим исходного текста", ["Редактировать", "Подсветка"],
                 key="view", horizontal=True, label_visibility="collapsed")
        if state.view == "Редактировать":
            st.text_area("Исходный текст", key="source_text", height=390,
                         max_chars=MAX_TEXT_CHARS, label_visibility="collapsed",
                         placeholder="Вставьте текст на русском, английском или казахском…")
        else:
            with st.expander("Изменить исходный текст", expanded=False):
                st.text_area("Исходный текст", key="source_text", height=200,
                             max_chars=MAX_TEXT_CHARS, label_visibility="collapsed")
            if analysis:
                markup(render_text(analysis, state.threshold))
            else:
                markup('<div class="empty-panel"><strong>Найдём сложные места.</strong>'
                       '<p>Введите текст и нажмите «Анализировать».</p></div>')
        markup(f'<div class="panel-footer"><span class="legend hard-legend">Будет упрощено: оценка ≤ {state.threshold}</span>'
               f'<span>{len(state.source_text):,} / 30 000</span></div>')

    with right, st.container(key="edited_panel"):
        markup('<div class="panel-heading"><div><span class="eyebrow">02 / EDITED</span>'
               '<h2>Текст проще</h2></div><span class="panel-tag tag-edited">РЕЗУЛЬТАТ</span></div>')
        if analysis:
            st.caption("Новые формулировки выделены зелёным." if result and result.changes
                       else "Полный текст появится здесь с заменами после упрощения.")
            markup(render_text(analysis, state.threshold, result.changes if result else {}))
        else:
            markup('<div class="empty-panel"><span class="empty-numeral">Aa</span>'
                   '<strong>Смысл останется.<br>Читать станет легче.</strong>'
                   '<p>Здесь будет весь текст: простые предложения и упрощённые сложные.</p></div>')
        if result and analysis:
            edited = assemble_text(analysis, result.changes)
            st.download_button("Скачать весь текст", edited.encode("utf-8"),
                               "readable-text.txt", mime="text/plain; charset=utf-8",
                               on_click="ignore", width="stretch")
            with st.expander("Скопировать весь текст"):
                st.code(edited, language=None, wrap_lines=True)
        markup('<div class="panel-footer"><span class="legend edited-legend">Изменённые предложения</span>'
               '<span>Абзацы сохранены</span></div>')

    if analysis:
        with st.expander("Оценки и изменения по предложениям"):
            selected_ids = {s.id for s in selected_sentences(analysis, state.threshold)}
            rows = []
            for s in analysis.sentences:
                change = result.changes.get(s.id) if result else None
                status = ("Упрощено" if change else (result.skipped.get(s.id, "Без изменений")
                          if result else ("Выбрано" if s.id in selected_ids else "Выше порога")))
                rows.append({"№": s.id + 1, "До": round(s.score, 2),
                             "После": round(change.score, 2) if change else round(s.score, 2),
                             "Выбрано": s.id in selected_ids, "Статус": status,
                             "Исходное предложение": s.text,
                             "Результат": change.text if change else s.text})
            st.dataframe(rows, hide_index=True, width="stretch")
            if result:
                st.caption(f"Запросов к ИИ: {result.api_calls}. Повторно использовано проверенных замен: {result.cached}. "
                           "Если предложение разделено, «После» — минимальная оценка его новых частей.")
    markup('<footer class="page-footer"><strong>READABLE.</strong>'
           '<span>Читаемость — ориентир. Проверьте смысл новых формулировок.</span></footer>')


if __name__ == "__main__":
    main()
