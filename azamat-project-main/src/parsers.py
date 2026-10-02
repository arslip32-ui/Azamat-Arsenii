"""
src/parsers.py
Unified NLP parser wrapper for trilingual readability assessment.
Supports English (spaCy), Russian (Stanza), Kazakh (Stanza).
All three languages return the same Token/Sentence/Document structure.
"""

import logging
from typing import List, Optional, Dict

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Unified Data Structures
# ─────────────────────────────────────────────

class Token:
    """
    Unified token representation across all three languages.
    head_idx: index of the head token in the sentence (-1 if root)
    """
    def __init__(
        self,
        text: str,
        lemma: str,
        upos: str,
        head_idx: int,
        dep_rel: str,
        idx: int
    ):
        self.text = text
        self.lemma = lemma
        self.upos = upos        # Universal POS tag (NOUN, VERB, ADJ, etc.)
        self.head_idx = head_idx  # Index of head token; -1 = root
        self.dep_rel = dep_rel  # Dependency relation (nsubj, obj, etc.)
        self.idx = idx          # Position in sentence (0-based)

    def __repr__(self):
        return (
            f"Token(text={self.text!r}, upos={self.upos}, "
            f"head={self.head_idx}, dep={self.dep_rel})"
        )


class Sentence:
    """Unified sentence representation: a list of Tokens."""
    def __init__(self, text: str, tokens: List[Token]):
        self.text = text
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens)

    def __repr__(self):
        preview = self.text[:60] + ("..." if len(self.text) > 60 else "")
        return f"Sentence({preview!r}, tokens={len(self.tokens)})"


class Document:
    """Unified document representation: a list of Sentences."""
    def __init__(self, text: str, sentences: List[Sentence]):
        self.text = text
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __repr__(self):
        return f"Document(sentences={len(self.sentences)})"


# ─────────────────────────────────────────────
# Parser Manager
# ─────────────────────────────────────────────

class ParserManager:
    """
    Manages NLP parsers for English, Russian, and Kazakh.

    Uses lazy loading: parsers are only loaded when first called,
    not all at initialization time.

    Usage:
        pm = ParserManager()
        doc = pm.parse("Hello world.", "en")
        for sent in doc.sentences:
            for token in sent.tokens:
                print(token)
    """

    SUPPORTED_LANGUAGES = {"en", "ru", "kk"}
    LANG_NAMES = {"en": "English", "ru": "Russian", "kk": "Kazakh"}

    def __init__(self, languages: Optional[List[str]] = None):
        """
        Args:
            languages: List of language codes to support.
                       Defaults to ['en', 'ru', 'kk'].
        """
        self.languages = set(languages or ["en", "ru", "kk"])
        unsupported = self.languages - self.SUPPORTED_LANGUAGES
        if unsupported:
            raise ValueError(f"Unsupported languages: {unsupported}")

        # Lazy cache: populated on first parse() call per language
        self._parsers: Dict[str, object] = {}

    # ── Public API ────────────────────────────

    def parse(self, text: str, lang: str) -> Document:
        """
        Parse text in the given language.

        Args:
            text: Input text (one or more sentences).
            lang: Language code ('en', 'ru', or 'kk').

        Returns:
            Document with unified Token/Sentence structure.

        Raises:
            ValueError: Unsupported language code.
            RuntimeError: Parser failed to load.
        """
        if lang not in self.languages:
            raise ValueError(
                f"Language '{lang}' not in supported set {self.languages}. "
                f"Choose from: {self.SUPPORTED_LANGUAGES}"
            )

        parser = self._load_parser(lang)
        if parser is None:
            raise RuntimeError(
                f"{self.LANG_NAMES[lang]} parser is not available. "
                f"Check installation and model downloads."
            )

        if lang == "en":
            return self._parse_spacy(text, parser)
        else:
            return self._parse_stanza(text, parser)

    def available_languages(self) -> List[str]:
        """Return list of languages whose parsers loaded successfully."""
        available = []
        for lang in self.languages:
            parser = self._load_parser(lang)
            if parser is not None:
                available.append(lang)
        return available

    # ── Lazy Loader ───────────────────────────

    def _load_parser(self, lang: str):
        """Load parser on first use; cache and return it."""
        if lang in self._parsers:
            return self._parsers[lang]

        logger.info(f"Loading {self.LANG_NAMES[lang]} parser...")

        if lang == "en":
            parser = self._load_spacy()
        else:
            parser = self._load_stanza(lang)

        self._parsers[lang] = parser
        if parser:
            logger.info(f"✓ {self.LANG_NAMES[lang]} parser ready.")
        return parser

    def _load_spacy(self):
        try:
            import spacy
            return spacy.load("en_core_web_sm")
        except OSError:
            logger.warning(
                "spaCy English model not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
            return None
        except ImportError:
            logger.warning("spaCy not installed. Run: pip install spacy")
            return None

    def _load_stanza(self, lang: str):
        try:
            import stanza
            processors = "tokenize,pos,lemma,depparse"
            if lang == "ru":
                processors = "tokenize,pos,lemma,depparse"
            return stanza.Pipeline(
                lang,
                processors=processors,
                verbose=False
            )
        except Exception as e:
            logger.warning(
                f"Stanza {self.LANG_NAMES[lang]} parser failed: {e}\n"
                f"Try: import stanza; stanza.download('{lang}')"
            )
            return None

    # ── Parsers ───────────────────────────────

    def _parse_spacy(self, text: str, nlp) -> Document:
        """Convert spaCy output to unified Document."""
        doc = nlp(text)
        sentences = []

        for spacy_sent in doc.sents:
            offset = spacy_sent[0].i
            tokens = []
            for token in spacy_sent:
                # Root token points to itself in spaCy; map to -1
                head_idx = token.head.i - offset if token.head.i != token.i else -1
                tokens.append(Token(
                    text=token.text,
                    lemma=token.lemma_,
                    upos=token.pos_,
                    head_idx=head_idx,
                    dep_rel=token.dep_,
                    idx=token.i - offset
                ))
            sentences.append(Sentence(spacy_sent.text, tokens))

        return Document(text, sentences)

    def _parse_stanza(self, text: str, nlp) -> Document:
        """Convert Stanza output to unified Document."""
        stanza_doc = nlp(text)
        sentences = []

        for stanza_sent in stanza_doc.sentences:
            tokens = []
            for word in stanza_sent.words:
                # Stanza head is 1-based; 0 = root → map to -1
                head_idx = word.head - 1 if word.head > 0 else -1
                tokens.append(Token(
                    text=word.text,
                    lemma=word.lemma or word.text,
                    upos=word.upos or "X",
                    head_idx=head_idx,
                    dep_rel=word.deprel or "dep",
                    idx=word.id - 1
                ))
            sentences.append(Sentence(stanza_sent.text, tokens))

        return Document(text, sentences)

    def __repr__(self):
        loaded = [
            self.LANG_NAMES[k]
            for k in self._parsers
            if self._parsers[k] is not None
        ]
        return f"ParserManager(loaded={loaded})"


# ─────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    TEST_SENTENCES = {
        "en": "The quick brown fox jumps over the lazy dog.",
        "ru": "Быстрая коричневая лиса прыгает через ленивую собаку.",
        "kk": "Жылдам қоңыр түлкі жалқау иттің үстінен секіреді.",
    }

    pm = ParserManager()

    print("\n" + "=" * 60)
    print("SMOKE TEST: Trilingual Parser")
    print("=" * 60)

    for lang, text in TEST_SENTENCES.items():
        print(f"\n[{lang.upper()}] {text}")
        try:
            doc = pm.parse(text, lang)
            sent = doc.sentences[0]
            print(f"  Tokens ({len(sent)}):")
            for tok in sent.tokens:
                head_text = (
                    sent.tokens[tok.head_idx].text
                    if tok.head_idx >= 0 else "ROOT"
                )
                print(
                    f"    [{tok.idx}] {tok.text:<15} "
                    f"upos={tok.upos:<6} "
                    f"head→{head_text} ({tok.dep_rel})"
                )
        except RuntimeError as e:
            print(f"  ✗ FAILED: {e}")

    print("\n" + "=" * 60)
    print(f"Parser status: {pm}")
    print("=" * 60)
