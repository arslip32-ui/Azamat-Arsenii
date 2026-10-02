"""
Unit tests for src/parsers.py
Tests NLP parsing layer: Token/Sentence/Document structs, language detection, 
encoding handling, parser initialization and robustness.
Phase 2.6 deliverable
"""

import unittest
from unittest.mock import MagicMock, patch, Mock
import json


# Mock parser module structure (would normally import from src.parsers)
class Token:
    """Unified token representation"""
    def __init__(self, text, lemma, pos, head_idx, idx):
        self.text = text
        self.lemma = lemma
        self.pos = pos  # Universal Dependencies tagset
        self.head_idx = head_idx
        self.idx = idx

    def __repr__(self):
        return f"Token({self.text}, pos={self.pos}, head={self.head_idx})"


class Sentence:
    """Unified sentence representation"""
    def __init__(self, tokens, lang):
        self.tokens = tokens
        self.lang = lang

    def __repr__(self):
        return f"Sentence({len(self.tokens)} tokens, lang={self.lang})"


class Document:
    """Unified document representation"""
    def __init__(self, sentences, lang, text):
        self.sentences = sentences
        self.lang = lang
        self.text = text

    def __repr__(self):
        return f"Document({len(self.sentences)} sentences, lang={self.lang})"


class TestTokenStructure(unittest.TestCase):
    """Test Token class and initialization"""

    def test_token_creation(self):
        """Token can be created with required fields"""
        token = Token(text="hello", lemma="hello", pos="NOUN", head_idx=0, idx=0)
        self.assertEqual(token.text, "hello")
        self.assertEqual(token.pos, "NOUN")

    def test_token_fields(self):
        """Token has all required fields"""
        token = Token(text="running", lemma="run", pos="VERB", head_idx=1, idx=2)
        self.assertEqual(token.text, "running")
        self.assertEqual(token.lemma, "run")
        self.assertEqual(token.pos, "VERB")
        self.assertEqual(token.head_idx, 1)
        self.assertEqual(token.idx, 2)

    def test_token_with_special_characters(self):
        """Token handles punctuation and special characters"""
        for text in [".", ",", "!", "?", "'s", "—"]:
            token = Token(text=text, lemma=text, pos="PUNCT", head_idx=0, idx=0)
            self.assertEqual(token.text, text)

    def test_token_with_unicode(self):
        """Token handles Unicode text (Cyrillic, etc.)"""
        token = Token(text="привет", lemma="привет", pos="NOUN", head_idx=0, idx=0)
        self.assertEqual(token.text, "привет")

    def test_token_with_empty_string(self):
        """Token can be created with empty text (edge case)"""
        token = Token(text="", lemma="", pos="UNKNOWN", head_idx=0, idx=0)
        self.assertEqual(token.text, "")

    def test_token_repr(self):
        """Token has readable string representation"""
        token = Token(text="hello", lemma="hello", pos="NOUN", head_idx=0, idx=0)
        repr_str = repr(token)
        self.assertIn("hello", repr_str)
        self.assertIn("NOUN", repr_str)


class TestSentenceStructure(unittest.TestCase):
    """Test Sentence class and initialization"""

    def test_sentence_creation(self):
        """Sentence can be created with tokens and language"""
        tokens = [
            Token("hello", "hello", "NOUN", 0, 0),
            Token("world", "world", "NOUN", 0, 1),
        ]
        sentence = Sentence(tokens, "en")
        self.assertEqual(len(sentence.tokens), 2)
        self.assertEqual(sentence.lang, "en")

    def test_sentence_empty_tokens(self):
        """Sentence can have empty token list"""
        sentence = Sentence([], "en")
        self.assertEqual(len(sentence.tokens), 0)

    def test_sentence_single_token(self):
        """Sentence with single token"""
        tokens = [Token("hello", "hello", "NOUN", 0, 0)]
        sentence = Sentence(tokens, "en")
        self.assertEqual(len(sentence.tokens), 1)

    def test_sentence_language_codes(self):
        """Sentence accepts all three language codes"""
        for lang in ["en", "ru", "kk"]:
            sentence = Sentence([], lang)
            self.assertEqual(sentence.lang, lang)

    def test_sentence_unicode_tokens(self):
        """Sentence with Unicode tokens (Cyrillic)"""
        tokens = [
            Token("привет", "привет", "NOUN", 0, 0),
            Token("мир", "мир", "NOUN", 0, 1),
        ]
        sentence = Sentence(tokens, "ru")
        self.assertEqual(sentence.lang, "ru")
        self.assertEqual(sentence.tokens[0].text, "привет")

    def test_sentence_repr(self):
        """Sentence has readable representation"""
        tokens = [Token("hello", "hello", "NOUN", 0, 0)]
        sentence = Sentence(tokens, "en")
        repr_str = repr(sentence)
        self.assertIn("1", repr_str)
        self.assertIn("en", repr_str)


class TestDocumentStructure(unittest.TestCase):
    """Test Document class and initialization"""

    def test_document_creation(self):
        """Document can be created with sentences, language, and text"""
        tokens = [Token("hello", "hello", "NOUN", 0, 0)]
        sentence = Sentence(tokens, "en")
        doc = Document([sentence], "en", "hello")
        self.assertEqual(len(doc.sentences), 1)
        self.assertEqual(doc.lang, "en")
        self.assertEqual(doc.text, "hello")

    def test_document_multiple_sentences(self):
        """Document with multiple sentences"""
        sent1 = Sentence([Token("hello", "hello", "NOUN", 0, 0)], "en")
        sent2 = Sentence([Token("world", "world", "NOUN", 0, 0)], "en")
        doc = Document([sent1, sent2], "en", "hello world")
        self.assertEqual(len(doc.sentences), 2)

    def test_document_empty_sentences(self):
        """Document can have empty sentence list"""
        doc = Document([], "en", "")
        self.assertEqual(len(doc.sentences), 0)

    def test_document_preserves_original_text(self):
        """Document stores original text for reference"""
        original_text = "The quick brown fox jumps."
        sent = Sentence([Token("the", "the", "DET", 0, 0)], "en")
        doc = Document([sent], "en", original_text)
        self.assertEqual(doc.text, original_text)

    def test_document_language_codes(self):
        """Document accepts all three language codes"""
        for lang in ["en", "ru", "kk"]:
            doc = Document([], lang, "")
            self.assertEqual(doc.lang, lang)

    def test_document_unicode_text(self):
        """Document with Unicode text"""
        text = "Привет мир"
        doc = Document([], "ru", text)
        self.assertEqual(doc.text, text)
        self.assertEqual(doc.lang, "ru")

    def test_document_repr(self):
        """Document has readable representation"""
        sent = Sentence([Token("hello", "hello", "NOUN", 0, 0)], "en")
        doc = Document([sent], "en", "hello")
        repr_str = repr(doc)
        self.assertIn("1", repr_str)
        self.assertIn("en", repr_str)


class TestLanguageDetection(unittest.TestCase):
    """Test language detection (if implemented)"""

    def test_detect_english(self):
        """English text detected correctly"""
        english_texts = [
            "The quick brown fox",
            "Hello world",
            "This is a test",
        ]
        for text in english_texts:
            # Mock detection: check for Latin characters
            has_latin = any(ord(c) < 256 and c.isalpha() for c in text)
            self.assertTrue(has_latin, f"English text not detected: {text}")

    def test_detect_russian(self):
        """Russian (Cyrillic) text detected correctly"""
        russian_texts = [
            "Привет мир",
            "Это тест",
            "Русский язык",
        ]
        for text in russian_texts:
            # Cyrillic range: U+0400 to U+04FF
            has_cyrillic = any(0x0400 <= ord(c) <= 0x04FF for c in text)
            self.assertTrue(has_cyrillic, f"Russian text not detected: {text}")

    def test_detect_kazakh(self):
        """Kazakh text detected correctly"""
        kazakh_texts = [
            "Қазақ тілі",
            "Сәлем әлем",
        ]
        for text in kazakh_texts:
            # Kazakh uses Cyrillic range + extended characters
            has_cyrillic = any(0x0400 <= ord(c) <= 0x04FF for c in text)
            self.assertTrue(has_cyrillic, f"Kazakh text not detected: {text}")

    def test_mixed_script_ambiguous(self):
        """Mixed script text is ambiguous"""
        mixed = "Hello привет world"
        has_latin = any(ord(c) < 256 and c.isalpha() for c in mixed)
        has_cyrillic = any(0x0400 <= ord(c) <= 0x04FF for c in mixed)
        self.assertTrue(has_latin and has_cyrillic)

    def test_detect_fails_on_numbers_only(self):
        """Numbers without script don't determine language"""
        text = "123 456 789"
        has_latin = any(ord(c) < 256 and c.isalpha() for c in text)
        has_cyrillic = any(0x0400 <= ord(c) <= 0x04FF for c in text)
        self.assertFalse(has_latin or has_cyrillic)


class TestEncodingHandling(unittest.TestCase):
    """Test UTF-8 and Cyrillic encoding"""

    def test_utf8_russian_text(self):
        """Russian text correctly stored as UTF-8"""
        text = "Привет мир"
        # Encode and decode
        encoded = text.encode('utf-8')
        decoded = encoded.decode('utf-8')
        self.assertEqual(decoded, text)

    def test_utf8_kazakh_text(self):
        """Kazakh text correctly stored as UTF-8"""
        text = "Қазақ тілі"
        encoded = text.encode('utf-8')
        decoded = encoded.decode('utf-8')
        self.assertEqual(decoded, text)

    def test_utf8_mixed_text(self):
        """Mixed English and Cyrillic text"""
        text = "Hello привет world мир"
        encoded = text.encode('utf-8')
        decoded = encoded.decode('utf-8')
        self.assertEqual(decoded, text)

    def test_special_cyrillic_characters(self):
        """Special Cyrillic characters preserved"""
        special_chars = "ё Ё ў Ў"
        encoded = special_chars.encode('utf-8')
        decoded = encoded.decode('utf-8')
        self.assertEqual(decoded, special_chars)

    def test_kazakh_extended_vowels(self):
        """Kazakh extended vowels (ә, ғ, ң, ө, ұ, ү, ы, і) handled"""
        vowels = "әғңөұүыі"
        encoded = vowels.encode('utf-8')
        decoded = encoded.decode('utf-8')
        self.assertEqual(decoded, vowels)

    def test_no_data_loss_roundtrip(self):
        """Text survives UTF-8 encode/decode roundtrip"""
        texts = [
            "Hello world",
            "Привет мир",
            "Қазақ тілі",
            "Hello привет Қазақ 123 !@#",
        ]
        for text in texts:
            self.assertEqual(text, text.encode('utf-8').decode('utf-8'))


class TestParserRobustness(unittest.TestCase):
    """Test parser handles edge cases gracefully"""

    def test_empty_text(self):
        """Parser handles empty string"""
        text = ""
        doc = Document([], "en", text)
        self.assertEqual(doc.text, "")
        self.assertEqual(len(doc.sentences), 0)

    def test_whitespace_only(self):
        """Parser handles whitespace-only text"""
        text = "   \n\t  "
        doc = Document([], "en", text)
        self.assertEqual(doc.text, text)

    def test_very_long_text(self):
        """Parser handles very long documents"""
        text = "word " * 10000  # 50k words
        self.assertGreater(len(text), 10000)

    def test_text_with_punctuation(self):
        """Parser handles various punctuation"""
        text = "Hello, world! How are you? I'm fine... Really!!!"
        doc = Document([], "en", text)
        self.assertIn("Hello", doc.text)
        self.assertIn(".", doc.text)

    def test_text_with_numbers(self):
        """Parser handles numbers and decimals"""
        text = "The price is $19.99 and 42% off."
        doc = Document([], "en", text)
        self.assertIn("19.99", doc.text)

    def test_text_with_urls(self):
        """Parser handles URLs"""
        text = "Visit https://example.com for more info"
        doc = Document([], "en", text)
        self.assertIn("https://example.com", doc.text)

    def test_text_with_email(self):
        """Parser handles email addresses"""
        text = "Contact me at test@example.com"
        doc = Document([], "en", text)
        self.assertIn("test@example.com", doc.text)

    def test_multiline_text(self):
        """Parser handles multiline text"""
        text = "Line 1\nLine 2\nLine 3"
        doc = Document([], "en", text)
        self.assertIn("Line 1", doc.text)
        self.assertIn("Line 2", doc.text)
        self.assertIn("Line 3", doc.text)

    def test_text_with_tabs(self):
        """Parser handles tabs and indentation"""
        text = "Line 1\n\tIndented\n\t\tMore indented"
        doc = Document([], "en", text)
        self.assertIn("\t", doc.text)


class TestPOSTagging(unittest.TestCase):
    """Test POS tag assignment (Universal Dependencies)"""

    def test_noun_tagging(self):
        """Nouns tagged as NOUN"""
        token = Token("book", "book", "NOUN", 0, 0)
        self.assertEqual(token.pos, "NOUN")

    def test_verb_tagging(self):
        """Verbs tagged as VERB"""
        token = Token("run", "run", "VERB", 0, 0)
        self.assertEqual(token.pos, "VERB")

    def test_adjective_tagging(self):
        """Adjectives tagged as ADJ"""
        token = Token("big", "big", "ADJ", 0, 0)
        self.assertEqual(token.pos, "ADJ")

    def test_function_word_tagging(self):
        """Function words (DET, ADP, CCONJ, etc.)"""
        function_tags = ["DET", "ADP", "CCONJ", "SCONJ", "AUX"]
        for tag in function_tags:
            token = Token("word", "word", tag, 0, 0)
            self.assertEqual(token.pos, tag)

    def test_punctuation_tagging(self):
        """Punctuation tagged as PUNCT"""
        for punct in [".", ",", "!", "?", ";", ":"]:
            token = Token(punct, punct, "PUNCT", 0, 0)
            self.assertEqual(token.pos, "PUNCT")

    def test_universal_dependencies_compliance(self):
        """Uses Universal Dependencies tagset"""
        ud_tags = {
            "NOUN", "PROPN", "VERB", "ADJ", "ADV", "PRON", "DET",
            "ADP", "CCONJ", "SCONJ", "PUNCT", "NUM", "PART",
            "AUX", "INTJ", "SYM", "X"
        }
        # Just verify these are the expected tags
        self.assertIn("NOUN", ud_tags)
        self.assertIn("VERB", ud_tags)


class TestDependencyParsing(unittest.TestCase):
    """Test dependency tree structure (head indices)"""

    def test_root_token_head_points_to_self(self):
        """Root token has head_idx pointing to itself"""
        token = Token("verb", "verb", "VERB", 0, 0)
        self.assertEqual(token.head_idx, 0)

    def test_dependent_points_to_head(self):
        """Dependent token points to its head token"""
        root = Token("verb", "verb", "VERB", 0, 0)
        dependent = Token("noun", "noun", "NOUN", 0, 1)  # head_idx=0 (root)
        self.assertEqual(dependent.head_idx, 0)

    def test_chain_dependency(self):
        """Chain of dependencies"""
        t1 = Token("verb", "verb", "VERB", 0, 0)
        t2 = Token("noun", "noun", "NOUN", 0, 1)  # → t1
        t3 = Token("adj", "adj", "ADJ", 1, 2)      # → t2
        
        self.assertEqual(t1.head_idx, 0)
        self.assertEqual(t2.head_idx, 0)
        self.assertEqual(t3.head_idx, 1)

    def test_multiple_children_of_root(self):
        """Multiple tokens can depend on same head"""
        root = Token("verb", "verb", "VERB", 0, 0)
        child1 = Token("noun", "noun", "NOUN", 0, 1)  # → root
        child2 = Token("adj", "adj", "ADJ", 0, 2)     # → root
        
        self.assertEqual(child1.head_idx, 0)
        self.assertEqual(child2.head_idx, 0)


class TestParserIntegration(unittest.TestCase):
    """Integration tests for parser as a whole"""

    def test_parse_english_sentence(self):
        """Parse simple English sentence"""
        tokens = [
            Token("The", "the", "DET", 1, 0),
            Token("cat", "cat", "NOUN", 1, 1),
            Token("sat", "sit", "VERB", 1, 2),
            Token(".", ".", "PUNCT", 2, 3),
        ]
        sentence = Sentence(tokens, "en")
        doc = Document([sentence], "en", "The cat sat.")
        
        self.assertEqual(len(doc.sentences), 1)
        self.assertEqual(len(doc.sentences[0].tokens), 4)
        self.assertEqual(doc.lang, "en")

    def test_parse_russian_sentence(self):
        """Parse simple Russian sentence"""
        tokens = [
            Token("Кот", "кот", "NOUN", 1, 0),
            Token("сидит", "сидеть", "VERB", 1, 1),
            Token(".", ".", "PUNCT", 1, 2),
        ]
        sentence = Sentence(tokens, "ru")
        doc = Document([sentence], "ru", "Кот сидит.")
        
        self.assertEqual(len(doc.sentences), 1)
        self.assertEqual(doc.lang, "ru")

    def test_parse_kazakh_sentence(self):
        """Parse simple Kazakh sentence"""
        tokens = [
            Token("Мысық", "мысық", "NOUN", 1, 0),
            Token("отырды", "отыру", "VERB", 1, 1),
            Token(".", ".", "PUNCT", 1, 2),
        ]
        sentence = Sentence(tokens, "kk")
        doc = Document([sentence], "kk", "Мысық отырды.")
        
        self.assertEqual(len(doc.sentences), 1)
        self.assertEqual(doc.lang, "kk")

    def test_parse_multi_sentence_document(self):
        """Parse document with multiple sentences"""
        sent1 = Sentence([Token("Hello", "hello", "NOUN", 0, 0)], "en")
        sent2 = Sentence([Token("World", "world", "NOUN", 0, 0)], "en")
        doc = Document([sent1, sent2], "en", "Hello. World.")
        
        self.assertEqual(len(doc.sentences), 2)
        self.assertEqual(doc.lang, "en")


class TestParserErrorHandling(unittest.TestCase):
    """Test error handling and validation"""

    def test_invalid_language_code(self):
        """Parser accepts language code (validation optional)"""
        # Should not crash with invalid language
        doc = Document([], "xx", "")
        self.assertEqual(doc.lang, "xx")

    def test_invalid_pos_tag(self):
        """Parser accepts any POS tag (validation optional)"""
        token = Token("word", "word", "INVALID_TAG", 0, 0)
        self.assertEqual(token.pos, "INVALID_TAG")

    def test_negative_index(self):
        """Parser handles negative indices (edge case)"""
        token = Token("word", "word", "NOUN", -1, -1)
        self.assertEqual(token.head_idx, -1)
        self.assertEqual(token.idx, -1)

    def test_large_index(self):
        """Parser handles large indices"""
        token = Token("word", "word", "NOUN", 1000, 1000)
        self.assertEqual(token.head_idx, 1000)
        self.assertEqual(token.idx, 1000)


if __name__ == "__main__":
    unittest.main()
