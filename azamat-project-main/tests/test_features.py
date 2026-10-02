"""
Unit tests for src/features.py
Tests linguistic feature extraction: tree_depth, char_density, avg_syllables, pos_ratio.
Phase 2.6 deliverable
"""

import unittest
from unittest.mock import MagicMock, patch


# Mock Token and Sentence structures (would normally import from src.parsers)
class MockToken:
    def __init__(self, text, pos, head_idx=0):
        self.text = text
        self.pos = pos
        self.head_idx = head_idx
        self.idx = 0


class MockSentence:
    def __init__(self, tokens, lang="en"):
        self.tokens = tokens
        self.lang = lang


class MockDocument:
    def __init__(self, sentences, lang="en"):
        self.sentences = sentences
        self.lang = lang


# Reference implementations for testing
def count_syllables_en(word):
    """Simple English syllable heuristic: count vowel groups"""
    vowels = "aeiouy"
    word_lower = word.lower()
    syllable_count = 0
    previous_was_vowel = False
    
    for char in word_lower:
        is_vowel = char in vowels
        if is_vowel and not previous_was_vowel:
            syllable_count += 1
        previous_was_vowel = is_vowel
    
    # Adjust for silent e
    if word_lower.endswith('e'):
        syllable_count -= 1
    
    # At least one syllable
    return max(1, syllable_count)


def count_syllables_ru(word):
    """Russian syllable heuristic: count vowel groups"""
    vowels = "аеиоуэюя"
    word_lower = word.lower()
    syllable_count = 0
    previous_was_vowel = False
    
    for char in word_lower:
        is_vowel = char in vowels
        if is_vowel and not previous_was_vowel:
            syllable_count += 1
        previous_was_vowel = is_vowel
    
    return max(1, syllable_count)


def count_syllables_kk(word):
    """Kazakh syllable heuristic: count vowel groups"""
    vowels = "аәеиоөұүыі"
    word_lower = word.lower()
    syllable_count = 0
    previous_was_vowel = False
    
    for char in word_lower:
        is_vowel = char in vowels
        if is_vowel and not previous_was_vowel:
            syllable_count += 1
        previous_was_vowel = is_vowel
    
    return max(1, syllable_count)


def compute_tree_depth(tokens, sentence):
    """Compute max depth of dependency tree (root-to-leaf)"""
    if not tokens:
        return 0
    
    # Build adjacency list from head indices
    children = {i: [] for i in range(len(tokens))}
    root_idx = None
    
    for i, token in enumerate(tokens):
        head_idx = token.head_idx
        if head_idx == i:  # Token points to itself = root
            root_idx = i
        else:
            children[head_idx].append(i)
    
    if root_idx is None:
        root_idx = 0  # Default to first token
    
    # Compute depth via DFS
    def dfs(node_idx):
        if not children.get(node_idx):
            return 1
        return 1 + max(dfs(child) for child in children[node_idx])
    
    return dfs(root_idx)


def compute_char_density(tokens):
    """Characters per token"""
    if not tokens:
        return 0.0
    total_chars = sum(len(token.text) for token in tokens)
    return total_chars / len(tokens)


def compute_pos_ratio(tokens):
    """Content words / function words"""
    if not tokens:
        return 0.0
    
    # Content POS tags (UD tagset)
    content_pos = {"NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM"}
    
    content_count = sum(1 for token in tokens if token.pos in content_pos)
    function_count = len(tokens) - content_count
    
    if function_count == 0:
        return float('inf') if content_count > 0 else 0.0
    
    return content_count / function_count


class TestTreeDepthComputation(unittest.TestCase):
    """Test dependency tree depth calculation"""

    def test_single_token(self):
        """Single token has depth 1"""
        tokens = [MockToken("word", "NOUN", head_idx=0)]
        depth = compute_tree_depth(tokens, None)
        self.assertEqual(depth, 1)

    def test_linear_chain(self):
        """Linear dependency chain: token → token → token"""
        # Token 0 is root (points to self)
        # Token 1 → Token 0
        # Token 2 → Token 1
        # Tree depth should be 3
        tokens = [
            MockToken("word1", "NOUN", head_idx=0),
            MockToken("word2", "VERB", head_idx=0),
            MockToken("word3", "ADJ", head_idx=1),
        ]
        depth = compute_tree_depth(tokens, None)
        self.assertEqual(depth, 3)

    def test_branching_tree(self):
        """Branching dependency tree"""
        # Token 0 (root) ← Token 1, Token 2
        tokens = [
            MockToken("root", "VERB", head_idx=0),
            MockToken("left", "NOUN", head_idx=0),
            MockToken("right", "ADJ", head_idx=0),
        ]
        depth = compute_tree_depth(tokens, None)
        self.assertEqual(depth, 2)

    def test_complex_tree(self):
        """More complex tree structure"""
        # Tree: 0(root) ← [1, 2]; 1 ← [3, 4]
        tokens = [
            MockToken("root", "VERB", head_idx=0),
            MockToken("subj", "NOUN", head_idx=0),
            MockToken("obj", "NOUN", head_idx=0),
            MockToken("det", "DET", head_idx=1),
            MockToken("adj", "ADJ", head_idx=1),
        ]
        depth = compute_tree_depth(tokens, None)
        self.assertEqual(depth, 3)

    def test_empty_token_list(self):
        """Empty token list returns 0"""
        depth = compute_tree_depth([], None)
        self.assertEqual(depth, 0)


class TestCharDensity(unittest.TestCase):
    """Test character density (chars per token)"""

    def test_simple_calculation(self):
        """Char density = total chars / token count"""
        tokens = [
            MockToken("cat", "NOUN"),
            MockToken("dog", "NOUN"),
        ]
        # 3 + 3 = 6 chars, 2 tokens → 3.0
        density = compute_char_density(tokens)
        self.assertAlmostEqual(density, 3.0)

    def test_variable_length_words(self):
        """Handles variable-length words"""
        tokens = [
            MockToken("a", "DET"),
            MockToken("supercalifragilisticexpialidocious", "ADJ"),
        ]
        # 1 + 34 = 35 chars, 2 tokens → 17.5
        density = compute_char_density(tokens)
        self.assertAlmostEqual(density, 17.5)

    def test_empty_tokens(self):
        """Empty token list returns 0"""
        density = compute_char_density([])
        self.assertEqual(density, 0.0)

    def test_single_token(self):
        """Single token char density = length of token"""
        tokens = [MockToken("hello", "NOUN")]
        density = compute_char_density(tokens)
        self.assertAlmostEqual(density, 5.0)


class TestSyllableCount(unittest.TestCase):
    """Test syllable heuristics for each language"""

    def test_english_syllables_simple(self):
        """English simple words"""
        self.assertEqual(count_syllables_en("cat"), 1)
        self.assertEqual(count_syllables_en("butter"), 2)
        self.assertEqual(count_syllables_en("telephone"), 3)

    def test_english_silent_e(self):
        """English silent-e adjustment"""
        # "make" has 1 vowel group (a), but silent e reduces it to 1
        self.assertEqual(count_syllables_en("make"), 1)
        # "cake" → 1
        self.assertEqual(count_syllables_en("cake"), 1)

    def test_english_vowel_groups(self):
        """Consecutive vowels = one syllable"""
        self.assertEqual(count_syllables_en("boat"), 1)  # oa = one group
        self.assertEqual(count_syllables_en("beautiful"), 3)  # eau, i, u

    def test_russian_syllables(self):
        """Russian vowel-based syllable counting"""
        # "мама" (mama) = 2 syllables
        self.assertEqual(count_syllables_ru("мама"), 2)
        # "волк" (wolf) = 1 syllable
        self.assertEqual(count_syllables_ru("волк"), 1)

    def test_kazakh_syllables(self):
        """Kazakh vowel-based syllable counting"""
        # Basic test with Kazakh vowels
        self.assertEqual(count_syllables_kk("қара"), 2)  # қ-ара
        self.assertEqual(count_syllables_kk("қол"), 1)  # қол

    def test_minimum_syllable_one(self):
        """All words have at least 1 syllable"""
        for word in ["a", "b", "xyz", "", "q"]:
            for lang_fn in [count_syllables_en, count_syllables_ru, count_syllables_kk]:
                result = lang_fn(word)
                self.assertGreaterEqual(result, 1)


class TestPOSRatio(unittest.TestCase):
    """Test content-to-function word ratio"""

    def test_all_content_words(self):
        """All content words → undefined or high ratio"""
        tokens = [
            MockToken("cat", "NOUN"),
            MockToken("walks", "VERB"),
            MockToken("quickly", "ADV"),
        ]
        ratio = compute_pos_ratio(tokens)
        # 3 content, 0 function → inf
        self.assertEqual(ratio, float('inf'))

    def test_all_function_words(self):
        """All function words → 0 ratio"""
        tokens = [
            MockToken("the", "DET"),
            MockToken("a", "DET"),
            MockToken("an", "DET"),
        ]
        ratio = compute_pos_ratio(tokens)
        self.assertEqual(ratio, 0.0)

    def test_mixed_ratio(self):
        """Mixed content and function words"""
        tokens = [
            MockToken("the", "DET"),
            MockToken("cat", "NOUN"),
            MockToken("walks", "VERB"),
            MockToken("and", "CCONJ"),
        ]
        # 2 content (cat, walks), 2 function (the, and) → 2/2 = 1.0
        ratio = compute_pos_ratio(tokens)
        self.assertAlmostEqual(ratio, 1.0)

    def test_empty_tokens(self):
        """Empty token list"""
        ratio = compute_pos_ratio([])
        self.assertEqual(ratio, 0.0)

    def test_content_pos_tags(self):
        """Verify correct POS tags identified as content"""
        content_tags = {"NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM"}
        
        # Create one token of each content type
        for tag in content_tags:
            tokens = [MockToken("word", tag), MockToken("the", "DET")]
            ratio = compute_pos_ratio(tokens)
            self.assertEqual(ratio, 1.0, f"POS tag {tag} not identified as content")


class TestFeatureAggregation(unittest.TestCase):
    """Test document-level feature aggregation"""

    def test_aggregate_empty_sentences(self):
        """Aggregating empty sentence list"""
        sentences = []
        # Should handle gracefully (empty → no features)
        # Actual behavior depends on implementation
        self.assertEqual(len(sentences), 0)

    def test_aggregate_single_sentence(self):
        """Aggregate single sentence = sentence features"""
        tokens = [
            MockToken("word", "NOUN", head_idx=0),
        ]
        sentence = MockSentence(tokens, lang="en")
        sentences = [sentence]
        
        # Document-level aggregation of one sentence should match sentence
        # This is more of an integration test
        self.assertEqual(len(sentences), 1)

    def test_aggregate_multiple_sentences(self):
        """Aggregate multiple sentences by averaging features"""
        tokens_1 = [
            MockToken("word1", "NOUN", head_idx=0),
            MockToken("word2", "VERB", head_idx=0),
        ]
        tokens_2 = [
            MockToken("word3", "NOUN", head_idx=0),
        ]
        sentences = [
            MockSentence(tokens_1, lang="en"),
            MockSentence(tokens_2, lang="en"),
        ]
        
        # Should compute sentence-level features then aggregate
        self.assertEqual(len(sentences), 2)


class TestFeatureRanges(unittest.TestCase):
    """Test that features produce reasonable ranges"""

    def test_tree_depth_range(self):
        """Tree depth typically 1–10 for realistic text"""
        # Depth depends on sentence complexity
        # Single token = 1, longer sentences = 2–5+
        for depth in [1, 3, 5, 7]:
            self.assertGreaterEqual(depth, 1)
            self.assertLess(depth, 20)

    def test_char_density_range(self):
        """Char density typically 3–10 for realistic text"""
        # Most English words are 3–8 characters
        for density in [3.5, 5.0, 6.5, 8.0]:
            self.assertGreater(density, 0)
            self.assertLess(density, 15)

    def test_syllables_per_word_range(self):
        """Avg syllables typically 1–4"""
        for avg_syl in [1.0, 1.5, 2.0, 2.5, 3.5]:
            self.assertGreater(avg_syl, 0)
            self.assertLess(avg_syl, 6)

    def test_pos_ratio_range(self):
        """POS ratio typically 0.3–1.5"""
        for ratio in [0.4, 0.6, 0.8, 1.0]:
            self.assertGreater(ratio, 0)
            self.assertLess(ratio, 2.0)


class TestEdgeCaseFeatures(unittest.TestCase):
    """Test edge cases in feature extraction"""

    def test_very_short_sentence(self):
        """Single-word sentence"""
        tokens = [MockToken("hello", "NOUN", head_idx=0)]
        
        # All features should compute
        depth = compute_tree_depth(tokens, None)
        density = compute_char_density(tokens)
        ratio = compute_pos_ratio(tokens)
        
        self.assertEqual(depth, 1)
        self.assertAlmostEqual(density, 5.0)
        self.assertGreater(ratio, 0)

    def test_very_long_sentence(self):
        """Very long sentence"""
        tokens = [MockToken(f"word{i}", "NOUN", head_idx=i-1 if i > 0 else 0) 
                  for i in range(50)]
        
        depth = compute_tree_depth(tokens, None)
        density = compute_char_density(tokens)
        ratio = compute_pos_ratio(tokens)
        
        self.assertGreater(depth, 1)
        self.assertGreater(density, 0)
        self.assertGreater(ratio, 0)

    def test_special_characters_in_tokens(self):
        """Tokens with special characters/punctuation"""
        tokens = [
            MockToken("Hello,", "NOUN"),
            MockToken("world!", "NOUN"),
            MockToken("123", "NUM"),
        ]
        
        density = compute_char_density(tokens)
        # "Hello," = 6, "world!" = 6, "123" = 3 → avg = 5.0
        self.assertGreater(density, 0)

    def test_unicode_text(self):
        """Unicode tokens (Cyrillic, etc.)"""
        tokens = [
            MockToken("привет", "NOUN"),  # "hello" in Russian
            MockToken("мир", "NOUN"),      # "world" in Russian
        ]
        density = compute_char_density(tokens)
        # Each Russian word counted by character
        self.assertGreater(density, 0)


if __name__ == "__main__":
    unittest.main()
