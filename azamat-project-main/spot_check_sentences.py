from src.parsers import ParserManager
from src.features import extract_features
from src.normalize import calculate_readability_score, aggregate_features

parser = ParserManager()

# Test sentences for each language
test_cases = {
    'en': {
        'easy': "The cat sat.",
        'medium': "The quick brown fox jumps over the lazy dog.",
        'hard': "Notwithstanding multifaceted considerations pertaining to infrastructural paradigms, the proposition remains decidedly untenable."
    },
    'ru': {
        'easy': "Кот сидит.",
        'medium': "Быстрая коричневая лиса прыгает через ленивую собаку.",
        'hard': "Несмотря на многоаспектные соображения, касающиеся инфраструктурных парадигм, предложение остается решительно неприемлемым."
    },
    'kk': {
        'easy': "Мысық отырды.",
        'medium': "Тез қоңыр түлкі ленивое итті өтіп секіреді.",
        'hard': "Өндіктік аспектілі ойларға қарамастан инфрақұрылымдық парадигмаларға қатысты, ұсыныс ұрықсыз."
    }
}

for lang, sentences in test_cases.items():
    print(f"\n{lang.upper()}:")
    print("-" * 50)
    
    for difficulty, text in sentences.items():
        try:
            doc = parser.parse(text, lang=lang)
            features_list = extract_features(doc, lang=lang)
            aggregated = aggregate_features(features_list)
            score = calculate_readability_score(aggregated, lang=lang)
            
            print(f"{difficulty:8} | Score: {score:6.1f} | Text: {text[:50]}...")
        except Exception as e:
            print(f"{difficulty:8} | ERROR: {e}")