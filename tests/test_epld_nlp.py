import unittest
from epld_nlp import PashtoNormalizer, MorphologyAwarePashtoTokenizer, CrossLingualSentimentPipeline

class TestEPLD(unittest.TestCase):
    def test_normalizer(self):
        norm = PashtoNormalizer().normalize("دا كتاب گټور دی")
        self.assertIn("ک", norm)
        self.assertIn("ګ", norm)

    def test_tokenizer(self):
        analysis = MorphologyAwarePashtoTokenizer().analyze_token("کتابتونونه")
        self.assertEqual(analysis.stem, "کتابتون")
        self.assertEqual(analysis.suffix, "ونه")

    def test_sentiment(self):
        pipe = CrossLingualSentimentPipeline()
        self.assertEqual(pipe.predict_label("دا کتاب د زده کوونکو لپاره ډېر ګټور او معلوماتي دی."), "positive")
        self.assertEqual(pipe.predict_label("نن د انټرنیټ اړیکه ډېره ورو او ناهیلې کوونکې وه."), "negative")

if __name__ == "__main__":
    unittest.main()
