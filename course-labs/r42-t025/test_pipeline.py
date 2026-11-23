import unittest
from pipeline import TextTokenizer, UNK_ID

class TestTextPreprocessing(unittest.TestCase):
    def setUp(self):
        self.train_texts = ["red apple", "green apple", "blue sky", "dark sky"]
        self.tokenizer = TextTokenizer()
        self.tokenizer.fit(self.train_texts)

    def test_vocab_learned(self):
        self.assertIn("apple", self.tokenizer.vocab)
        self.assertIn("sky", self.tokenizer.vocab)
        self.assertEqual(self.tokenizer.vocab["<UNK>"], 0)

    def test_unknown_token_maps_to_unk(self):
        encoded = self.tokenizer.encode("yellow banana")
        self.assertEqual(encoded, [UNK_ID, UNK_ID])

    def test_mixed_known_unknown(self):
        encoded = self.tokenizer.encode("red banana")
        self.assertEqual(encoded[0], self.tokenizer.vocab["red"])
        self.assertEqual(encoded[1], UNK_ID)

    def test_deliberate_failure(self):
        encoded = self.tokenizer.encode("apple")
        with self.assertRaises(AssertionError):
            self.assertEqual(encoded[0], 99999)

if __name__ == "__main__":
    unittest.main()
