import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import VocabularyEncoder

class TestPreprocessingContract(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(os.path.dirname(__file__), "fixture_data.json")
        with open(fixture_path, "r") as f:
            self.fixture = json.load(f)
        self.train_data = self.fixture[:4]
        self.test_data = self.fixture[4:]

    def test_vocab_fit_train_only(self):
        enc = VocabularyEncoder()
        enc.fit(self.train_data)
        self.assertIn("apple", enc.vocab)
        self.assertIn("sky", enc.vocab)
        self.assertNotIn("fruit", enc.vocab)
        self.assertNotIn("cloud", enc.vocab)

    def test_unseen_token_mapped_to_unk(self):
        enc = VocabularyEncoder()
        enc.fit(self.train_data)
        encoded_test = enc.transform(self.test_data)
        # "red fruit" -> "fruit" is unseen, should be 0 (<unk>)
        # "blue cloud" -> "cloud" is unseen, should be 0 (<unk>)
        self.assertEqual(encoded_test[0][1], 0)
        self.assertEqual(encoded_test[1][1], 0)

    def test_unfit_transform_raises(self):
        enc = VocabularyEncoder()
        with self.assertRaises(RuntimeError):
            enc.transform(self.test_data)

if __name__ == "__main__":
    unittest.main()
