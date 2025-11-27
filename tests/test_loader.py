import sys
import os
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from src.config import Config
from src.data.loader import get_calibration_data

class TestDataPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Config.set_model("qwen")
        cls.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_ID)

    def test_gsm8k_format(self):
        data = get_calibration_data("gsm8k", n_samples=2)
        
        self.assertEqual(len(data), 2)
        self.assertIsInstance(data[0], str)
        self.assertTrue(len(data[0]) > 10)

    def test_wikitext_format(self):
        data = get_calibration_data("wikitext", n_samples=2)
        
        self.assertEqual(len(data), 2)
        self.assertNotIn("Please reason step by step", data[0])
        self.assertNotIn("<|im_start|>system", data[0])

if __name__ == '__main__':
    unittest.main()

