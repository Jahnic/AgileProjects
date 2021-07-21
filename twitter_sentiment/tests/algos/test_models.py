import unittest
import torch
import numpy as np
from src.algos.models import BERTGRUSentiment
from transformers import BertTokenizer, BertModel


class TestBERT(unittest.TestCase):
    def test_BERT_output(self):
        """
        Test that get the number of expected results in an expected range
        """
        bert = BertModel.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BERTGRUSentiment(bert, 64, 1, 2, True, 0.6)
        input_negative = tokenizer("This airline sucks hard!", padding=True, return_tensors="pt")
        input_positive = tokenizer("This airline is great! Best experience of my life.", padding=True, return_tensors="pt")
        negative_tensor = input_negative["input_ids"]
        positive_tensor = input_positive["input_ids"]
        output_negative = model(negative_tensor)
        output_positive = model(positive_tensor)
        print("Positive output:", output_positive)
        print("Negative output:", output_negative)
        rows_neg, cols_neg = output_negative.shape
        rows_pos, cols_pos = output_positive.shape

        self.assertEqual((rows_neg + cols_neg), 2)
        self.assertEqual((rows_pos + cols_pos), 2)
