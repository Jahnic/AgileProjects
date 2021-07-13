import pandas as pd
from src.functions.prepare_data.main import create_train_dev_test
import unittest


class TestCreateTrainDevTest(unittest.TestCase):
    def test_create_train_dev_test(self):
        """
        Test simple dataframe split
        """
        df = pd.DataFrame([['positive', 'love this airline'],
                          ['positive', 'I love this airline'],
                          ['positive', 'I love this airline!'],
                          ['positive', 'I loved this airline'],
                          ['negative', 'I hate this airline'],
                          ['negative', 'Really hate this airline'],
                          ['negative', 'dont like this airline'],
                          ['neutral', 'this airline'],
                          ['neutral', 'okay airline'],
                          ['neutral', 'airline']], columns=['airline_sentiment', 'text'])
        self.assertEqual(df.shape, (10, 2))
        train_df, dev_df, test_df = create_train_dev_test(df, 42)
        self.assertEqual(train_df.shape, (8, 2))
        self.assertEqual(dev_df.shape, (1, 2))
        self.assertEqual(test_df.shape, (1, 2))

    def test_create_train_dev_test_dups(self):
        """
        Test dataframe split with duplicates
        """
        df = pd.DataFrame([['positive', 'love this airline'],
                          ['positive', 'I love this airline'],
                          ['positive', 'I love this airline!'],
                          ['positive', 'I loved this airline'],
                          ['negative', 'I hate this airline'],
                          ['negative', 'Really hate this airline'],
                          ['negative', 'dont like this airline'],
                          ['neutral', 'this airline'],
                          ['neutral', 'okay airline'],
                          ['neutral', 'airline'],
                           ['positive', 'love this airline'],
                           ['positive', 'I love this airline'],
                           ['positive', 'I love this airline!'],
                           ['positive', 'I loved this airline'],
                           ['negative', 'I hate this airline'],
                           ['negative', 'Really hate this airline'],
                           ['negative', 'dont like this airline'],
                           ['neutral', 'this airline'],
                           ['neutral', 'okay airline'],
                           ['neutral', 'airline']], columns=['airline_sentiment', 'text'])
        self.assertEqual(df.shape, (20, 2))
        train_df, dev_df, test_df = create_train_dev_test(df, 42)
        self.assertEqual(train_df.shape, (8, 2))
        self.assertEqual(dev_df.shape, (1, 2))
        self.assertEqual(test_df.shape, (1, 2))


# Run only if test_main.py called directly not when imported
if __name__ == '__main__':
    unittest.main()

