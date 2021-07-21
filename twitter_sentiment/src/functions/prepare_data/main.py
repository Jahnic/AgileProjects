from typing import Tuple
from google.cloud import storage
import pandas as pd
from loguru import logger
from pandas import DataFrame
import json
from google.oauth2 import service_account

# gcloud authentication
with open('data/raw/thinking-window-iam.json') as source:
    info = json.load(source)
gcloud_credentials = service_account.Credentials.from_service_account_info(info)
storage_client = storage.Client(credentials=gcloud_credentials)

index_to_class = {'positive': 1, 'neutral': 0, 'negative': 0}
class_to_index = {value: key for key, value in index_to_class.items()}
TRAIN_SIZE = .80
DEV_SIZE = .10
RAW_DATA_BUCKET = 'thinking-window-318910_raw_data'
SPLIT_DATA_BUCKET = 'thinking-window-318910_split_data'
RANDOM_STATE = 42


def create_train_dev_test(raw_dfs: DataFrame, random_state: int) \
        -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Takes in a raw dataframe of tweets, remove duplicates, converts class to integer,
    and splits into train, dev, and test
        Args:
            raw_dfs: Raw dataframe to split
            random_state: The random state to use when shuffling
        Return: Train, Dev, and Test dataframes
    """
    raw_dfs = raw_dfs.drop_duplicates()
    raw_dfs = raw_dfs[['airline_sentiment', 'text']]
    # Convert the written class (positive, neutral,negative) to an integer
    raw_dfs['airline_sentiment'] = raw_dfs['airline_sentiment'].apply(lambda x: index_to_class[x])

    # Sample data to shuffle rows before splitting
    raw_dfs = raw_dfs.sample(frac=1, random_state=random_state)
    data_shape = raw_dfs.shape
    logger.info(f'Final dataframe shape: {data_shape}')

    train_length = round(data_shape[0] * TRAIN_SIZE)
    dev_length = round(data_shape[0] * DEV_SIZE)

    train_df = raw_dfs.iloc[: train_length]
    dev_df = raw_dfs.iloc[train_length: (train_length + dev_length)]
    test_df = raw_dfs.iloc[(train_length + dev_length):]

    return train_df, dev_df, test_df


def split_data(event: dict, context: any) -> None:
    """
    Reads in the raw data from RAW_DATA_BUCKET and then
    splits it into train, dev, and test which are store in SPLIT_DATA_BUCKET
    Args:
        event: The dictionary with data specific to this type of event.
                    the 'data field contains a description of the event in
                    the Cloud Storage 'object' format described here:
                    https://cloud.google.com/storage/docs/json_api/v1/objects#resource
        context: (google.cloud.functions.Context): Metadata of triggering event.

    Returns: None
    """
    raw_bucket = storage_client.get_bucket(RAW_DATA_BUCKET)

    # Combine all files found in RAW_DATA_BUCKET into one data frame
    raw_dfs = []
    for blob in raw_bucket.list_blobs():
        file_name = f'gs://{RAW_DATA_BUCKET}/{blob.name}'
        logger.info(f'Processing file {file_name}')
        raw_dfs.append(pd.read_csv(file_name))

    raw_dfs = pd.concat(raw_dfs)
    train_df, dev_df, test_df = create_train_dev_test(raw_dfs, 42)

    train_df.to_csv(f'gs://{SPLIT_DATA_BUCKET}/train_data.csv', index=False)
    dev_df.to_csv(f'gs://{SPLIT_DATA_BUCKET}/dev_data.csv', index=False)
    test_df.to_csv(f'gs://{SPLIT_DATA_BUCKET}/test_data.csv', index=False)
    logger.info(f'Saved train, dev, test data to {SPLIT_DATA_BUCKET}')
