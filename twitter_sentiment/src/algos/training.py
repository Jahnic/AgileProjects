import pickle

from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.optim as optim

from src.algos.models import BERTGRUSentiment
from transformers import BertTokenizer, BertModel
from torchtext.legacy.data import Field, LabelField, TabularDataset, BucketIterator
from sklearn.metrics import f1_score

import numpy as np
import pandas as pd
import json
import warnings
from typing import Tuple, List
from loguru import logger
from google.cloud import secretmanager
from google.cloud.storage import Client, Bucket, Blob
from google.oauth2 import service_account

# Filtering warnings because Pytorch deprecating functions,
# but not clear on what to use instead yet
warnings.filterwarnings("ignore")

# gcloud authentication
with open('../../data/raw/thinking-window-iam.json') as source:
    info = json.load(source)
gcloud_credentials = service_account.Credentials.from_service_account_info(info)
# Global Variables
PROJECT_ID = "thinking-window-318910"
SECRET_VERSION = "latest"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")
HYPER_BUCKET_PATH = f"{PROJECT_ID}_hyperparams"
SPLIT_DATA_PATH = f"{PROJECT_ID}_split_data"
VAL_BUCKET_PATH = f"{PROJECT_ID}_best_val"
VOCAB_BUCKET_PATH = f"{PROJECT_ID}_vocab"
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
GSC_MODEL_PATH = f"gs://{VAL_BUCKET_PATH}/BERT_model.pt"
RELATIVE_MODEL_PATH = "binary_transformer_model.pt"


def get_and_download_data() -> None:
    """
    Retrieves the train, dev, and test sets from cloud storage and saves the files locally.
    Return: None
    """
    train_df = pd.read_csv(f"gs://{SPLIT_DATA_PATH}/train_bin_data.csv")
    dev_df = pd.read_csv(f"gs://{SPLIT_DATA_PATH}/dev_bin_data.csv")
    test_df = pd.read_csv(f"gs://{SPLIT_DATA_PATH}/test_bin_data.csv")

    train_df.to_csv("train.csv", index=False, header=False)
    dev_df.to_csv("dev.csv", index=False, header=False)
    test_df.to_csv("test.csv", index=False, header=False)


def load_secret(secrets: secretmanager.SecretManagerServiceClient(), secret_key: str,
                version: str = SECRET_VERSION) -> str:
    """
    Args:
        secrets: The secret manager from Google Cloud
        secret_key: The name of the secret you wish to retrieve
        version: The version of the secret
    Returns: The secret value
    """
    name = f"projects/126517798921/secrets/{secret_key}/versions/{version}"
    payload = secrets.access_secret_version(request={"name": name}).payload.data.decode("UTF-8")
    return payload


def tokenize_and_cut(sentence: str,
                     max_input_length: int = 100) -> List[str]:
    """
    Tokenizes and pads a sentence up until its maximum length
    """

    tokens = TOKENIZER.tokenize(sentence, padding=True, max_length=max_input_length,
                                truncation=True, return_tensors="pt")
    return tokens


def prepare_binary_data(hyper_params: dict) \
        -> Tuple[BucketIterator, BucketIterator, BucketIterator, Field]:
    """
    ...
    Creates the train, validation, and test iterators for PyTorch as well as the vocabulary.
    Assumes you have train_bin_data.csv, dev_bin_data.csv, and test_bin_data.csv in the same directory
    as the code. Operates on the raw test data, so does all tokenization, sorting, length handling, and batching.
    Assumes CSV files have 2 columns with no headers. The first column is the binary label [0,1] and
    the second column is the text of the tweet.

    Args:
        hyper_params: The dictionary of hyperparameters. Needs to at least contain:
        "BATCH_SIZE" and "MIN_FREQ" as keys

    Returns: Train BucketIterator, Validation BucketIterator, Test BucketIterator, Text Field
    """
    # initialize special tokens
    init_token_idx = TOKENIZER.cls_token_id
    eos_token_idx = TOKENIZER.sep_token_id
    pad_token_idx = TOKENIZER.pad_token_id
    unk_token_idx = TOKENIZER.unk_token_id
    logger.info(f"Init token, eos token, pad token, unk token: "
                f"{init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx}")

    text_field = Field(batch_first=True,
                       use_vocab=False,
                       tokenize=tokenize_and_cut,
                       preprocessing=TOKENIZER.convert_tokens_to_ids,
                       init_token=init_token_idx,
                       eos_token=eos_token_idx,
                       pad_token=pad_token_idx,
                       unk_token=unk_token_idx)

    label_field = LabelField(dtype=torch.float)

    # Prepare batched data
    # first column -> label
    # second column -> text
    fields = [('label', label_field), ('text', text_field)]
    train_data, valid_data, test_data = TabularDataset.splits(path=".", train='train.csv',
                                                              validation='dev.csv',
                                                              test='test.csv', format='CSV', fields=fields)

    # Build vocabulary from training
    label_field.build_vocab(train_data, min_freq=hyper_params["MIN_FREQ"])

    # Make batches sorted by text length
    train_iterator = BucketIterator(train_data, batch_size=hyper_params['BATCH_SIZE'], sort_key=lambda x: len(x.text),
                                    device=DEVICE, sort=True, sort_within_batch=True)
    valid_iterator = BucketIterator(valid_data, batch_size=hyper_params['BATCH_SIZE'], sort_key=lambda x: len(x.text),
                                    device=DEVICE, sort=True, sort_within_batch=True)
    test_iterator = BucketIterator(test_data, batch_size=hyper_params['BATCH_SIZE'], sort_key=lambda x: len(x.text),
                                   device=DEVICE, sort=True, sort_within_batch=True)

    return train_iterator, valid_iterator, test_iterator, text_field


def train(bert: BertModel, hyper_params: dict, train_iter: BucketIterator,
          valid_iter: BucketIterator, experiment: Experiment) -> float:
    """
    Trains the model pushing experimentation to Comet ML and saving the best model
    to Google Cloud Storage if it beats current best model on record (by validation F1)

    Args:
        bert: A pretrained BERT model
        hyper_params: Dictionary that must at least contain the following keys:
        "HIDDEN_DIM", "OUTPUT_DIM", "N_LAYERS", "BIDIRECTIONAL", "NUM_EPOCHS", "DROPOUT"
        train_iter: The training data iterator
        valid_iter: The validation data iterator
        experiment: The Comet ML experiment for tracking

    Returns: The best validation F1 score from training

    """
    # instantiate model
    model = BERTGRUSentiment(bert,
                             hyper_params["HIDDEN_DIM"],
                             hyper_params["OUTPUT_DIM"],
                             hyper_params["N_LAYERS"],
                             hyper_params["BIDIRECTIONAL"],
                             hyper_params["DROPOUT"])

    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(DEVICE)
    criterion = criterion.to(DEVICE)

    # ----------------- Train Model ----------------- #
    logger.info("Starting training...")

    with experiment.train():
        step = 0
        best_val_f1 = 0
        epoch_loss = 0
        n_epochs = hyper_params["NUM_EPOCHS"]
        for epoch in range(1, (n_epochs + 1), 1):
            model.train()
            all_preds = np.array([])
            all_y = np.array([])
            for batch in train_iter:
                optimizer.zero_grad()
                predictions = model(batch.text).squeeze(1)
                loss = criterion(predictions, batch.label)
                optimizer.zero_grad()  # might be unnecessary
                loss.backward()
                optimizer.step()

                # sigmoid transform predictions and round
                rounded_preds = torch.round(torch.sigmoid(predictions))
                # convert predictions and labels to numpy
                rounded_preds = rounded_preds.cpu().detach().numpy()
                y = batch.label.cpu().detach().numpy()

                # append predictions and labels
                all_preds = np.append(all_preds, rounded_preds)
                all_y = np.append(all_y, y)

                # update metrics
                epoch_loss += loss.item()
                batch_f1 = f1_score(rounded_preds, y, average="weighted")
                step += 1
                experiment.log_metric("batch_f1", batch_f1, step=step)

            epoch_f1 = f1_score(all_y, all_preds, average="weighted")
            experiment.log_metric("epoch_f1", epoch_f1, step=epoch)

            with experiment.validate():
                model.eval()
                epoch_loss = 0
                all_preds = np.array([])
                all_y = np.array([])

                # with torch.no_grad():
                for batch in valid_iter:
                    # batch predictions and labels
                    predictions = model(batch.text).squeeze(1)
                    labels = batch.label
                    loss = criterion(predictions, batch.label)

                    # sigmoid transform predictions and round
                    rounded_preds = torch.round(torch.sigmoid(predictions))
                    # convert predictions and labels to numpy
                    rounded_preds = rounded_preds.cpu().detach().numpy()
                    y = labels.cpu().detach().numpy()

                    # append predictions and labels
                    all_preds = np.append(all_preds, rounded_preds)
                    all_y = np.append(all_y, y)

                    # update metrics
                    epoch_loss += loss.item()
                    epoch_f1 += f1_score(rounded_preds, y)

                val_epoch_f1 = f1_score(all_y, all_preds, average="weighted")
                experiment.log_metric("val_epoch_f1", val_epoch_f1, step=epoch)
                experiment.log_confusion_matrix(y_true=all_y, y_predicted=all_preds,
                                                labels=["Negative", "Positive"],
                                                title=f"Confusion Matrix Validation Epoch {epoch}",
                                                file_name=f"confusion-matrix-{epoch}.json")
                if val_epoch_f1 > best_val_f1:
                    best_val_f1 = val_epoch_f1
                    torch.save(model.state_dict(), "binary_transformer_model.pt")

    return best_val_f1


def push_current_results(best_val_f1: float, val_blob: Blob, vocab_bucket: Bucket,
                         text_field: Field) -> None:
    """
   Pushes the current model results to the cloud
    Args:
        best_val_f1: The best validation F1 score from the trained model
        val_blob: Validation blob to store validation score
        vocab_bucket: Bucket to store vocab
        text_field: Text field used by model
    Returns: None
    """
    val_dict = {"best_val_f1": best_val_f1}
    val_blob.reload()
    val_blob.upload_from_string(json.dumps(val_dict))
    upload_model()
    vocab_blob = vocab_bucket.blob("vocab.pkl")
    vocab_blob.upload_from_string(pickle.dumps(text_field.vocab))


def push_results_to_cloud(val_bucket: Bucket, vocab_bucket: Bucket, text_field: Field, best_val_f1: float) -> None:
    """
    Push the appropriate data to the cloud based on the results.
    If no validation results, assume these results are the best and push the model, vocab, and val F1
    If find existing val F1, check if better.
    If not better, don't push anything.
    If better, push model, vocab, and new best val F1
    Args:
        val_bucket: The bucket for validation results
        vocab_bucket: The bucket for vocab
        text_field: The text_field created when preparing the data
        best_val_f1: The best validation F1 from the model trained
    Returns: None
    """
    val_blob = val_bucket.blob("best_val.json")
    # If find an existing validation score on cloud, check if beat it
    if val_blob.exists():
        current_best_val_f1 = json.loads(val_blob.download_as_bytes())['best_val_f1']
        logger.info(f"Best F1 Validation on Record: {current_best_val_f1}")
        if best_val_f1 > current_best_val_f1:
            logger.info("Beat best F1 Validation on record, uploading new model")
            push_current_results(best_val_f1=best_val_f1, vocab_bucket=vocab_bucket,
                                 text_field=text_field, val_blob=val_blob
                                 )
        else:
            logger.info("Didn't beat best F1 Validation on record, not uploading model")
    # If don't find existing score, push these results
    else:
        logger.info("No validation results on record, uploading model")
        push_current_results(best_val_f1=best_val_f1, vocab_bucket=vocab_bucket,
                             text_field=text_field, val_blob=val_blob
                             )


def convert_json_string_values(json_dict):
    for name, param in json_dict.items():
        if param != ('True' or 'False'):
            try:
                json_dict[name] = int(param)
            except ValueError:
                json_dict[name] = float(param)
        else:
            json_dict[name] = bool(param)
    return json_dict


def upload_model() -> None:
    """
    Uploads saved model to Google Cloud Storage (VAL_BUCKET_PATH)
    """
    model_name = RELATIVE_MODEL_PATH
    # Initialize upload
    client = Client(project=PROJECT_ID)
    bucket = client.get_bucket(VAL_BUCKET_PATH)
    blob = bucket.blob(model_name)
    logger.info(f'Start "{model_name}" upload to {bucket}')
    # Change chunk size to prevent request timeout
    blob._chunk_size = 5 * 1024 * 1024  # 5 MB
    blob.upload_from_filename(model_name)


def main() -> None:
    """
    Gets all the necessary data, prepares it, trains the model, and pushes the appropriate results to cloud
    Returns: None
    """
    # Setup Google Cloud access and Comet ML
    secrets = secretmanager.SecretManagerServiceClient()
    client = Client(project=PROJECT_ID, credentials=gcloud_credentials)
    hyper_bucket = client.get_bucket(HYPER_BUCKET_PATH)
    vocab_bucket = client.get_bucket(VOCAB_BUCKET_PATH)
    val_bucket = client.get_bucket(VAL_BUCKET_PATH)
    comet_api = load_secret(secrets, "comet_api_key", version="2").strip()
    experiment = Experiment(api_key=comet_api,
                            project_name="twitter_sentiment",
                            workspace="jahnic", )

    # Load and log hyperparameters to Comet
    blob = hyper_bucket.get_blob('hyper_params.json')
    hyper_params = json.loads(blob.download_as_string())
    hyper_params = convert_json_string_values(hyper_params)
    experiment.log_parameters(hyper_params)

    # Train the model
    get_and_download_data()
    train_iter, valid_iter, _, text_field = prepare_binary_data(hyper_params)
    bert = BertModel.from_pretrained('bert-base-uncased')
    best_val_f1 = train(bert, hyper_params, train_iter, valid_iter, experiment)
    logger.info(f"Best F1 Validation: {best_val_f1}")
    push_results_to_cloud(val_bucket=val_bucket, vocab_bucket=vocab_bucket,
                          text_field=text_field, best_val_f1=best_val_f1)


if __name__ == '__main__':
    main()
