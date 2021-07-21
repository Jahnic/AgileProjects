import torch
import torch.nn as nn
import torch.optim as optim

from transformers import BertTokenizer, BertModel
from torchtext.legacy.data import Field, LabelField, TabularDataset, BucketIterator
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import json
import time
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
source_folder = "../data/interim/"

hyper_params = {
    "tokenize": "spacy",
    "lower": True,
    "batch_size": 64,
    "hidden_size": 64,
    "embedding_size": 32,
    "num_classes": 3,
    "num_layers": 1,
    "learning_rate": 0.001,
    "min_freq": 5,
    "bidirectional": True,
    "num_epochs": 75,
    "disabled": False,
    "dropout_p": 0.6,
    "save_model_path": "../models/pytorch_model.pt",
    "save_vocab_path": "../models/pytorch_vocab.pkl"
}

# define tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# max sentance lenth
max_input_length = 100

def tokenize_and_cut(sentence):
    '''Tokenizes a sentances and removes two characters from maximum length (512)'''
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    return tokens

init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id

print(init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx)

TEXT = Field(batch_first = True,
                  use_vocab = False,
                  tokenize = tokenize_and_cut,
                  preprocessing = tokenizer.convert_tokens_to_ids,
                  init_token = init_token_idx,
                  eos_token = eos_token_idx,
                  pad_token = pad_token_idx,
                  unk_token = unk_token_idx)

LABEL = LabelField(dtype = torch.float)

# Prepare batched data

# first column -> label
# second column -> text
fields = [('label', LABEL), ('text', TEXT)]

train_data, valid_data, test_data = TabularDataset.splits(path=source_folder, train='train_bin_data.csv', validation='dev_bin_data.csv',
                                           test='test_bin_data.csv', format='CSV', fields=fields)
# sort by length of text -> batch of same size sentences
train_iterator = BucketIterator(train_data, batch_size=hyper_params['batch_size'], sort_key=lambda x: len(x.text),
                            device=device, sort=True, sort_within_batch=True)
valid_iterator = BucketIterator(valid_data, batch_size=hyper_params['batch_size'], sort_key=lambda x: len(x.text),
                            device=device, sort=True, sort_within_batch=True)
test_iterator = BucketIterator(test_data, batch_size=hyper_params['batch_size'], sort_key=lambda x: len(x.text),
                            device=device, sort=True, sort_within_batch=True)

# Build vocabulary from training
LABEL.build_vocab(train_data)
# initialize pretrained model
bert = BertModel.from_pretrained('bert-base-uncased')


class BERTGRUSentiment(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):

        super().__init__()

        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, sent len]

        with torch.no_grad():
            embedded = self.bert(text)[0]
            # embedded = [batch size, sent len, emb dim]
        _, hidden = self.rnn(embedded)
        # hidden = [n layers * n directions, batch size, emb dim]

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])
            # hidden = [batch size, hid dim]

        output = self.out(hidden)
        # output = [batch size, out dim]

        return output

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc


# instantiate model
HIDDEN_DIM = 64
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.6

model = BERTGRUSentiment(bert,
                         HIDDEN_DIM,
                         OUTPUT_DIM,
                         N_LAYERS,
                         BIDIRECTIONAL,
                         DROPOUT)

# ----------------- Train Model ----------------- #

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)


def train(model, iterator, optimizer, criterion):
    # metrics
    epoch_loss = 0
    epoch_acc = 0
    test_f1 = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)

        # sigmoid transform predictions and round
        rounded_preds = torch.round(torch.sigmoid(predictions))

        # convert predictions and labels to numpy
        rounded_preds = rounded_preds.cpu().detach().numpy()
        y = batch.label.cpu().detach().numpy()

        loss.backward()
        optimizer.step()

        # update metrics
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        test_f1 += f1_score(rounded_preds, y)

    return epoch_loss / len(iterator), epoch_acc / len(iterator), test_f1 / len(iterator)


def evaluate(model, iterator, criterion):
    # metrics
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0

    # predictions and lables
    all_preds = np.array([])
    all_y = np.array([])

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            # batch predictions and labels
            predictions = model(batch.text).squeeze(1)
            labels = batch.label

            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, labels)

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
            epoch_acc += acc.item()
            epoch_f1 += f1_score(rounded_preds, y)

    # final report
    print(classification_report(all_y, all_preds))

    # confusion matrix
    cm = confusion_matrix(all_y, all_preds, labels=[1, 0])
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['NEGATIVE', 'POSITIVE'])
    ax.yaxis.set_ticklabels(['NEGATIVE', 'POSITIVE'])

    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_f1 / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 5

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc, train_f1 = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc, valid_f1 = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), '../models/binary_transformer_model.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | Train f1: {train_f1:.2f}')
    print(f'\t Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc * 100:.2f}%  | Val. f1: {valid_f1:.2f}')


