import streamlit as st
import torch
from features.build_features import preprocess, index_to_class
from models.binary_BERT import production_hyper_params, BERTGRUSentiment
from data.get_data import get_recent_tweets
import pandas as pd
import math
from transformers import BertModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert = BertModel.from_pretrained('bert-base-uncased')
model = BERTGRUSentiment(bert, production_hyper_params['hidden_dim'],
                         production_hyper_params['output_dim'],
                         production_hyper_params['n_layers'],
                         production_hyper_params['bidirectional'],
                         production_hyper_params['dropout'])
model.load_state_dict(torch.load("../models/binary_transformer_model.pt"))
model.eval()

# Title of page
st.title('Negative tweet detector')

twitter_account = st.text_input("Enter a Twitter Username", "SpiritAirlines")
recent_tweets = get_recent_tweets(twitter_account, n=1)
prediction_classes = []
prediction_negative_probability = []
for tweet_text, _ in recent_tweets:
    tweet_tensor = preprocess(tweet_text)
    prediction = model(tweet_tensor)
    # sigmoid transform prediction and round to binary -> 0/1
    prediction_rounded = float(torch.round(torch.sigmoid(prediction)))
    # probability of negative sentiment
    neg_probability = -1 * float(torch.sigmoid(prediction))
    print(prediction_rounded)
    print(neg_probability)
    print("-" * 25)
    prediction_negative_probability.append(math.exp(neg_probability) * 100)
    prediction_class = index_to_class[prediction_rounded]
    prediction_classes.append(prediction_class)

df = pd.DataFrame()
df['Tweet'] = [x[0] for x in recent_tweets]
df['Link'] = [x[1] for x in recent_tweets]
df['Sentiment'] = prediction_classes
df['Negative Probability'] = prediction_negative_probability
df = df.sort_values('Negative Probability', ascending=False)

st.table(df)
