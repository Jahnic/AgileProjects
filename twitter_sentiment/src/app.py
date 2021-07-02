import streamlit as st
import torch
from features.build_features import preprocess, index_to_class
from models.binary_BERT import production_hyper_params, BERTGRUSentiment
from transformers import BertModel
from pathlib import Path

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
st.title('Nagative tweet detector')

# take input and preprocess
tweet_text = st.text_input('Enter a Tweet', 'I do not like this airline!!')
tweet_tensor = preprocess(tweet_text)
pred = model(tweet_tensor)
# sigmoid transform prediction and round to binary -> 0/1
pred = float(torch.round(torch.sigmoid(pred)))

# output
st.write(f'This is a {index_to_class[pred]} tweet.')