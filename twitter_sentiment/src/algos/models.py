import torch
import torch.nn as nn
from torch import tensor


class BERTGRUSentiment(nn.Module):
    """
    Transformers based BERT model
    """

    def __init__(self, bert, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout) -> None:
        """
        Args:
            bert: BERT model
            hidden_dim: Hidden size for BERT
            output_dim: Number of classes to predict
            n_layers: Number of layers for BERT
            bidirectional: Whether model should be bidirectional
            dropout: Dropout probability
        """
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

    def forward(self, text: tensor) -> tensor:
        """
        Forward pass for the BERT model
        Args:
            text: Tensor of shape batch size x maximum sentence length
        Returns:
            Tensor of class size which contains probabilities
        """

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
