import torch
from torch import nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class lstm_model(nn.Module):

    def __init__(self, vecob_size, num_layers=1, embedding_dim=300, hidden_dim=512, bidirectional=False, dropout=0.5):

        super(lstm_model, self).__init__()
        # [101246] => [300]         [200,118]=>[200,118,300]
        self.embedding = nn.Embedding(vecob_size, embedding_dim=embedding_dim)
        # [300] => [512]
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        # [512] => [1]
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h0=0, c0=0):

        embedding = self.dropout(self.embedding(x))

        if h0==0&c0==0:
            _,(hn, _) = self.rnn(embedding)
        else:
            _, (hn, _) = self.rnn(embedding, (h0, c0))

        hidden = self.dropout(hn[-1])

        out = self.fc(hidden)

        out = self.sigmoid(out)

        return out


class rnn_model(nn.Module):

    def __init__(self, vecob_size, num_layers=1, embedding_dim=300, hidden_size=512, bidirectional=False, dropout=0.25):

        super(rnn_model, self).__init__()

        self.embedding = nn.Embedding(vecob_size, embedding_dim=embedding_dim)

        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(0.3)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h0=0, c0=0):

        embedding = self.dropout(self.embedding(x))

        _,hn = self.rnn(embedding)

        hidden = self.dropout(hn[-1])

        out = self.fc(hidden)

        out = self.sigmoid(out)

        return out