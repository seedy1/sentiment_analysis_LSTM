import torch


class SentimentModel(torch.nn.Module):

    def __init__(self, vocab_size, output_size, embed_dim, hidden_dim, n_layers, dropout_amount=0.4):
        # super().__init__(SentimentModel, self)
        super().__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embed layer
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        # lstm layer
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=dropout_amount, batch_first=True)
        # self.lstm2 = torch.nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=dropout_amount, batch_first=True)
        self.dropout = torch.nn.Dropout(0.4)

        # linear and sigmoid layers
        self.fc1 = torch.nn.Linear(hidden_dim, 64)
        self.fc2 = torch.nn.Linear(64, 16)
        self.fc3 = torch.nn.Linear(16, output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size()
        embed = self.embedding(x)
        lstm_out, hidden = self.lstm(embed, hidden)

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully connected layers
        out = self.dropout(lstm_out)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        sig_out = self.sigmoid(out)

        # flatten
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]

        return sig_out, hidden

    def init_hidden(self, batch_size):

        weight = next( self.parameters()).data
        hidden = (
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),#.cuda(),
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()#.cuda()
        )

        return hidden

