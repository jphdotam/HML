import torch.nn as nn
import torch.nn.functional as F


class RNN_LSTM(nn.Module):
    def __init__(self, n_inputs, n_outputs, dropout=0.2, batch_first=True):
        super(RNN_LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=n_inputs, hidden_size=64, num_layers=1, batch_first=batch_first, dropout=dropout)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=batch_first, dropout=dropout)
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=100, num_layers=1, batch_first=batch_first, dropout=dropout)

        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        self.lstm3.flatten_parameters()

        self.classifier = nn.Linear(100, n_outputs)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, hidden = self.lstm1(x)
        x, hidden = self.lstm2(x)
        x, hidden = self.lstm3(x)
        x = self.classifier(x[:,-1])  # Get last time step
        return x


class RNN_GRU(nn.Module):
    def __init__(self, n_inputs, n_outputs, dropout=0.2, batch_first=True):
        super(RNN_GRU, self).__init__()
        self.gru1 = nn.GRU(input_size=n_inputs, hidden_size=64, num_layers=1, batch_first=batch_first, dropout=dropout)
        self.gru2 = nn.GRU(input_size=64, hidden_size=128, num_layers=1, batch_first=batch_first, dropout=dropout)
        self.gru3 = nn.GRU(input_size=128, hidden_size=100, num_layers=1, batch_first=batch_first, dropout=dropout)

        self.classifier = nn.Linear(100, n_outputs)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, hidden = self.gru1(x)
        x, hidden = self.gru2(x)
        x, hidden = self.gru3(x)
        x = self.classifier(x[:,-1])  # Get last time step
        return x


if __name__ == "__main__":
    import torch

    def get_n_params(model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    net = RNN_GRU(12, 3)
    x = torch.zeros(2, 12, 500)  # forward() within the network will reorder this appropriately
    y = net(x)
    print(get_n_params(net))

