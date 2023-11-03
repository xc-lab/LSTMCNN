import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(torch.nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.linear1 = torch.nn.Linear(128*5, 128)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.linear2 = torch.nn.Linear(128, 2)  # 2个隐层

    def forward(self, x):
        x = x.view(-1, 1 * 128 * 5)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x





class Mnist(nn.Module):
    def __init__(self, in_channels=1, n_classes=2):
        super(Mnist, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(32*5*8, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x





class LSTMCNN(nn.Module):
    def __init__(self, in_channels=1, n_classes=2):
        super(LSTMCNN, self).__init__()
        self.lstm1 = nn.LSTM(input_size=5, hidden_size=5, num_layers=1, batch_first=True)
        # self.lstm2 = nn.LSTM(input_size=5, hidden_size=5, num_layers=1, batch_first=True)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(32*8*10, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = torch.squeeze(x)
        x1, _ = self.lstm1(x)
        # x2, _ = self.lstm2(x1)
        x = torch.cat((x, x1), dim=2)
        x = torch.unsqueeze(x, 1)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class GRUCNN(nn.Module):
    def __init__(self, in_channels=1, n_classes=2):
        super(GRUCNN, self).__init__()
        self.gru1 = nn.GRU(input_size=5, hidden_size=5, num_layers=1, batch_first=True)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32*8*10, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32, n_classes)
        )
    def forward(self, x):
        x = torch.squeeze(x)
        x1, _ = self.gru1(x)
        x = torch.cat((x, x1), dim=2)
        x = torch.unsqueeze(x, 1)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class RNNCNN(nn.Module):
    def __init__(self, in_channels=1, n_classes=2):
        super(RNNCNN, self).__init__()
        self.rnn1 = nn.RNN(input_size=5, hidden_size=5, num_layers=1, batch_first=True)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32*8*10, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32, n_classes)
        )
    def forward(self, x):
        x = torch.squeeze(x)
        x1, _ = self.rnn1(x)
        x = torch.cat((x, x1), dim=2)
        x = torch.unsqueeze(x, 1)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



