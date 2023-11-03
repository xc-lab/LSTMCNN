import torch
import torch.nn as nn
from torchstat import stat
import time
from ptflops import get_model_complexity_info



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
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(3,1), stride=(2,1), padding=(1,0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
            # nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16*32*5, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class RNNCNN(nn.Module):
    def __init__(self, in_channels=1, n_classes=2):
        super(RNNCNN, self).__init__()
        self.rnn1 = nn.RNN(input_size=5, hidden_size=5, num_layers=1, batch_first=True)
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
        x1, _ = self.rnn1(x)
        # x2, _ = self.lstm2(x1)
        x = torch.cat((x, x1), dim=1)
        x = torch.unsqueeze(x, 0)
        x = torch.unsqueeze(x, 0)
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
            nn.Linear(32*8*5, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = torch.squeeze(x)
        x1, _ = self.lstm1(x)
        # x2, _ = self.lstm2(x1)
        # x = torch.cat((x, x1, x2), dim=1)
        x = torch.unsqueeze(x1, 0)
        x = torch.unsqueeze(x, 0)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class GRUCNN(nn.Module):
    def __init__(self, in_channels=1, n_classes=2):
        super(GRUCNN, self).__init__()
        self.lstm1 = nn.GRU(input_size=5, hidden_size=5, num_layers=1, batch_first=True)
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
        x = torch.cat((x, x1), dim=1)
        x = torch.unsqueeze(x, 0)
        x = torch.unsqueeze(x, 0)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def measure_inference_speed(model, data, max_iter=200, log_interval=50):
    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    fps = 0

    # benchmark with 2000 image and take the average
    for i in range(max_iter):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(*data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f'Done image [{i + 1:<3}/ {max_iter}], '
                    f'fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    flush=True)

        if (i + 1) == max_iter:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(
                f'Overall fps: {fps:.1f} img / s, '
                f'times per image: {1000 / fps:.1f} ms / img',
                flush=True)
            break
    return fps

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=(5,1), stride=(2,1), padding=(2,0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
            nn.Conv2d(48, 128, kernel_size=(5,1), stride=(2,1), padding=(2,0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
            nn.Conv2d(128, 192, kernel_size=(3,1), stride=(1,1), padding=(1,0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=(3,1), stride=(1,1), padding=(1,0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(192*2*4, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = AlexNet()




# 方法1
with torch.cuda.device(0):
  flops, params = get_model_complexity_info(model, (1, 128, 2), as_strings=True, print_per_layer_stat=True) #不用写batch_size大小，默认batch_size=1
  print('Flops:  ' + flops)
  print('Params: ' + params)


# #方法1：torchstat
# print('Method 2')
# stat(model, (1, 128, 5))
#
#
# # 另一个方法
# net = model.cuda()
# data = torch.randn((1, 1, 128, 5)).cuda()
# measure_inference_speed(net, (data,))

