import torch
import torch.nn as nn
import torch.nn.functional as F



import numpy as np
torch.manual_seed(0);


class Cnn1d(nn.Module):
    def __init__(self, *, nx, nt, cnnSize=32, cp1=(64, 3, 2), cp2=(128, 5, 2)):
        super(Cnn1d, self).__init__()
        self.nx = nx
        self.nt = nt
        cOut, f, p = cp1
        self.conv1 = nn.Conv1d(nx, cOut, f)
        self.pool1 = nn.MaxPool1d(p)
        lTmp = int(calConvSize(nt, f, 0, 1, 1) / p)

        cIn = cOut
        cOut, f, p = cp2
        self.conv2 = nn.Conv1d(cIn, cOut, f)
        self.pool2 = nn.MaxPool1d(p)
        lTmp = int(calConvSize(lTmp, f, 0, 1, 1) / p)

        self.flatLength = int(cOut * lTmp)
        self.fc1 = nn.Linear(self.flatLength, cnnSize)
        self.fc2 = nn.Linear(cnnSize, cnnSize)

    def forward(self, x):
        # x- [nt,ngrid,nx]
        x1 = x
        x1 = x1.permute(1, 2, 0)
        x1 = self.pool1(F.relu(self.conv1(x1)))
        x1 = self.pool2(F.relu(self.conv2(x1)))
        x1 = x1.view(-1, self.flatLength)
        x1 = F.relu(self.fc1(x1))
        x1 = self.fc2(x1)

        return x1


def calConvSize(lin, kernel, stride, padding=0, dilation=1):
    lout = (lin + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
    return int(lout)


def calPoolSize(lin, kernel, stride=None, padding=0, dilation=1):
    if stride is None:
        stride = kernel
    lout = (lin + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
    return int(lout)


def calFinalsize1d(nobs, noutk, ksize, stride, pool):
    nlayer = len(ksize)
    Lout = nobs
    for ii in range(nlayer):
        Lout = calConvSize(lin=Lout, kernel=ksize[ii], stride=stride[ii])
        if pool is not None:
            Lout = calPoolSize(lin=Lout, kernel=pool[ii])
    Ncnnout = int(Lout * noutk)  # total CNN feature number after convolution
    return Ncnnout

# def relu(x):
#      return np.maximum(0, x-20000)
class trycnn(nn.Module):
    def __init__(self, output_dim=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1,12)),
            nn.MaxPool2d(kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=1, padding=0),
        )
        self.regression = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(9344, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(100, output_dim)
        )



    def forward(self, x):
        x1 = self.features(x)
        # x1=nn.ReLU(0,x1-20000)

        x2 = x1.view(x1.shape[0], -1)

        y = self.regression(x2)
        return y

class TryCnn(nn.Module):
    # def __init__(self, output_dim=1):
    #     super().__init__()
    #     self.features = nn.Sequential(
    #         nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,4)),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
    #
    #         nn.Conv2d(in_channels=32, out_channels=128, kernel_size=2),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
    #
    #     )

    def __init__(self, output_dim=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3,4)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2 ),

            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        )

        self.regression = nn.Sequential(
            nn.Dropout(p=0.25),
            # nn.Dropout(p=0.25),
            nn.Linear( 7168, 95),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(95, output_dim)
        )
    def forward(self, x):
        x1 = self.features(x)
        # print(x1.size)
        # x1=nn.ReLU(0,x1-20000)
        # x2=x1.view(100,-1)
        # print(x1.shape[0])
        x2 = x1.view(x1.shape[0], -1)
        y = self.regression(x2)
        return y


class LeNet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)

        return x
