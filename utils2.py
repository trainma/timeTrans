import torch
import numpy as np
from torch.autograd import Variable
import pandas as pd
import argparse
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default='./data/electricity.txt', help='location of the data file')
parser.add_argument('--model', type=str, default='transformer', help='')
parser.add_argument('--window', type=int, default=24 * 7, help='window size')
parser.add_argument('--horizon', type=int, default=3)
parser.add_argument('--d_model', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--dec_layers', type=int, default=1)
parser.add_argument('--position', type=str, default=3)
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=60, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='batch size')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=54321, help='random seed')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N', help='report interval')
parser.add_argument('--save', type=str, default='save/model.pt', help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--amsgrad', type=str, default=True)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--L1Loss', type=bool, default=False)
parser.add_argument('--normalize', type=int, default=2)
args = parser.parse_args()


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, device, args):
        self.device = device
        self.P = args.window
        self.h = args.horizon
        self.data_scale = MinMaxScaler()
        self.label_scale = MinMaxScaler()
        # fin = open(file_name)
        df = pd.read_excel(file_name)
        self.rawdat = np.array(df)
        # self.rawdat = np.loadtxt(fin, delimiter=',')  # (26304,321)
        self.dat = self.data_scale.fit_transform(self.rawdat)
        # self.dat = np.zeros(self.rawdat.shape)  # (26304,321)
        self.label = self.label_scale.fit_transform(self.rawdat[:, 0].reshape(-1, 1))
        self.n, self.m = self.dat.shape  # n=26304 m=321
        # self.scale = np.ones(self.m)
        # self._normalized(args.normalize)
        # 分训练样本 测试样本

        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        # self.scale = torch.as_tensor(self.scale, device=device, dtype=torch.float)

        # tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)
        tmp = self.label_scale.inverse_transform(np.array(self.test[1].clone().cpu()))
        # self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)

       # self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))
        self.rae = np.mean(np.abs(tmp - np.mean(tmp)))
        # fin.close()

    # def _normalized(self, normalize):
    #     # normalized by the maximum value of entire matrix.
    #
    #     if (normalize == 0):
    #         self.dat = self.rawdat
    #
    #     if (normalize == 1):
    #         self.dat = self.rawdat / np.max(self.rawdat)
    #
    #     # normlized by the maximum value of each row(sensor).
    #     if (normalize == 2):
    #         for i in range(self.m):
    #             self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
    #             self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

    def _split(self, train, valid, test):
        # P:window h:horizon

        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _batchify(self, idx_set, horizon):
        # n:row m:colum P:window
        n = len(idx_set)
        # X:(num,windows,features)
        X = torch.zeros((n, self.P, self.m), device=self.device)
        # Y:(num,features)
        Y = torch.zeros((n, 1), device=self.device)
        # Y = torch.zeros((n, self.m), device=self.device)
        # self.dat : 归一化的矩阵
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.as_tensor(self.dat[start:end, :], device=self.device)
            Y[i, :] = torch.as_tensor(self.label[idx_set[i], :], device=self.device)

        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length, device=self.device)
        else:
            index = torch.as_tensor(range(length), device=self.device, dtype=torch.long)
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]

            yield Variable(X), Variable(Y)
            start_idx += batch_size


if __name__ == '__main__':
    device = torch.device('cuda')
    Data = Data_utility('./data/airquality.xlsx', 0.7, 0.2, device, args)
    for X, Y in Data.get_batches(Data.train[0], Data.train[1], args.batch_size, True):
        print(X, Y)
