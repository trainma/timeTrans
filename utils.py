import torch
import numpy as np
from torch.autograd import Variable


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, device, args):
        self.device = device
        self.P = args.window
        self.h = args.horizon
        fin = open(file_name)
        self.rawdat = np.loadtxt(fin, delimiter=',')  # (26304,321)
        self.dat = np.zeros(self.rawdat.shape)  # (26304,321)
        self.n, self.m = self.dat.shape  # n=26304 m=321
        self.scale = np.ones(self.m)

        self._normalized(args.normalize)
        # 分训练样本 测试样本
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale = torch.as_tensor(self.scale, device=device, dtype=torch.float)

        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)

        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))
        fin.close()

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

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
        Y = torch.zeros((n, self.m), device=self.device)
        # self.dat : 归一化的矩阵
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.as_tensor(self.dat[start:end, :], device=self.device)
            Y[i, :] = torch.as_tensor(self.dat[idx_set[i], :], device=self.device)

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
