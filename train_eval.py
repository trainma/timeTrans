import math
import torch.optim as optim
import torch
import numpy as np
import pandas as pd
from torch import nn
import argparse
from utils2 import *

np.seterr(divide='ignore', invalid='ignore')
from sklearn.metrics import mean_squared_error, mean_absolute_error

# parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
# parser.add_argument('--data', type=str, default='./data/airquality.xlsx', help='location of the data file')
# parser.add_argument('--model', type=str, default='transformer', help='')
# parser.add_argument('--window', type=int, default=24 * 7, help='window size')
# parser.add_argument('--horizon', type=int, default=3)
# parser.add_argument('--d_model', type=int, default=64)
# parser.add_argument('--num_layers', type=int, default=3)
# parser.add_argument('--dec_layers', type=int, default=1)
# parser.add_argument('--position', type=str, default=3)
# parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping')
# parser.add_argument('--epochs', type=int, default=200, help='upper epoch limit')
# parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size')
# parser.add_argument('--dropout', type=float, default=0.05, help='dropout applied to layers (0 = no dropout)')
# parser.add_argument('--seed', type=int, default=12555, help='random seed')
# parser.add_argument('--log_interval', type=int, default=2000, metavar='N', help='report interval')
# parser.add_argument('--save', type=str, default='save/model_air_2.pt', help='path to save the final model')
# parser.add_argument('--optim', type=str, default='adam')
# parser.add_argument('--amsgrad', type=str, default=True)
# parser.add_argument('--lr', type=float, default=0.0001)
# parser.add_argument('--L1Loss', type=bool, default=False)
# parser.add_argument('--normalize', type=int, default=2)
# args = parser.parse_args()
device = torch.device("cuda")


def calc_corr(a, b):
    a_avg = sum(a) / len(a)
    b_avg = sum(b) / len(b)
    # 计算分子，协方差————按照协方差公式，本来要除以n的，由于在相关系数中上下同时约去了n，于是可以不除以n
    cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])
    # 计算分母，方差乘积————方差本来也要除以n，在相关系数中上下同时约去了n，于是可以不除以n
    sq = math.sqrt(sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b]))

    corr_factor = cov_ab / sq
    return corr_factor


def MSE(pred, truth):
    return torch.mean(torch.square(pred - truth))


def RMSE(pred, truth):
    return torch.sqrt(torch.mean(torch.square(pred - truth)))


def evaluate(data, X, Y, model, evaluateL2, evaluateL1, args, save_csv=False):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    tmp_predict = torch.Tensor(0)
    tmp_test = torch.Tensor(0)

    for X, Y in data.get_batches(X, Y, args.batch_size, False):

        output = model(X)
        if predict is None:
            predict = output.clone().detach()  # predict torch.Size([128(batch_size), 1])
            test = Y  # test torch.Size([128(batch_size), 6(feature_size])
        else:
            predict = torch.cat((predict, output.clone().detach()))
            test = torch.cat((test, Y))

        scale = data.scale.expand(output.size(0), data.m)
        tmp_scale = scale[:, 0].view(-1, 1)
        # output = data.label_scale.inverse_transform(output.cpu().detach().numpy())
        # Y = data.label_scale.inverse_transform(Y.cpu().detach().numpy())

        # output = torch.as_tensor(output, device=device)
        # Y = torch.as_tensor(Y, device=device)

        # total_loss += float(evaluateL2(output, Y).data.item())
        # total_loss_l1 += float(evaluateL1(output, Y).data.item())

        total_loss += float(evaluateL2(output * tmp_scale, Y * tmp_scale).data.item())  # output*scale (bz,fz)(128 6)
        total_loss_l1 += float(evaluateL1(output * tmp_scale, Y * tmp_scale).data.item())

        tmp_predict = torch.cat((tmp_predict.cpu(), output.cpu().detach() * tmp_scale.cpu()), 0)
        tmp_test = torch.cat((tmp_test.cpu(), Y.cpu() * tmp_scale.cpu()), 0)

        n_samples += int((output.size(0) * data.m))

    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()

    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    # correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    # correlation = (correlation[index]).mean()

    truth = tmp_predict.reshape(-1, 1)
    test_result = tmp_test.reshape(-1, 1)

    # truth=np.array(Ytest.reshape(-1,1))
    # test_result=np.array(predict.reshape(-1,1))
    mse = MSE(test_result,truth)
    #mse = mean_squared_error(truth, test_result)
    rmse = RMSE(test_result,truth)
    mae = mean_absolute_error(truth, test_result)
    correlation = calc_corr(truth, test_result)
    truth = np.array(truth)
    test_result = np.array(test_result)

    if save_csv == True:
        df_truth = pd.DataFrame(truth)
        df_truth.columns = ['truth']
        df_test_result = pd.DataFrame(test_result)
        df_test_result.columns = ['predict']
        df_csv = pd.concat([df_truth, df_test_result], axis=1)
        df_csv.to_csv('./save/' + str(args.model) + '_pred2.csv')
        print('save csv successfully!')

    return rse, rae, correlation, rmse, mae


def train(data, X, Y, model, criterion, optim, args):
    model.train()
    total_loss = 0
    n_samples = 0
    # X: torch.Size([16, 168, 321])
    # Y: torch.Size([16, 321])
    i = 1
    for i, (X, Y) in enumerate(data.get_batches(X, Y, args.batch_size, False)):
        optim.zero_grad()
        output = model(X)
        scale = data.scale.expand(output.size(0), data.m)
        scale_tmp = scale[:, 0].view(-1, 1)
        # print(scale.shape)
        loss = criterion(output * scale_tmp, Y * scale_tmp)
        # loss = criterion(output, Y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optim.step()
        total_loss += loss.data.item()
        n_samples += int((output.size(0) * data.m))
    return total_loss / n_samples
    # return total_loss / i


def makeOptimizer(params, args):
    if args.optim == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, )
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(params, lr=args.lr, )
    elif args.optim == 'adadelta':
        optimizer = optim.Adadelta(params, lr=args.lr, )
    elif args.optim == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, )
    else:
        raise RuntimeError("Invalid optim method: " + args.method)
    return optimizer


if __name__ == '__main__':
    device = torch.device("cuda")
    Data = Data_utility(args.data, 0.7, 0.2, device, args)

    # loss function
    if args.L1Loss:
        criterion = nn.L1Loss(size_average=False)
    else:
        criterion = nn.MSELoss(size_average=False)
    evaluateL2 = nn.MSELoss(size_average=False)
    evaluateL1 = nn.L1Loss(size_average=False)

    criterion = criterion.to(device)
    evaluateL1 = evaluateL1.to(device)
    evaluateL2 = evaluateL2.to(device)

    with open(args.save, 'rb') as f:
        model = torch.load(f)
    test_acc, test_rae, test_corr, test_rmse, test_mae = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2,
                                                                  evaluateL1, args, save_csv=True)
    print('Best model performance：')
    print("| test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f} | test rmse {:5.4f} | test mae {:5.4f}".format(
        test_acc, test_rae, float(test_corr), test_rmse, test_mae))
