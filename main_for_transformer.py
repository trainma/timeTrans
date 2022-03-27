import argparse
import math
import time
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from models import LSTNet,MHA_Net,CNN,RNN,transformer
import importlib
from tqdm import tqdm
#from utils import *
from utils2 import *
from train_eval import train, evaluate, makeOptimizer

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default='./data/airquality.xlsx',help='location of the data file')
parser.add_argument('--model', type=str, default='transformer', help='')
parser.add_argument('--window', type=int, default=24 * 7,help='window size')
parser.add_argument('--horizon', type=int, default=3)
parser.add_argument('--d_model', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--dec_layers', type=int, default=1)
parser.add_argument('--position', type=str, default=3)
parser.add_argument('--clip', type=float, default=0.5,help='gradient clipping')
parser.add_argument('--epochs', type=int, default=200, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=54321,help='random seed')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',help='report interval')
parser.add_argument('--save', type=str, default='save/model_air_2.pt', help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--amsgrad', type=str, default=True)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--L1Loss', type=bool, default=False)
parser.add_argument('--normalize', type=int, default=2)
args = parser.parse_args()



# Choose device: cpu or gpu
args.cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Reproducibility.
# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)

# Load data
Data = Data_utility(args.data, 0.7, 0.2, device, args)

# loss function
if args.L1Loss:
    criterion = nn.L1Loss(size_average=False)
else:
    criterion = nn.MSELoss(size_average=False)
evaluateL2 = nn.MSELoss(size_average=False)
evaluateL1 = nn.L1Loss(size_average=False)

if args.cuda:
    criterion = criterion.to(device)
    evaluateL1 = evaluateL1.to(device)
    evaluateL2 = evaluateL2.to(device)

# Select model
# model = eval(args.model).Model(args, Data).to(device)
model = eval(args.model).TransAm(
            feature_size=args.d_model, 
            num_layers=args.num_layers, 
            dropout=args.dropout,
            dec_seq_len=args.dec_layers,
            max_len=args.window,
            batch_size=args.batch_size,
            feature_dim=Data.m).to(device)

train_method = train
eval_method = evaluate
nParams = sum([p.nelement() for p in model.parameters()])
print('number of parameters: %d' % nParams)

best_val = 10000000

optim = makeOptimizer(model.parameters(), args)

# While training you can press Ctrl + C to stop it.
try:
    print('Training start')
    for epoch in tqdm(range(1, args.epochs + 1),'training:'):
        epoch_start_time = time.time()

        train_loss = train_method(Data, Data.train[0], Data.train[1], model, criterion, optim, args)

        val_loss, val_rae, val_corr, val_rmse ,val_mae = eval_method(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1, args)
        print('| end of epoch {:3d} | time used: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.
                format( epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, float(val_corr)))

        if val_loss < best_val:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val = val_loss
        if epoch % 10 == 0:
            test_acc, test_rae, test_corr, test_rmse ,test_mae = eval_method(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,args)
            print("| test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}\n".format(test_acc, test_rae, float(test_corr)))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
test_acc, test_rae, test_corr,test_rmse ,test_mae = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,args,save_csv=True)
print('Best model performance：')
print("| test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f} | test rmse {:5.4f} | test mae {:5.4f}".format(test_acc, test_rae, float(test_corr),test_rmse ,test_mae))
