import os

os.environ['UEA_UCR_DATA_DIR'] = './Multivariate_ts/'
import uea_ucr_datasets

list_dta = uea_ucr_datasets.list_datasets()
import torch
import torch.utils.data as data
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import argparse
from mymodel import MYCNNTSC
import os
import logging
import sys
import csv
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import jaccard_score
import pandas as pd


# ###########################
#     FCN               :        mode=1,   ffh=16
#     FCN + GTA         :        mode=301, ffh=16
#     FCN + TPS         :        mode=400, adaD=3
#
#     ResNet            :        mode=2,   ffh=16
#     ResNet + GTA      :        mode=122, ffh=16
#     ResNet + TPS      :        mode=400, adaD=4
#
#     Inception32       :        mode=1400, hids=32 ,   ffh=16
#     Inception32 + GTA :        mode=1402, hids=32 ,   ffh=16
#     Inception32 + TPS :        mode=1408, hids=32
#
#
#     Inception64       :        mode=1400, hids=64 ,   ffh=16
#     Inception64 + GTA :        mode=1402, hids=64 ,   ffh=16
#     Inception64 + TPS :        mode=1408, hids=64
#
#
#     E+TPS             :        mode=400, adad=1
#     E+TPS              :        mode=400, adad=1, LrMo=3
# ---------for TPS-------
#     No PE             : pos_enc=0
#     Learnable PE      : pos_enc=2
#     Fixed Function PE : pos_enc=1
# ------
# ----- for non TPS
#     pos_enc=0
# -------------
# ############################
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', help="model name")
parser.add_argument('--root', help="data root", default='./data/c3d_feat')
parser.add_argument('--pcs', action="store_true", help="predict pcs score")
parser.add_argument('--mode', type=int, default=400)
parser.add_argument('--pow', type=int, default=2)
parser.add_argument('--no', type=int, default=0)
parser.add_argument('--LrEnb', type=int, default=1)
parser.add_argument('--LrMo', type=int, default=2)
parser.add_argument('--JobId', type=int, default=111)
parser.add_argument('--hids', type=int, default=128)
parser.add_argument('--adaD', type=int, default=3)
parser.add_argument('--adaH', type=int, default=0)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--pos_enc', type=int, default=0)  # 1,2
parser.add_argument('--ffh', type=int, default=4)

class MyDataset(uea_ucr_datasets.Dataset):
    def __init__(self, name, train=True):
        super().__init__(name, train)
        self.max_len = 0

    def set(self, max_len):
        self.max_len = max_len

    def __getitem__(self, index):
        """Get dataset instance."""
        instance = self.data_x.iloc[index, :]
        # Combine into a single dataframe and then into a numpy array
        instance_x = pd.concat(list(instance), axis=1).values
        instance_y = self.class_mapping[self.data_y[index]]
        if self.max_len:
            return np.pad(np.nan_to_num(instance_x.astype(np.float32)),
                          ((0, self.max_len - len(instance_x)), (0, 0))), instance_y
        else:
            return np.nan_to_num(instance_x.astype(np.float32)), instance_y


def reslog(line):
    with open(r'/acc.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(line)
        f.close()


def train_shuffle(logger, min_loss=200, max_acc=0, n_epochs=100, batchsize=64, ffh=4):
    round_max_spea = 0
    round_min_loss = 200
    if args.mode == 400 or args.mode == 1406 or args.mode == 1408:
        div = 2
    else:
        div = 2

    trainset = MyDataset(list_dta[args.no], train=True)

    testset = MyDataset(list_dta[args.no], train=False)
    tmp = 0
    for sample in trainset:
        tmp = max(tmp, len(sample[0]))
    for sample in testset:
        tmp = max(tmp, len(sample[0]))

    trainset.set(tmp)

    testset.set(tmp)

    trainLoader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batchsize, shuffle=True, num_workers=0)
    testLoader = torch.utils.data.DataLoader(testset,
                                             batch_size=batchsize, shuffle=False, num_workers=0)

    # build the model
    scoring = MYCNNTSC(mode=args.mode, input_shape=trainset.n_channels, hidden_size=args.hids, n_cl=trainset.n_classes,
                       lenght=tmp // div if args.no == 6 else tmp, pow=args.pow, LrEnb=args.LrEnb, LrMo=args.LrMo,
                       device=device, adaD=args.adaD, adaH=args.adaH, n_layers=args.n_layers, pos_enc=args.pos_enc,
                       ffh=ffh)
    if torch.cuda.is_available():
        scoring.cuda()  # turn the model into gpu
    total_params = sum(p.numel() for p in scoring.parameters() if p.requires_grad)
    # optimizer = optim.Adam(params=scoring.parameters(), lr=0.001)  # use SGD optimizer to optimize the loss function
    optimizer = optim.Adam(params=scoring.parameters(), lr=0.0001)  # use SGD optimizer to optimize the loss function
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=20, cooldown=20, min_lr=0.000001)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(n_epochs):  # total 40 epoches

        print("Epoch:  " + str(epoch) + "Total Params: %d" % total_params)
        total_cl_loss = 0
        total_sample = 0
        for i, (features, scores) in enumerate(trainLoader):  # get mini-batch
            
            if torch.cuda.is_available():
                features = Variable(features).cuda()
                scores = Variable(scores).cuda()

            if args.no == 6: ##to fit the mommory constraint samples of this dataset are divide into two across time classified seperatly and the scores are meaned.
                features = features.contiguous().view(-1, tmp // div, trainset.n_channels)
            logits, penal = scoring(features)

            if args.no == 6:
                scores = torch.repeat_interleave(scores, div, dim=0)
            if penal is None:

                cl_loss = loss_fn(logits, scores)
            else:
                cl_loss = scoring.loss(logits, scores) + penal

            # back propagation
            optimizer.zero_grad()
            cl_loss.backward()
            optimizer.step()
            total_cl_loss += cl_loss.data.item() * scores.shape[0]
            total_sample += scores.shape[0]

        print("Classification Loss: " + str(total_cl_loss / total_sample) + '\n')

        scoring.eval()
        val_pred = []
        val_sample = 0
        val_loss = 0
        val_truth = []
        for j, (features, scores) in enumerate(testLoader):
            val_truth.append(scores.numpy())
            bssss = features.shape[0]
            if torch.cuda.is_available():
                features = Variable(features).cuda()
                scores = Variable(scores).cuda()
            if args.no == 6:
                features = features.contiguous().view(-1, tmp // div, trainset.n_channels)
            cl_out, _ = scoring(features)
            if args.no == 6:
                cl_out = torch.sum(cl_out.contiguous().view(bssss, div, -1), dim=1)
            val_pred.append(torch.max(cl_out, dim=1)[1].data.cpu().numpy())

            cl_loss = loss_fn(cl_out, scores)
            val_loss += (cl_loss.data.item()) * scores.shape[0]
            val_sample += scores.shape[0]
        val_truth = np.concatenate(val_truth)
        val_pred = np.concatenate(val_pred)
        val_sr = np.sum(val_pred == val_truth) / len(val_truth)

        scheduler.step(val_loss / val_sample)
        if val_sr >= max_acc:
            torch.save(scoring.state_dict(), name + '.pth')
        min_loss = min(min_loss, val_loss / val_sample)
        if val_sr > max_acc:
            pre, rec, fsS, _ = precision_recall_fscore_support(val_truth, val_pred
                                                               ,
                                                               average='macro')

            J_s = jaccard_score(val_truth, val_pred,
                                average='macro')
        max_acc = max(max_acc, val_sr)
        round_min_loss = min(round_min_loss, val_loss / val_sample)
        round_max_spea = max(val_sr, round_max_spea)
        print("Val Loss: %.2f Correlation: %.2f Min Val Loss: %.2f Max Correlation: %.2f" %
              (val_loss / val_sample, val_sr, min_loss, max_acc))
        logger.info(
            ',{},{},{},{},{}'.format(val_loss / val_sample, val_sr, min_loss, max_acc, optimizer.param_groups[0]['lr']))
        scoring.train()
    return min_loss, max_acc, pre, rec, fsS, J_s


if __name__ == '__main__':
    min_loss = 200
    max_acc = 0
    args = parser.parse_args()
    save_path = '/log_'  # specify model and log save location path
    name = save_path + list_dta[args.no] + '_{}_'.format(args.mode) + (''.join(sys.argv[1:]))
    logF = name + '.csv'
    print(list_dta[args.no])

    # In order to fit the time-limit and memory limit of the servers we modified the batch size and number of epochs.
    if args.no == 22:
        ffh = 4
    else:
        ffh = args.ffh
    if args.no == 10:
        epoch = 100
    elif args.no == 21:

        epoch = 200
    elif args.no == 15:
        epoch = 10
    elif args.no == 6:
        epoch = 100
    else:
        epoch = 400
    if args.no == 9:
        batchsize = 16
    elif args.no == 19:
        batchsize = 4
    elif args.no == 6:
        batchsize = 2
    elif args.no == 26:
        batchsize = 32
    else:
        batchsize = 64

    if (os.path.isfile(logF)):
        os.remove(logF)
    logging.basicConfig(filename=logF,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger('myloger')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(
        ',loss,acc,minloss,maxAcc,lr')
    for time in range(1):
        min_loss, max_acc, pre, rec, fsS, J_s = train_shuffle(logger, min_loss, max_acc, epoch, batchsize, ffh)
        reslog([args.no, args.mode, args.JobId, args.adaD, args.adaH, args.hids, args.n_layers, args.pos_enc, ffh,
                args.pow, args.LrEnb, args.LrMo, min_loss, pre, rec, fsS, J_s, max_acc])
