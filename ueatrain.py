import os
os.environ['UEA_UCR_DATA_DIR'] = './Multivariate_ts/'
import uea_ucr_datasets
list_dta=uea_ucr_datasets.list_datasets()

# for a in list_dta:
#     d = uea_ucr_datasets.Dataset(a, train=True)
#     print(a)
#     print(d.n_channels)
#     print(d.n_classes)
    # first_instance = d[0]

from dataloader import videoDataset, transform
# from model import Scoring
import torch.nn as nn
import torch
import torch.utils.data as data
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
from scipy.stats import spearmanr as sr
import random
import argparse
from mymodel import MYCNNTSC
import os
import logging
import sys
import csv
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import jaccard_score, roc_curve, auc
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', help="model name")
parser.add_argument('--root', help="data root",default='./data/c3d_feat')
parser.add_argument('--pcs', action="store_true", help="predict pcs score")
parser.add_argument('--mode', type=int, default=400)
parser.add_argument('--pow', type=int, default=2)
parser.add_argument('--no', type=int, default=5)
parser.add_argument('--LrEnb', type=int, default=1)
parser.add_argument('--LrMo', type=int, default=2)
parser.add_argument('--JobId', type=int, default=111)
parser.add_argument('--hids', type=int, default=128)
parser.add_argument('--adaD', type=int, default=2)
parser.add_argument('--adaH', type=int, default=0)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--pos_enc', type=int, default=2)#1,2
parser.add_argument('--ffh', type=int, default=4)#-4,-2,0,2,4

from scipy.ndimage.filters import uniform_filter1d

class find_low_change_sec:
    def __init__(self,mode=0,mean_zero=False, ends_only=True):
        self.mode=mode
        self.mean_zero=mean_zero
        self.ends_only=ends_only
        # if mode==1:
        #     self.rpt_model=rpt.Pelt(model="rbf")


    def smooth_mask_matrix(self,mask_matrix, window_size=3, threshold=0.5):
        smoothed_matrix = uniform_filter1d(mask_matrix, window_size)
        # smoothed_matrix = np.max(smoothed_matrix,axis=0)
        smoothed_matrix = np.where(smoothed_matrix >= threshold, 1, 0)
        return smoothed_matrix
    def detect(self,signal):
        signal=signal.T
        if self.mode==0:
            mask=np.ones(signal.shape)
            signal_diff=np.diff(signal,axis=1)
            signal_diff=np.concatenate((signal_diff[:,0:1],signal_diff),axis=1)
            signal_diff=np.abs(signal_diff)

            var_mat=np.std(signal_diff,axis=1).repeat(signal.shape[1]).reshape(signal.shape)

            mask[signal_diff<var_mat/10]=0
            mask=np.max(mask,axis=0)
            mask = self.smooth_mask_matrix(mask_matrix=mask)
            mask2=np.zeros(signal.shape)
            mask2[signal_diff>var_mat/4]=1
            mask2=np.max(mask2,axis=0)

            mask= mask + mask2
            mask[mask>0]=1

        # elif self.mode==1:
        #     algo=rpt.Pelt(model="rbf").fit(signal)
        #     cp=algo.predict(pen=0.01)
        #     cp=np.concatenate(([0],cp,[signal.shape[1]]))
        #     mask=np.ones(signal.shape[1])
        #     for i in range(len(cp)-1):
        #         tmp_var=np.std(signal[:,cp[i]:cp[i+1]],axis=1)
        #         tmp_sym=np.zeros(signal.shape[0])
        #         tmp_sym[tmp_var<np.std(signal,axis=1)/10]=1
        #         if tmp_sym.sum()==1:
        #             mask[cp[i]:cp[i+1]]=0
        
        
        if self.mean_zero:
            mask_cp=np.where(np.abs(np.diff(mask))==1)[0]
            mask_cp=np.concatenate(([0],mask_cp+1,[signal.shape[1]]))
            for i in range(len(mask_cp)-1):
                if mask[mask_cp[i]]==0:
                    mean=np.mean(signal[:,mask_cp[i]:mask_cp[i+1]])
                    if not abs(mean)<0.1:
                        mask[mask_cp[i]:mask_cp[i+1]]=1
        if self.ends_only:
            first_in=np.where(mask==1)[0][0]
            last_in=np.where(mask==1)[0][-1]
            mask=np.ones(mask.shape)
            if first_in>0:
                mask[:first_in-1]=0
            if last_in<mask.shape[0]-1:
                mask[last_in+1:]=0

        return mask

class MyDataset(uea_ucr_datasets.Dataset):
    def __init__(self, name, train=True):
        super().__init__( name, train)
        self.max_len=0
        self.mask_gen=find_low_change_sec(mode=0,mean_zero=False, ends_only=True)
        self.mask_dict={}
    def set(self,max_len):
        self.max_len=max_len
    def __getitem__(self, index):
        """Get dataset instance."""
        instance = self.data_x.iloc[index, :]
        # Combine into a single dataframe and then into a numpy array
        instance_x = pd.concat(list(instance), axis=1).values
        instance_y = self.class_mapping[self.data_y[index]]
        # mask=np.ones(instance_x.shape)
        if index not in self.mask_dict.keys():
            mask=self.mask_gen.detect(instance_x).repeat(instance_x.shape[1]).reshape(instance_x.shape[0],-1)
            self.mask_dict[index]=mask
        else:
            mask=self.mask_dict[index]
        if self.max_len:

            return np.pad(np.nan_to_num(instance_x.astype(np.float32)), ((0, self.max_len - len(instance_x)), (0, 0))), instance_y ,np.pad(mask, ((0, self.max_len - len(instance_x)), (0, 0)))
        else:
            return np.nan_to_num(instance_x.astype(np.float32)), instance_y, mask

def reslog(line):
    with open(r'./logs/acc_sp.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(line)
        f.close()
def train_shuffle(logger,min_mse=200, max_corr=0,n_epochs=100,batchsize=64,ffh=4):
    round_max_spea = 0
    round_min_mse = 200
    if args.mode==400 or args.mode==1406 or args.mode==1408:
        div=2
    else:
        
        div=2
    

    trainset=MyDataset(list_dta[args.no], train=True)


    testset = MyDataset(list_dta[args.no], train=False)
    tmp=0
    for sample in trainset:
        tmp=max(tmp, len(sample[0]))
    for sample in testset:
        tmp=max(tmp, len(sample[0]))
    trainset.set(tmp)
    testset.set(tmp)
    # testset = videoDataset(root=args.root,
    #                        label="./data/test_dataset.txt", suffix='.npy', transform=transform, data=None, pcss=args.pcs)
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])
    trainLoader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=batchsize, shuffle=True, num_workers=0)
    valLoader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=batchsize, shuffle=False, num_workers=0)
    testLoader = torch.utils.data.DataLoader(testset,
                                             batch_size=batchsize, shuffle=False, num_workers=0)

    # build the model
    scoring = MYCNNTSC(mode=args.mode,input_shape=trainset.n_channels,hidden_size=args.hids,n_cl=trainset.n_classes,lenght=tmp//div if args.no==6 else tmp,pow=args.pow,LrEnb=args.LrEnb,LrMo=args.LrMo,device=device,adaD=args.adaD,adaH=args.adaH,n_layers=args.n_layers,pos_enc=args.pos_enc,ffh=ffh)
    if torch.cuda.is_available():
        scoring.cuda()  # turn the model into gpu
    # scoring.load_state_dict(torch.load("./models/merge/pcs.pt"))
    total_params = sum(p.numel() for p in scoring.parameters() if p.requires_grad)
    # loss_log.write("Total Params: " + str(total_params) + '\n')
    # optimizer = optim.Adam(params=scoring.parameters(), lr=0.001)  # use SGD optimizer to optimize the loss function
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=50, cooldown=20, min_lr=0.000001)
    optimizer = optim.Adam(params=scoring.parameters(), lr= 0.0001)  # use SGD optimizer to optimize the loss function
    # optimizer = optim.SGD(params=scoring.parameters(), lr=0.001)  # , weight_decay=10**(-3))
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=20, cooldown=20, min_lr=0.000001)
    loss_fn=torch.nn.CrossEntropyLoss()
    vali_sets={'val':valLoader,'test':testLoader}
    test_acc = 0
    test_loss = 0
    for epoch in range(n_epochs):  # total 40 epoches
        # scheduler.step()
        print("Epoch:  " + str(epoch) + "Total Params: %d" % total_params)
        total_regr_loss = 0
        total_sample = 0
        for i, (features, scores, mask) in enumerate(trainLoader):  # get mini-batch
            # print("%d batches have done" % i)
            if torch.cuda.is_available():
                features = Variable(features).cuda()
                scores = Variable(scores).cuda()
                mask = Variable(mask).cuda()
            # regression, logits = scoring(features)
            if args.no==6:
                features=features.contiguous().view(-1, tmp//div,trainset.n_channels)
                mask=mask.contiguous().view(-1, tmp//div,trainset.n_channels)
            logits, penal = scoring(features,mask)
            if args.no==6:
                #logits=torch.sum(logits.contiguous().view(batchsize, div,-1),dim=1)
                scores=torch.repeat_interleave(scores,div,dim=0)
            if penal is None:
                # regr_loss = scoring.loss(logits, scores)
                regr_loss=loss_fn(logits,scores)
            else:
                regr_loss = scoring.loss(logits, scores) + penal
            # new three lines are back propagation
            optimizer.zero_grad()
            regr_loss.backward()
            # nn.utils.clip_grad_norm(scoring.parameters(), 1.5)
            optimizer.step()
            total_regr_loss += regr_loss.data.item() * scores.shape[0]
            total_sample += scores.shape[0]

        # loss_log.write(str(total_regr_loss / total_sample) + '\n')

        print("Classification Loss: " + str(total_regr_loss / total_sample) + '\n')
        # the rest is used to evaluate the model with the test dataset
        # torch.save(scoring.state_dict(), './models/epoch{}.pt'.format(epoch))
        scoring.eval()
        save=0

        for phase in ['val','test']:
            save=0 if phase=='val' else save
            val_pred = []
            val_sample = 0
            val_loss = 0
            val_truth = []
            for j, (features, scores,mask) in enumerate(testLoader):
                val_truth.append(scores.numpy())
                bssss=features.shape[0]
                if torch.cuda.is_available():
                    features = Variable(features).cuda()
                    scores = Variable(scores).cuda()
                    mask = Variable(mask).cuda()
                if args.no==6:
                    features=features.contiguous().view(-1, tmp//div,trainset.n_channels)
                    mask=mask.contiguous().view(-1, tmp//div,trainset.n_channels)
                regression, _ = scoring(features,mask)
                if args.no==6:
                    regression=torch.sum(regression.contiguous().view(bssss, div,-1),dim=1)
                val_pred.append(torch.max(regression,dim=1)[1].data.cpu().numpy())
                # regr_loss = scoring.loss(regression, scores)
                regr_loss = loss_fn(regression, scores)
                val_loss += (regr_loss.data.item()) * scores.shape[0]
                val_sample += scores.shape[0]
            val_truth = np.concatenate(val_truth)
            val_pred = np.concatenate(val_pred)
            val_sr=np.sum(val_pred==val_truth)/len(val_truth)
            # val_sr, _ = sr(val_truth, val_pred)
            # if val_loss / val_sample < min_mse:

            if phase=='val':
                scheduler.step(val_loss / val_sample)
                min_mse = min(min_mse, val_loss / val_sample)
                max_corr = max(max_corr, val_sr)
                if val_sr >= max_corr:
                    save=1

            if save==1 and phase=='test':
                torch.save(scoring.state_dict(), name+'.pth')
                test_acc=val_sr
                test_loss=val_loss / val_sample



                pre, rec, fsS, _ = precision_recall_fscore_support(val_truth,val_pred
                                                                              ,
                                                                              average='macro')

                J_s = jaccard_score(val_truth, val_pred,
                                        average='macro')

            # round_min_mse = min(round_min_mse, val_loss / val_sample)
            # round_max_spea = max(val_sr, round_max_spea)
            if phase=='test':
                print("Test Loss: %.2f Correlation: %.2f min val Test Loss: %.2f Max val acc test Correlation: %.2f" %
                      (val_loss / val_sample, val_sr, test_loss, test_acc))
                logger.info(',{},{},{},{},{}'.format(val_loss / val_sample, val_sr, test_loss, test_acc,optimizer.param_groups[0]['lr']))
                scoring.train()
    # w.write('MSE: %.2f spearman: %.2f' % (round_min_mse, round_max_spea))
    # savemat(name + '_att_' + '.mat', SaveMatdict)
    return test_loss, test_acc,pre,rec,fsS,J_s


if __name__ == '__main__':
    min_mse = 200
    max_corr = 0
    args = parser.parse_args()
    name = './logs/log_' + list_dta[args.no] + '_{}_'.format(args.mode) + (''.join(sys.argv[1:]))
    logF = name + '.csv'
    print(list_dta[args.no])
    if args.no==22:
        ffh=4
    else:
        ffh=args.ffh
    if args.no==10:
        epoch=100
    elif args.no==21:
    
        epoch=200
    elif args.no==15:
        epoch=10
    elif args.no==6:
    
        #epoch=100
        epoch=100
    else:
        epoch=400
    if args.no==9:
        batchsize=16
    elif args.no==19:
        batchsize=4
    elif args.no==6:
        batchsize=2
    elif args.no==26:
        batchsize=32
    else:
        batchsize=64

    # epoch = 200 if args.no == 10 else 400
    if (os.path.isfile(logF)):
        os.remove(logF)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=logF,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger('myloger')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(
        ',loss,acc,minloss,maxAcc,lr')
    print('config')
    for time in range(1):
        min_mse, max_corr,pre,rec,fsS,J_s = train_shuffle(logger,min_mse, max_corr,epoch,batchsize,ffh)
        reslog([args.no,args.mode,args.JobId,args.adaD,args.adaH,args.hids,args.n_layers,args.pos_enc,ffh,args.pow,args.LrEnb,args.LrMo,min_mse,pre,rec,fsS,J_s,max_corr])
