from TimeSeriesClass import *
from TSCBERT import Classifier_TSCBERT, Classifier_FCN_MHAne,Inception_TBO,Records
import inspect
from ConvTran.model import Transformer2
class MYCNNTSC(nn.Module):
    def __init__(self,mode,input_shape,lenght,hidden_size=128,pow=2,LrEnb=0,LrMo=0,device='cuda',n_cl=1,adaD=0,adaH=1,n_layers=1,pos_enc=1,ffh=-4 ):
        super(MYCNNTSC, self).__init__()
        # n_cl=n_cl
        self.mode=mode
        D2=1
        self.emb=0
        # if input_shape>200:
        #     self.embedding = nn.Sequential(
        #         nn.Linear(input_shape, input_shape // 2),
        #         nn.ReLU(),
        #         nn.Linear(input_shape // 2, 128),
        #         nn.ReLU(),
        #     )
        #     self.emb=1
        #     input_shape=128
        if mode==1:
            self.TSC = Classifier_FCN(input_shape=input_shape,D=D2)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)

        elif mode==102:
            self.TSC = Classifier_FCN_Dil(input_shape=input_shape,D=D2,Dil=2)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)

        elif mode==104:
            self.TSC = Classifier_FCN_Dil(input_shape=input_shape,D=D2,Dil=4)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)

        elif mode==2:
            self.TSC = Classifier_RESNET(input_shape=input_shape, D=D2)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)

        elif mode ==3:

            self.TSC = Classifier_FCN_FTA(input_shape=input_shape, D=D2,length=lenght,ffh=ffh)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
        elif mode ==303:

            self.TSC = Classifier_FCN_FTA_E(input_shape=input_shape, D=D2,length=lenght)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)

        elif mode ==310:

            self.TSC = Classifier_FCN_FTA_H(input_shape=input_shape, D=D2,length=lenght,h=hidden_size)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)

        elif mode ==311:

            self.TSC = Classifier_FCN_FTA_B_H(input_shape=input_shape, D=D2,length=lenght,h=hidden_size)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)

        elif mode ==301:

            self.TSC = Classifier_FCN_FTA_B(input_shape=input_shape, D=D2,length=lenght,ffh=ffh)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
        elif mode ==312:

            self.TSC = Classifier_FCN_CTA(input_shape=input_shape, D=D2,length=lenght,ffh=ffh)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
        elif mode ==302:

            self.TSC = Classifier_FCN_FTA_B_E(input_shape=input_shape, D=D2,length=lenght)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
        elif mode ==501:

            self.TSC = Classifier_E_FTA_B(input_shape=input_shape, D=D2,length=lenght,ffh=ffh)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
        elif mode == 502:

            self.TSC = Classifier_E_CTA_B(input_shape=input_shape, D=D2, length=lenght)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
        elif mode == 503:

            self.TSC = Classifier_FCN_CTA_B(input_shape=input_shape, D=D2, length=lenght)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)

        elif mode ==8:

            self.TSC = Classifier_FCN_FTA2(input_shape=input_shape, D=D2,length=lenght)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
            torch.nn.init.xavier_uniform_(self.fc.weight)
            self.fc.bias.data.zero_()
        elif mode ==85:

            self.TSC = Classifier_FCN_FTAMHA1(input_shape=input_shape, D=D2,length=lenght)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)

        elif mode ==82:

            self.TSC = Classifier_FCN_FTA22(input_shape=input_shape, D=D2,length=lenght)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
            torch.nn.init.xavier_uniform_(self.fc.weight)
            self.fc.bias.data.zero_()
        elif mode ==83:

            self.TSC = Classifier_FCN_FTA23(input_shape=input_shape, D=D2,length=lenght)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)

        elif mode ==84:

            self.TSC = Classifier_FCN_FTA24(input_shape=input_shape, D=D2,length=lenght)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)

        elif mode ==4:

            self.TSC = Classifier_BERT(input_shape=input_shape,hidden_size=input_shape,length=lenght)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
        elif mode ==400:

            self.TSC = Classifier_TSCBERT(input_shape=input_shape,hidden_size=hidden_size,length=lenght,pow=pow,LrEnb=LrEnb,LrMo=LrMo,device=device,adaD=adaD,adaH=adaH,n_layers=n_layers,pos_enc=pos_enc,ffh=ffh)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
        elif mode == 401:

            self.TSC = Classifier_FCN_MHAne(input_shape=input_shape,D=1, length=lenght, pow=pow,
                                          LrEnb=LrEnb, LrMo=LrMo, device=device,  pos_enc=pos_enc, ffh=ffh)

            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
        elif mode ==5:

            self.TSC = Classifier_FCN_MHA3(input_shape=input_shape,length=lenght,D=D2)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)

        elif mode ==7:

            self.TSC = Classifier_FCN_MHA4(input_shape=input_shape,length=lenght,D=D2)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)

        elif mode ==6:

            self.TSC = Classifier_FCN_BERT(input_shape=input_shape,length=lenght,D=D2)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)

        elif mode==9:
            self.TSC = Classifier_RESNET_MH1(input_shape=input_shape, D=D2,length=lenght)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)

        elif mode==10:
            self.TSC = Classifier_RESNET_MH2(input_shape=input_shape, D=D2,length=lenght)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)

        elif mode==11:
            self.TSC = Classifier_RESNET_MH3(input_shape=input_shape, D=D2,length=lenght)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
            torch.nn.init.xavier_uniform_(self.fc.weight)
            self.fc.bias.data.zero_()
        elif mode==12:
            self.TSC = Classifier_RESNET_FTA1(input_shape=input_shape, D=D2,length=lenght,ffh=ffh)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
        elif mode==121:
            self.TSC = Classifier_RESNET_FTA1_E(input_shape=input_shape, D=D2,length=lenght)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
        elif mode==122:
            self.TSC = Classifier_RESNET_FTA1_B(input_shape=input_shape, D=D2,length=lenght,ffh=ffh)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
        elif mode==123:
            self.TSC = Classifier_RESNET_FTA1_B_E(input_shape=input_shape, D=D2,length=lenght)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)

        elif mode==13:
            self.TSC = Classifier_RESNET_FTA2(input_shape=input_shape, D=D2,length=lenght)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
        elif mode//100==14:
            self.TSC = Inception_TBO(type=mode%10,
                length=lenght, pow=pow,LrEnb=LrEnb, LrMo=LrMo, device=device,pos_enc=pos_enc, ffh=ffh,
                                      in_channels=input_shape, n_filters=hidden_size, kernel_sizes=[9, 19, 39], bottleneck_channels=int(32/D2),
                              use_residual=True, activation=nn.ReLU(), return_indices=False)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
        
        elif mode==15:
            
            self.TSC = LSTM_FCN( input_shape, n_cl, seq_len=lenght,D=D2,mode=0)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
        elif mode==16:
            self.TSC = LSTM_FCN( input_shape, n_cl, seq_len=lenght,D=D2,mode=1)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
        elif mode == 17:

            self.TSC = Classifier_TSCBERT(input_shape=input_shape, hidden_size=hidden_size, length=lenght, pow=pow,
                                          LrEnb=LrEnb, LrMo=LrMo, device=device, adaD=5, adaH=adaH,
                                          n_layers=n_layers, pos_enc=pos_enc, ffh=ffh)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
        elif mode==18:
            self.TSC = Transformer2(input_shape=input_shape, hidden_size=hidden_size,ffh=ffh,length=lenght)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
        # torch.nn.init.xavier_uniform_(self.fc.weight)
        # self.fc.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0.05)





    def forward(self, x1,mask):
        """Extract feature vectors from input images."""
        if self.emb:
            features = self.embedding(x1)
        else:
            features=x1
        batchsize=x1.shape[0]

        if self.mode==0:
            features = features.permute(1, 0, 2)
            hiddens, (ht, ct) = self.TSC(features)
            features2 = hiddens[-1]

        else:
            # signature = inspect.signature(getattr(self.TSC, "forward", None))
            # print(signature)
            # print(signature.parameters)
            # num_inputs = len(signature.parameters)-1
            # features = features.permute(0, 2, 1)
            if self.mode==301 or self.mode==122 or self.mode==400 or self.mode==1400 or self.mode==1402 or self.mode==1408 or self.mode== 15 or self.mode== 16 or self.mode== 17 or self.mode==1409:
                features2 = self.TSC(features.permute(0, 2, 1),mask.permute(0, 2, 1)).view(batchsize, -1)
            elif self.mode ==1 or self.mode ==2 or self.mode==18:
                features2 = self.TSC(features.permute(0, 2, 1)).view(batchsize, -1)
            else:
                features2 = self.TSC(features).view(batchsize, -1)

        features3=self.drop(features2)
        outputs=self.fc(features3)




        return outputs,None
    def loss(self, regression, actuals):
        """
        use mean square error for regression and cross entropy for classification
        """
        regr_loss_fn = nn.MSELoss()
        return regr_loss_fn(regression, actuals)