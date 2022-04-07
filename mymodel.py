from TSCBERT import Classifier_TSCBERT, Classifier_FCN_MHAne, Inception_TBO
from TimeSeriesClass import *


class MYCNNTSC(nn.Module):
    def __init__(self, mode, input_shape, lenght, hidden_size=128, pow=2, LrEnb=0, LrMo=0, device='cuda', n_cl=1,
                 adaD=0, adaH=1, n_layers=1, pos_enc=1, ffh=-4):
        super(MYCNNTSC, self).__init__()

        self.mode = mode
        D2 = 1
        self.emb = 0

        if mode == 1:
            self.TSC = Classifier_FCN(input_shape=input_shape, D=D2)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)

        elif mode == 2:
            self.TSC = Classifier_RESNET(input_shape=input_shape, D=D2)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)

        elif mode == 3:

            self.TSC = Classifier_FCN_FTA(input_shape=input_shape, D=D2, length=lenght, ffh=ffh)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)

        elif mode == 301:

            self.TSC = Classifier_FCN_FTA_B(input_shape=input_shape, D=D2, length=lenght, ffh=ffh)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
        elif mode == 400:

            self.TSC = Classifier_TSCBERT(input_shape=input_shape, hidden_size=hidden_size, length=lenght, pow=pow,
                                          LrEnb=LrEnb, LrMo=LrMo, device=device, adaD=adaD, adaH=adaH,
                                          n_layers=n_layers, pos_enc=pos_enc, ffh=ffh)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
        elif mode == 122:
            self.TSC = Classifier_RESNET_FTA1_B(input_shape=input_shape, D=D2, length=lenght, ffh=ffh)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)
        elif mode // 100 == 14:
            self.TSC = Inception_TBO(type=mode % 10,
                                     length=lenght, pow=pow, LrEnb=LrEnb, LrMo=LrMo, device=device, pos_enc=pos_enc,
                                     ffh=ffh,
                                     in_channels=input_shape, n_filters=hidden_size, kernel_sizes=[9, 19, 39],
                                     bottleneck_channels=int(32 / D2),
                                     use_residual=True, activation=nn.ReLU(), return_indices=False)
            self.drop = nn.Dropout(p=0.7)
            self.fc = nn.Linear(in_features=self.TSC.out_shape, out_features=n_cl)

        # torch.nn.init.xavier_uniform_(self.fc.weight)
        # self.fc.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0.05)

    def forward(self, x1):
        """Extract feature vectors from input images."""
        if self.emb:
            features = self.embedding(x1)
        else:
            features = x1
        batchsize = x1.shape[0]

        if self.mode == 0:
            features = features.permute(1, 0, 2)
            hiddens, (ht, ct) = self.TSC(features)
            features2 = hiddens[-1]

        else:
            features = features.permute(0, 2, 1)
            features2 = self.TSC(features).view(batchsize, -1)

        features3 = self.drop(features2)
        outputs = self.fc(features3)

        return outputs, None

    def loss(self, regression, actuals):
        """
        use mean square error for regression and cross entropy for classification
        """
        regr_loss_fn = nn.MSELoss()
        return regr_loss_fn(regression, actuals)
