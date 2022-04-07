import torch
import torch.nn as nn


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, kernel_size=8, padding=3):
        super(BasicBlock1D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.conv1 = nn.Conv1d(in_channels=inplanes, out_channels=planes, kernel_size=kernel_size, padding=padding)
        self.bn1 = norm_layer(planes, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out


class BasicBlock1D_Dil(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, kernel_size=8, padding=3,
                 Dilation=1):
        super(BasicBlock1D_Dil, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.conv1 = nn.Conv1d(in_channels=inplanes, out_channels=planes, kernel_size=kernel_size, padding=padding,
                               dilation=Dilation)
        self.bn1 = norm_layer(planes, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out


class ResBlock1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_layer=None):
        super(ResBlock1D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.conv1 = BasicBlock1D(inplanes=inplanes, planes=planes, kernel_size=7, padding=3, norm_layer=norm_layer)

        self.conv2 = BasicBlock1D(inplanes=planes, planes=planes, kernel_size=5, padding=2, norm_layer=norm_layer)

        self.conv3 = nn.Conv1d(in_channels=planes, out_channels=planes, kernel_size=3, padding=1)
        self.bn3 = norm_layer(planes, momentum=0.01)
        self.downsample = nn.Conv1d(in_channels=inplanes, out_channels=planes, kernel_size=1, padding=0)
        self.bnd = norm_layer(planes, momentum=0.01)
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            identity = self.bnd(identity)

        out += identity
        out = self.relu(out)

        return out


class Classifier_FCN(nn.Module):

    def __init__(self, input_shape, D=1):
        super(Classifier_FCN, self).__init__()

        self.input_shape = input_shape
        self.out_shape = int(128 / D)
        # self.conv0 = BasicBlock1D(inplanes=input_shape, planes=32, kernel_size=1, padding=0)
        self.conv1 = BasicBlock1D(inplanes=input_shape, planes=int(128 / D), kernel_size=8, padding=3)

        self.conv2 = BasicBlock1D(inplanes=int(128 / D), planes=int(256 / D), kernel_size=5, padding=2)

        self.conv3 = BasicBlock1D(inplanes=int(256 / D), planes=int(128 / D), kernel_size=3, padding=1)

        self.AVG = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.AVG(x)
        return x


class Classifier_FCN_Dil(nn.Module):

    def __init__(self, input_shape, D=1, Dil=4):
        super(Classifier_FCN_Dil, self).__init__()

        self.input_shape = input_shape
        self.out_shape = int(128 / D)
        # self.conv0 = BasicBlock1D(inplanes=input_shape, planes=32, kernel_size=1, padding=0)
        self.conv1 = BasicBlock1D_Dil(inplanes=input_shape, planes=int(128 / D), kernel_size=8,
                                      padding=int((8 - 1) * Dil / 2), Dilation=Dil)

        self.conv2 = BasicBlock1D_Dil(inplanes=int(128 / D), planes=int(256 / D), kernel_size=5,
                                      padding=int((5 - 1) * Dil / 2), Dilation=Dil)

        self.conv3 = BasicBlock1D_Dil(inplanes=int(256 / D), planes=int(128 / D), kernel_size=3,
                                      padding=int((3 - 1) * Dil / 2), Dilation=Dil)

        self.AVG = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.AVG(x)
        return x


class Classifier_FCN_BERT(nn.Module):

    def __init__(self, input_shape, D=1, length=96):
        super(Classifier_FCN_BERT, self).__init__()

        self.input_shape = input_shape
        self.out_shape = int(128 / D)

        self.conv1 = BasicBlock1D(inplanes=input_shape, planes=int(128 / D), kernel_size=7, padding=3)

        self.conv2 = BasicBlock1D(inplanes=int(128 / D), planes=int(256 / D), kernel_size=5, padding=2)

        self.conv3 = BasicBlock1D(inplanes=int(256 / D), planes=int(128 / D), kernel_size=3, padding=1)

        # self.AVG = nn.AdaptiveAvgPool1d(1)
        self.TSC = Classifier_BERT(input_shape=int(128 / D),
                                   hidden_size=int(128 / D), length=length)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.TSC(x)
        # x=self.AVG(x)
        return x


class Se1Block(nn.Module):

    def __init__(self, channel, reduction=16):
        super(Se1Block, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class Se1Block_drop(nn.Module):

    def __init__(self, channel, reduction=16):
        super(Se1Block_drop, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.drop(y)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class Se1Block_dropB(nn.Module):

    def __init__(self, channel, length, reduction=16):
        super(Se1Block_dropB, self).__init__()

        self.fc0 = nn.Sequential(nn.Linear(channel, 1, bias=False),
                                 nn.ReLU(inplace=True)
                                 )
        self.fc = nn.Sequential(
            nn.Linear(length, length // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(length // reduction, length, bias=False),
            nn.Sigmoid()
        )
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        b, l, _ = x.size()
        y = self.fc0(x).view(b, l)
        y = self.drop(y)
        y = self.fc(y).view(b, l, 1)
        return x * y.expand_as(x)


class Se2Block_drop(nn.Module):

    def __init__(self, channel, reduction=16):
        super(Se2Block_drop, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x, out):
        b, c, _ = x.size()
        y = out
        y = self.drop(y)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class FTABlock(nn.Module):

    def __init__(self, channel=24, reduction=16):
        super(FTABlock, self).__init__()
        # self.SE=Se1Block(channel, reduction=reduction)
        self.SE = Se1Block_drop(channel, reduction=reduction)

    def _addRI(self, input):
        input = torch.stack([input, torch.zeros_like(input)], -1)
        return input

    # def forward(self,x):
    #     BS = x.shape
    #     xf = torch.fft(self._addRI(x),2)
    #     xf = xf.permute(0,3,2,1).contiguous().view(-1,BS[2],BS[1])
    #     xf = self.SE(xf)
    #     xf = xf.view(BS[0],-1,BS[2],BS[1]).permute(0,3,2,1)
    #     x = torch.ifft(xf,2)[...,0]
    #     return x
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.SE(x)
        x = x.permute(0, 2, 1)
        return x


class FTABlock_H(nn.Module):

    def __init__(self, channel=24, reduction=16, h=4):
        super(FTABlock_H, self).__init__()
        # self.SE=Se1Block(channel, reduction=reduction)
        assert channel % h == 0
        self.SE = nn.ModuleList(Se1Block_drop(channel // h, reduction=reduction) for _ in range(h))
        self.h = h

    def forward(self, x):
        z = []
        xs = x.shape
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(xs[0], self.h, -1, xs[1])
        for i, SE in enumerate(self.SE):
            z.append(SE(x[:, i, :, :]))
        x = torch.cat(z, dim=1)
        x = x.contiguous().view(xs[0], -1, xs[1])
        x = x.permute(0, 2, 1)
        return x


class FTABlockB(nn.Module):

    def __init__(self, channel=64, length=24, reduction=16):
        super(FTABlockB, self).__init__()
        # self.SE=Se1Block(channel, reduction=reduction)
        self.SE = Se1Block_dropB(channel=channel, length=length, reduction=reduction)

    def _addRI(self, input):
        input = torch.stack([input, torch.zeros_like(input)], -1)
        return input

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.SE(x)
        x = x.permute(0, 2, 1)
        return x


class FTABlockB_H(nn.Module):

    def __init__(self, channel=64, length=24, reduction=16, h=4):
        super(FTABlockB_H, self).__init__()
        # self.SE=Se1Block(channel, reduction=reduction)
        assert length % h == 0
        self.SE = nn.ModuleList(Se1Block_drop(length // h, reduction=reduction) for _ in range(h))
        self.h = h

    def forward(self, x):
        z = []
        xs = x.shape
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(xs[0], self.h, -1, xs[1])
        for i, SE in enumerate(self.SE):
            z.append(SE(x[:, i, :, :]))
        x = torch.cat(z, dim=1)
        x = x.contiguous().view(xs[0], -1, xs[1])
        x = x.permute(0, 2, 1)
        return x


class FTABlock2(nn.Module):

    def __init__(self, channel=24, reduction=16):
        super(FTABlock2, self).__init__()
        # self.SE=Se1Block(channel, reduction=reduction)
        self.SE = Se2Block_drop(channel, reduction=reduction)

    def _addRI(self, input):
        input = torch.stack([input, torch.zeros_like(input)], -1)
        return input

    # def forward(self,x):
    #     BS = x.shape
    #     xf = torch.fft(self._addRI(x),2)
    #     xf = xf.permute(0,3,2,1).contiguous().view(-1,BS[2],BS[1])
    #     xf = self.SE(xf)
    #     xf = xf.view(BS[0],-1,BS[2],BS[1]).permute(0,3,2,1)
    #     x = torch.ifft(xf,2)[...,0]
    #     return x
    def forward(self, x, out):
        x = x.permute(0, 2, 1)
        x = self.SE(x, out)
        x = x.permute(0, 2, 1)
        return x


class Classifier_FCN_FTA(nn.Module):

    def __init__(self, input_shape, D=1, length=24, ffh=16):
        super(Classifier_FCN_FTA, self).__init__()

        self.input_shape = input_shape
        self.out_shape = int(128 / D)

        # self.embedding = BERTEmbedding2(input_dim=input_shape, max_len=length)
        self.conv1 = BasicBlock1D(inplanes=input_shape, planes=int(128 / D), kernel_size=8, padding=3)
        self.FTA1 = FTABlock(channel=length - 1, reduction=ffh)

        self.conv2 = BasicBlock1D(inplanes=int(128 / D), planes=int(256 / D), kernel_size=5, padding=2)
        self.FTA2 = FTABlock(channel=length - 1, reduction=ffh)

        self.conv3 = BasicBlock1D(inplanes=int(256 / D), planes=int(128 / D), kernel_size=3, padding=1)
        self.FTA3 = FTABlock(channel=length - 1, reduction=ffh)

        self.AVG = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.conv1(x)
        x = self.FTA1(x)
        x = self.conv2(x)
        x = self.FTA2(x)
        x = self.conv3(x)
        x = self.FTA3(x)
        x = self.AVG(x)
        return x


class Classifier_FCN_FTA_E(nn.Module):

    def __init__(self, input_shape, D=1, length=24):
        super(Classifier_FCN_FTA_E, self).__init__()

        self.input_shape = input_shape
        self.out_shape = int(128 / D)

        # self.embedding = BERTEmbedding2(input_dim=input_shape, max_len=length)
        self.conv1 = BasicBlock1D(inplanes=input_shape, planes=int(128 / D), kernel_size=8, padding=3)
        # self.FTA1 = FTABlock(channel=length-1)

        self.conv2 = BasicBlock1D(inplanes=int(128 / D), planes=int(256 / D), kernel_size=5, padding=2)
        # self.FTA2 = FTABlock(channel=length-1)

        self.conv3 = BasicBlock1D(inplanes=int(256 / D), planes=int(128 / D), kernel_size=3, padding=1)
        self.FTA3 = FTABlock(channel=length - 1)

        self.AVG = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.conv1(x)
        # x = self.FTA1(x)
        x = self.conv2(x)
        # x = self.FTA2(x)
        x = self.conv3(x)
        x = self.FTA3(x)
        x = self.AVG(x)
        return x


class Classifier_FCN_FTA_B(nn.Module):

    def __init__(self, input_shape, D=1, length=24, ffh=16):
        super(Classifier_FCN_FTA_B, self).__init__()

        self.input_shape = input_shape
        self.out_shape = int(128 / D)

        self.embedding = BERTEmbedding2(input_dim=input_shape, max_len=length)
        self.conv1 = BasicBlock1D(inplanes=input_shape, planes=int(128 / D), kernel_size=7, padding=3)
        self.FTA1 = FTABlockB(channel=int(128 / D), length=length, reduction=ffh)

        self.conv2 = BasicBlock1D(inplanes=int(128 / D), planes=int(256 / D), kernel_size=5, padding=2)
        self.FTA2 = FTABlockB(channel=int(256 / D), length=length, reduction=ffh)

        self.conv3 = BasicBlock1D(inplanes=int(256 / D), planes=int(128 / D), kernel_size=3, padding=1)
        self.FTA3 = FTABlockB(channel=int(128 / D), length=length, reduction=ffh)

        self.AVG = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.conv1(x)
        x = self.FTA1(x)
        x = self.conv2(x)
        x = self.FTA2(x)
        x = self.conv3(x)
        x = self.FTA3(x)
        x = self.AVG(x)
        return x


class Classifier_FCN_FTA_B_E(nn.Module):

    def __init__(self, input_shape, D=1, length=24):
        super(Classifier_FCN_FTA_B_E, self).__init__()

        self.input_shape = input_shape
        self.out_shape = int(128 / D)

        self.embedding = BERTEmbedding2(input_dim=input_shape, max_len=length)
        self.conv1 = BasicBlock1D(inplanes=input_shape, planes=int(128 / D), kernel_size=7, padding=3)
        # self.FTA1 = FTABlockB(channel=int(128 / D),length=length)

        self.conv2 = BasicBlock1D(inplanes=int(128 / D), planes=int(256 / D), kernel_size=5, padding=2)
        # self.FTA2 = FTABlockB(channel=int(256 / D),length=length)

        self.conv3 = BasicBlock1D(inplanes=int(256 / D), planes=int(128 / D), kernel_size=3, padding=1)
        self.FTA3 = FTABlockB(channel=int(128 / D), length=length)

        self.AVG = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.conv1(x)
        # x = self.FTA1(x)
        x = self.conv2(x)
        # x = self.FTA2(x)
        x = self.conv3(x)
        x = self.FTA3(x)
        x = self.AVG(x)
        return x


class Classifier_FCN_FTA_H(nn.Module):

    def __init__(self, input_shape, D=1, length=24, h=4):
        super(Classifier_FCN_FTA_H, self).__init__()

        self.input_shape = input_shape
        self.out_shape = int(128 / D)

        self.embedding = BERTEmbedding2(input_dim=input_shape, max_len=length)
        self.conv1 = BasicBlock1D(inplanes=input_shape, planes=int(128 / D), kernel_size=7, padding=3)
        self.FTA1 = FTABlock_H(channel=length, h=h)

        self.conv2 = BasicBlock1D(inplanes=int(128 / D), planes=int(256 / D), kernel_size=5, padding=2)
        self.FTA2 = FTABlock_H(channel=length, h=h)

        self.conv3 = BasicBlock1D(inplanes=int(256 / D), planes=int(128 / D), kernel_size=3, padding=1)
        self.FTA3 = FTABlock_H(channel=length, h=h)

        self.AVG = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.conv1(x)
        x = self.FTA1(x)
        x = self.conv2(x)
        x = self.FTA2(x)
        x = self.conv3(x)
        x = self.FTA3(x)
        x = self.AVG(x)
        return x


class Classifier_E_FTA_B(nn.Module):

    def __init__(self, input_shape, D=1, length=24, ffh=16):
        super(Classifier_E_FTA_B, self).__init__()

        self.input_shape = input_shape
        self.out_shape = int(128 / D)
        if input_shape >= 128:
            # if hidden_size>=128:
            self.embedding = nn.Sequential(
                nn.Linear(input_shape, input_shape // 2),
                nn.ReLU(),
                nn.Linear(input_shape // 2, 128),
                nn.ReLU(),
            )
        else:
            self.embedding = nn.Sequential(
                nn.Linear(input_shape, input_shape * 2),
                nn.ReLU(),
                nn.Linear(input_shape * 2, 128),
                nn.ReLU(),
            )

        self.FTA = FTABlockB(channel=int(128 / D), length=length, reduction=ffh)

        self.AVG = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)

        # x = self.embedding(x)
        x = self.FTA(x)
        x = self.AVG(x)
        return x


class Classifier_E_CTA_B(nn.Module):

    def __init__(self, input_shape, D=1, length=24):
        super(Classifier_E_CTA_B, self).__init__()

        self.input_shape = input_shape
        self.out_shape = int(128 / D)
        if input_shape >= 128:
            # if hidden_size>=128:
            self.embedding = nn.Sequential(
                nn.Linear(input_shape, input_shape // 2),
                nn.ReLU(),
                nn.Linear(input_shape // 2, 128),
                nn.ReLU(),
            )
        else:
            self.embedding = nn.Sequential(
                nn.Linear(input_shape, input_shape * 2),
                nn.ReLU(),
                nn.Linear(input_shape * 2, 128),
                nn.ReLU(),
            )

        self.FTA = CTABlockB(channel=int(128 / D), length=length)

        self.AVG = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)

        # x = self.embedding(x)
        x = self.FTA(x)
        x = self.AVG(x)
        return x


# class CTABlockB(nn.Module):

#    def __init__(self, channel=64,length=24, reduction=16):
#        super(CTABlockB, self).__init__()
#        # self.SE=Se1Block(channel, reduction=reduction)
#        # self.SE = Se1Block_dropB(channel=channel,length=length, reduction=reduction)
#        self.fc0 = nn.Sequential(nn.Linear(channel, 1, bias=False),
#                                 nn.Sigmoid()
#                                 )

#    def _addRI(self, input):
#        input = torch.stack([input, torch.zeros_like(input)], -1)
#        return input

#    def forward(self, x):
#        x = x.permute(0, 2, 1)
#        x = x*self.fc0(x).expand_as(x)
#
#        x = x.permute(0, 2, 1)
#        return x

class CTABlockB(nn.Module):

    def __init__(self, channel=64, length=24, reduction=16):
        super(CTABlockB, self).__init__()
        # self.SE=Se1Block(channel, reduction=reduction)
        # self.SE = Se1Block_dropB(channel=channel,length=length, reduction=reduction)
        self.fc0 = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False),
                                 nn.ReLU(),
                                 nn.Linear(channel // reduction, 1, bias=False),
                                 nn.Sigmoid()
                                 )

    def _addRI(self, input):
        input = torch.stack([input, torch.zeros_like(input)], -1)
        return input

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x * self.fc0(x).expand_as(x)

        x = x.permute(0, 2, 1)
        return x


class Classifier_FCN_CTA_B(nn.Module):

    def __init__(self, input_shape, D=1, length=24):
        super(Classifier_FCN_CTA_B, self).__init__()

        self.input_shape = input_shape
        self.out_shape = int(128 / D)

        self.conv1 = BasicBlock1D(inplanes=input_shape, planes=int(128 / D), kernel_size=7, padding=3)
        self.FTA1 = CTABlockB(channel=int(128 / D), length=length)

        self.conv2 = BasicBlock1D(inplanes=int(128 / D), planes=int(256 / D), kernel_size=5, padding=2)
        self.FTA2 = CTABlockB(channel=int(256 / D), length=length)

        self.conv3 = BasicBlock1D(inplanes=int(256 / D), planes=int(128 / D), kernel_size=3, padding=1)
        self.FTA3 = CTABlockB(channel=int(128 / D), length=length)

        # self.FTA = CTABlockB(channel=int(128 / D), length=length)

        self.AVG = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)

        # x = self.embedding(x)
        x = self.conv1(x)
        x = self.FTA1(x)
        x = self.conv2(x)
        x = self.FTA2(x)
        x = self.conv3(x)
        x = self.FTA3(x)
        x = self.AVG(x)
        return x


class Classifier_FCN_FTA_B_H(nn.Module):

    def __init__(self, input_shape, D=1, length=24, h=4):
        super(Classifier_FCN_FTA_B_H, self).__init__()

        self.input_shape = input_shape
        self.out_shape = int(128 / D)

        self.embedding = BERTEmbedding2(input_dim=input_shape, max_len=length)
        self.conv1 = BasicBlock1D(inplanes=input_shape, planes=int(128 / D), kernel_size=7, padding=3)
        self.FTA1 = FTABlockB_H(channel=int(128 / D), length=length, h=h)

        self.conv2 = BasicBlock1D(inplanes=int(128 / D), planes=int(256 / D), kernel_size=5, padding=2)
        self.FTA2 = FTABlockB_H(channel=int(256 / D), length=length, h=h)

        self.conv3 = BasicBlock1D(inplanes=int(256 / D), planes=int(128 / D), kernel_size=3, padding=1)
        self.FTA3 = FTABlockB_H(channel=int(128 / D), length=length, h=h)

        self.AVG = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.conv1(x)
        x = self.FTA1(x)
        x = self.conv2(x)
        x = self.FTA2(x)
        x = self.conv3(x)
        x = self.FTA3(x)
        x = self.AVG(x)
        return x


class Classifier_FCN_FTA2(nn.Module):

    def __init__(self, input_shape, D=1, length=24):
        super(Classifier_FCN_FTA2, self).__init__()

        self.input_shape = input_shape
        self.out_shape = int(128 / D)

        self.embedding = BERTEmbedding2(input_dim=input_shape, max_len=length)
        self.FTA0 = FTABlock2(channel=length)
        self.conv1 = BasicBlock1D(inplanes=input_shape, planes=int(128 / D), kernel_size=7, padding=3)
        self.FTA1 = FTABlock(channel=length)

        self.conv2 = BasicBlock1D(inplanes=int(128 / D), planes=int(256 / D), kernel_size=5, padding=2)
        self.FTA2 = FTABlock(channel=length)

        self.conv3 = BasicBlock1D(inplanes=int(256 / D), planes=int(128 / D), kernel_size=3, padding=1)
        self.FTA3 = FTABlock(channel=length)

        self.AVG = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, out):
        # x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.FTA0(x, out)
        x = self.conv1(x)
        x = self.FTA1(x)
        x = self.conv2(x)
        x = self.FTA2(x)
        x = self.conv3(x)
        x = self.FTA3(x)
        x = self.AVG(x)
        return x


class Classifier_FCN_FTAMHA1(nn.Module):

    def __init__(self, input_shape, D=1, length=24):
        super(Classifier_FCN_FTAMHA1, self).__init__()

        self.input_shape = input_shape
        self.out_shape = int(128 / D)

        self.embedding = BERTEmbedding2(input_dim=input_shape, max_len=length)
        self.FTA0 = FTABlock2(channel=length)
        self.conv1 = BasicBlock1D(inplanes=input_shape, planes=int(128 / D), kernel_size=7, padding=3)
        self.FTA1 = FTABlock(channel=length)

        self.conv2 = BasicBlock1D(inplanes=int(128 / D), planes=int(256 / D), kernel_size=5, padding=2)
        self.FTA2 = FTABlock(channel=length)

        self.conv3 = BasicBlock1D(inplanes=int(256 / D), planes=int(128 / D), kernel_size=3, padding=1)
        self.FTA3 = FTABlock(channel=length)
        self.MHA3 = MHA(int(128 / D), self.attheads, self.drop)
        self.AVG = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, out):
        batch_size = x.shape[0]
        if self.training:
            bernolliMatrix = torch.tensor([self.mask_prob]).float().cuda().repeat(self.max_len).unsqueeze(0).repeat(
                [batch_size, 1])
            self.bernolliDistributor = torch.distributions.Bernoulli(bernolliMatrix)
            sample = self.bernolliDistributor.sample()
            mask = (sample > 0).unsqueeze(1).repeat(1, sample.size(1), 1).unsqueeze(1)
        else:
            mask = torch.ones(batch_size, 1, self.max_len, self.max_len).cuda()
        x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.FTA0(x, out)
        x = self.conv1(x)
        x = self.FTA1(x)
        x = self.conv2(x)
        x = self.FTA2(x)
        x = self.conv3(x)
        x = self.FTA3(x)
        x = self.MHA3(x, mask)
        x = self.AVG(x)
        return x


class Classifier_FCN_FTA22(nn.Module):

    def __init__(self, input_shape, D=1, length=24):
        super(Classifier_FCN_FTA22, self).__init__()

        self.input_shape = input_shape
        self.out_shape = int(128 / D)

        self.embedding = BERTEmbedding2(input_dim=input_shape, max_len=length)
        self.FTA0 = FTABlock2(channel=length)
        self.conv1 = BasicBlock1D(inplanes=input_shape, planes=int(128 / D), kernel_size=7, padding=3)
        self.FTA1 = FTABlock(channel=length)

        self.conv2 = BasicBlock1D(inplanes=int(128 / D), planes=int(256 / D), kernel_size=5, padding=2)
        self.FTA2 = FTABlock(channel=length)

        self.conv3 = BasicBlock1D(inplanes=int(256 / D), planes=int(128 / D), kernel_size=3, padding=1)
        self.FTA3 = FTABlock(channel=length)

        self.AVG = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, out):
        # x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.FTA0(x, out)
        x = self.conv1(x)
        # x = self.FTA1(x)
        x = self.conv2(x)
        # x = self.FTA2(x)
        x = self.conv3(x)
        # x = self.FTA3(x)
        x = self.AVG(x)
        return x


class Classifier_FCN_FTA23(nn.Module):

    def __init__(self, input_shape, D=1, length=24):
        super(Classifier_FCN_FTA23, self).__init__()

        self.input_shape = input_shape
        self.out_shape = int(128 / D)

        self.embedding = BERTEmbedding2(input_dim=input_shape, max_len=length)
        self.FTA0 = FTABlock2(channel=length)
        self.conv1 = BasicBlock1D(inplanes=input_shape, planes=int(128 / D), kernel_size=7, padding=3)
        self.FTA1 = FTABlock(channel=length)

        self.conv2 = BasicBlock1D(inplanes=int(128 / D), planes=int(256 / D), kernel_size=5, padding=2)
        self.FTA2 = FTABlock(channel=length)

        self.conv3 = BasicBlock1D(inplanes=int(256 / D), planes=int(128 / D), kernel_size=3, padding=1)
        self.FTA3 = FTABlock(channel=length)

        self.AVG = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, out):
        # x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.FTA0(x, out)
        x = self.conv1(x)
        x = self.FTA1(x)
        x = self.conv2(x)
        # x = self.FTA2(x)
        x = self.conv3(x)
        # x = self.FTA3(x)
        x = self.AVG(x)
        return x


class Classifier_FCN_FTA24(nn.Module):

    def __init__(self, input_shape, D=1, length=24):
        super(Classifier_FCN_FTA24, self).__init__()

        self.input_shape = input_shape
        self.out_shape = int(128 / D)

        self.embedding = BERTEmbedding2(input_dim=input_shape, max_len=length)
        self.FTA0 = FTABlock2(channel=length)
        self.conv1 = BasicBlock1D(inplanes=input_shape, planes=int(128 / D), kernel_size=7, padding=3)
        self.FTA1 = FTABlock(channel=length)

        self.conv2 = BasicBlock1D(inplanes=int(128 / D), planes=int(256 / D), kernel_size=5, padding=2)
        self.FTA2 = FTABlock(channel=length)

        self.conv3 = BasicBlock1D(inplanes=int(256 / D), planes=int(128 / D), kernel_size=3, padding=1)
        self.FTA3 = FTABlock(channel=length)

        self.AVG = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, out):
        # x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.FTA0(x, out)
        x = self.conv1(x)
        x = self.FTA1(x)
        x = self.conv2(x)
        x = self.FTA2(x)
        x = self.conv3(x)
        # x = self.FTA3(x)
        x = self.AVG(x)
        return x


class MHA(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        # self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        # self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = x.permute(0, 2, 1)
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        # x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x).permute(0, 2, 1)


class Classifier_FCN_MHA1(nn.Module):

    def __init__(self, input_shape, D=1, length=24):

        super(Classifier_FCN_MHA1, self).__init__()

        self.input_shape = input_shape
        self.out_shape = int(128 / D)
        self.attheads = 8
        self.drop = 0.5
        self.mask_prob = 0.5
        self.max_len = length
        self.embedding = BERTEmbedding(input_dim=input_shape, max_len=length)

        self.conv1 = BasicBlock1D(inplanes=input_shape, planes=int(128 / D), kernel_size=7, padding=3)
        self.MHA1 = MHA(int(64 / D), self.attheads, self.drop)

        self.conv2 = BasicBlock1D(inplanes=int(128 / D), planes=int(256 / D), kernel_size=5, padding=2)
        self.MHA2 = MHA(int(128 / D), self.attheads, self.drop)

        self.conv3 = BasicBlock1D(inplanes=int(256 / D), planes=int(128 / D), kernel_size=3, padding=1)
        self.MHA3 = MHA(int(256 / D), self.attheads * 2, self.drop)

        self.AVG = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        batch_size = x.shape[0]
        if self.training:
            bernolliMatrix = torch.tensor([self.mask_prob]).float().cuda().repeat(self.max_len).unsqueeze(0).repeat(
                [batch_size, 1])
            self.bernolliDistributor = torch.distributions.Bernoulli(bernolliMatrix)
            sample = self.bernolliDistributor.sample()
            mask = (sample > 0).unsqueeze(1).repeat(1, sample.size(1), 1).unsqueeze(1)
        else:
            mask = torch.ones(batch_size, 1, self.max_len, self.max_len).cuda()

        x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.MHA1(x, mask)
        x = self.conv1(x)
        x = self.MHA2(x, mask)
        x = self.conv2(x)
        x = self.MHA3(x, mask)
        x = self.conv3(x)

        x = self.AVG(x)
        return x


class Classifier_FCN_MHA2(nn.Module):

    def __init__(self, input_shape, D=1, length=24):

        super(Classifier_FCN_MHA2, self).__init__()

        self.input_shape = input_shape
        self.out_shape = int(128 / D)
        self.attheads = 8
        self.drop = 0.5
        self.mask_prob = 0.5
        self.max_len = length
        self.embedding = BERTEmbedding2(input_dim=input_shape, max_len=length)

        self.conv1 = BasicBlock1D(inplanes=input_shape, planes=int(128 / D), kernel_size=7, padding=3)
        self.MHA1 = MHA(int(64 / D), self.attheads, self.drop)

        self.conv2 = BasicBlock1D(inplanes=int(128 / D), planes=int(256 / D), kernel_size=5, padding=2)
        self.MHA2 = MHA(int(128 / D), self.attheads, self.drop)

        self.conv3 = BasicBlock1D(inplanes=int(256 / D), planes=int(128 / D), kernel_size=3, padding=1)
        self.MHA3 = MHA(int(256 / D), self.attheads * 2, self.drop)

        self.AVG = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        batch_size = x.shape[0]
        if self.training:
            bernolliMatrix = torch.tensor([self.mask_prob]).float().cuda().repeat(self.max_len).unsqueeze(0).repeat(
                [batch_size, 1])
            self.bernolliDistributor = torch.distributions.Bernoulli(bernolliMatrix)
            sample = self.bernolliDistributor.sample()
            mask = (sample > 0).unsqueeze(1).repeat(1, sample.size(1), 1).unsqueeze(1)
        else:
            mask = torch.ones(batch_size, 1, self.max_len, self.max_len).cuda()

        x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.MHA1(x, mask)
        x = self.conv1(x)
        x = self.MHA2(x, mask)
        x = self.conv2(x)
        x = self.MHA3(x, mask)
        x = self.conv3(x)

        x = self.AVG(x)
        return x


class Classifier_FCN_MHA3(nn.Module):

    def __init__(self, input_shape, D=1, length=24):

        super(Classifier_FCN_MHA3, self).__init__()

        self.input_shape = input_shape
        self.out_shape = int(128 / D)
        self.attheads = 8
        self.drop = 0.5
        self.mask_prob = 0.5
        self.max_len = length
        self.embedding = BERTEmbedding2(input_dim=input_shape, max_len=length)

        self.conv1 = BasicBlock1D(inplanes=input_shape, planes=int(128 / D), kernel_size=7, padding=3)
        # self.MHA1 = MHA(int(128/D),self.attheads,self.drop)

        self.conv2 = BasicBlock1D(inplanes=int(128 / D), planes=int(256 / D), kernel_size=5, padding=2)
        # self.MHA2 = MHA(int(256/D),self.attheads*2,self.drop)

        self.conv3 = BasicBlock1D(inplanes=int(256 / D), planes=int(128 / D), kernel_size=3, padding=1)
        self.MHA3 = MHA(int(128 / D), self.attheads, self.drop)

        self.AVG = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        batch_size = x.shape[0]
        if self.training:
            bernolliMatrix = torch.tensor([self.mask_prob]).float().cuda().repeat(self.max_len).unsqueeze(0).repeat(
                [batch_size, 1])
            self.bernolliDistributor = torch.distributions.Bernoulli(bernolliMatrix)
            sample = self.bernolliDistributor.sample()
            mask = (sample > 0).unsqueeze(1).repeat(1, sample.size(1), 1).unsqueeze(1)
        else:
            mask = torch.ones(batch_size, 1, self.max_len, self.max_len).cuda()

        x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        # x = self.MHA3(x, mask)
        x = self.conv1(x)
        # x = self.MHA1(x,mask)
        x = self.conv2(x)
        # x = self.MHA2(x,mask)
        x = self.conv3(x)
        x = self.MHA3(x, mask)

        x = self.AVG(x)
        return x


class Classifier_FCN_MHA4(nn.Module):

    def __init__(self, input_shape, D=1, length=24):

        super(Classifier_FCN_MHA4, self).__init__()

        self.input_shape = input_shape
        self.out_shape = int(128 / D)
        self.attheads = 8
        self.drop = 0.5
        self.mask_prob = 0.5
        self.max_len = length
        self.embedding = BERTEmbedding2(input_dim=input_shape, max_len=length)

        self.conv1 = BasicBlock1D(inplanes=input_shape, planes=int(128 / D), kernel_size=7, padding=3)
        # self.MHA1 = MHA(int(128/D),self.attheads,self.drop)

        self.conv2 = BasicBlock1D(inplanes=int(128 / D), planes=int(256 / D), kernel_size=5, padding=2)
        # self.MHA2 = MHA(int(256/D),self.attheads*2,self.drop)

        self.conv3 = BasicBlock1D(inplanes=int(256 / D), planes=int(128 / D), kernel_size=3, padding=1)
        self.MHA3 = MHA(int(128 / D), self.attheads, self.drop)

        self.AVG = nn.AdaptiveAvgPool1d(1)

        clsToken = torch.zeros(1, 1, int(128 / D)).float().cuda()
        clsToken.require_grad = True
        self.clsToken = nn.Parameter(clsToken)
        torch.nn.init.normal_(clsToken, std=0.02)

    def forward(self, x):
        batch_size = x.shape[0]

        if self.training:
            bernolliMatrix = torch.cat((torch.tensor([1]).float().cuda(),
                                        (torch.tensor([self.mask_prob]).float().cuda()).repeat(self.max_len)),
                                       0).unsqueeze(0).repeat([batch_size, 1])
            self.bernolliDistributor = torch.distributions.Bernoulli(bernolliMatrix)
            sample = self.bernolliDistributor.sample()
            mask = (sample > 0).unsqueeze(1).repeat(1, sample.size(1), 1).unsqueeze(1)
        else:
            mask = torch.ones(batch_size, 1, self.max_len + 1, self.max_len + 1).cuda()

        x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        # x = self.MHA3(x, mask)
        x = self.conv1(x)
        # x = self.MHA1(x,mask)
        x = self.conv2(x)
        # x = self.MHA2(x,mask)
        x = self.conv3(x)

        x = torch.cat((self.clsToken.repeat(batch_size, 1, 1), x.permute(0, 2, 1)), 1).permute(0, 2, 1)
        x = self.MHA3(x, mask)
        x = x.permute(0, 2, 1)

        # x=self.AVG(x)
        return x[:, 0, :]


class Classifier_FCN_MHA5(nn.Module):

    def __init__(self, input_shape, D=1, length=24):

        super(Classifier_FCN_MHA5, self).__init__()

        self.input_shape = input_shape
        self.out_shape = input_shape
        self.attheads = 8
        self.drop = 0.5
        self.mask_prob = 0.5
        self.max_len = length
        self.embedding = BERTEmbedding2(input_dim=input_shape, max_len=length + 1)

        # self.conv1 = BasicBlock1D(inplanes=input_shape,planes=int(128/D),kernel_size=7,padding=3)
        # self.MHA1 = MHA(int(128/D),self.attheads,self.drop)

        # self.conv2 = BasicBlock1D(inplanes=int(128/D), planes = int(256/D), kernel_size=5, padding=2)
        # self.MHA2 = MHA(int(256/D),self.attheads*2,self.drop)

        # self.conv3 = BasicBlock1D(inplanes=int(256/D), planes = int(128/D), kernel_size=3, padding=1)
        self.MHA3 = MHA(input_shape, self.attheads, self.drop)

        # self.AVG = nn.AdaptiveAvgPool1d(1)

        clsToken = torch.zeros(1, 1, input_shape).float().cuda()
        clsToken.require_grad = True
        self.clsToken = nn.Parameter(clsToken)
        torch.nn.init.normal_(clsToken, std=0.02)

    def forward(self, x):
        batch_size = x.shape[0]
        input_vectors = x
        norm = input_vectors.norm(p=2, dim=-2, keepdim=True)  # it on the feature dimentsion not time
        x = input_vectors.div(norm)

        if self.training:
            bernolliMatrix = torch.cat((torch.tensor([1]).float().cuda(),
                                        (torch.tensor([self.mask_prob]).float().cuda()).repeat(self.max_len)),
                                       0).unsqueeze(0).repeat([batch_size, 1])
            self.bernolliDistributor = torch.distributions.Bernoulli(bernolliMatrix)
            sample = self.bernolliDistributor.sample()
            mask = (sample > 0).unsqueeze(1).repeat(1, sample.size(1), 1).unsqueeze(1)
        else:
            mask = torch.ones(batch_size, 1, self.max_len + 1, self.max_len + 1).cuda()

        x = torch.cat((self.clsToken.repeat(batch_size, 1, 1), x.permute(0, 2, 1)), 1).permute(0, 2, 1)
        x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.MHA3(x, mask).permute(0, 2, 1)
        x = x[:, 0, :]
        # x = x.permute(0, 2, 1)
        # x = self.MHA3(x, mask)
        # x=self.conv1(x)
        # x = self.MHA1(x,mask)
        # x=self.conv2(x)
        # x = self.MHA2(x,mask)
        # x=self.conv3(x)

        # x=self.AVG(x)
        return x


class Classifier_BERT(nn.Module):

    def __init__(self, input_shape, hidden_size=512, D=1, length=24):
        super(Classifier_BERT, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = 1
        self.attn_heads = 1
        self.length = length
        self.bert = BERT5(hidden_size, self.length, hidden=self.hidden_size, n_layers=self.n_layers,
                          attn_heads=self.attn_heads)
        self.out_shape = self.hidden_size
        if input_shape >= hidden_size:
            # if hidden_size>=128:
            self.embedding = nn.Sequential(
                nn.Linear(input_shape, input_shape // 2),
                nn.ReLU(),
                nn.Linear(input_shape // 2, hidden_size),
                nn.ReLU(),
            )
            self.hidden_size = hidden_size

        else:
            self.embedding = nn.Sequential(
                nn.Linear(input_shape, input_shape * 2),
                nn.ReLU(),
                nn.Linear(input_shape * 2, hidden_size),
                nn.ReLU(),
            )
            self.hidden_size = hidden_size

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.embedding(x)
        input_vectors = x
        norm = input_vectors.norm(p=2, dim=-1, keepdim=True)
        input_vectors = input_vectors.div(norm)
        output, maskSample = self.bert(input_vectors)
        classificationOut = output[:, 0, :]
        # sequenceOut = output[:, 1:, :]
        # norm = sequenceOut.norm(p=2, dim=-1, keepdim=True)
        # sequenceOut = sequenceOut.div(norm)
        # output = self.dp(classificationOut)
        # x = self.fc_action(output)

        return classificationOut


class Classifier_RESNET(nn.Module):

    def __init__(self, input_shape, D=1):
        super(Classifier_RESNET, self).__init__()
        self.out_shape = int(128 / D)
        self.blk1 = ResBlock1D(input_shape, int(64 / D))
        self.blk2 = ResBlock1D(int(64 / D), int(128 / D))
        self.blk3 = ResBlock1D(int(128 / D), int(128 / D))
        self.AVG = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.AVG(x)
        return x


class Classifier_RESNET_MH1(nn.Module):

    def __init__(self, input_shape, D=1, length=96):
        super(Classifier_RESNET_MH1, self).__init__()
        self.out_shape = int(128 / D)
        self.blk1 = ResBlock1D(input_shape, int(64 / D))
        self.blk2 = ResBlock1D(int(64 / D), int(128 / D))
        self.blk3 = ResBlock1D(int(128 / D), int(128 / D))
        self.AVG = nn.AdaptiveAvgPool1d(1)
        self.attheads = 8
        self.drop = 0.5
        self.mask_prob = 0.5
        self.max_len = length
        self.embedding = BERTEmbedding2(input_dim=input_shape, max_len=length)

        # self.conv1 = BasicBlock1D(inplanes=input_shape, planes=int(128 / D), kernel_size=7, padding=3)
        self.MHA1 = MHA(int(128 / D), self.attheads, self.drop)

    def forward(self, x):
        batch_size = x.shape[0]
        if self.training:
            bernolliMatrix = torch.tensor([self.mask_prob]).float().cuda().repeat(self.max_len).unsqueeze(0).repeat(
                [batch_size, 1])
            self.bernolliDistributor = torch.distributions.Bernoulli(bernolliMatrix)
            sample = self.bernolliDistributor.sample()
            mask = (sample > 0).unsqueeze(1).repeat(1, sample.size(1), 1).unsqueeze(1)
        else:
            mask = torch.ones(batch_size, 1, self.max_len, self.max_len).cuda()

        x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.MHA1(x, mask)
        x = self.AVG(x)
        return x


class Classifier_RESNET_MH2(nn.Module):

    def __init__(self, input_shape, D=1, length=96):
        super(Classifier_RESNET_MH2, self).__init__()
        self.out_shape = int(128 / D)
        self.blk1 = ResBlock1D(input_shape, int(64 / D))
        self.blk2 = ResBlock1D(int(64 / D), int(128 / D))
        self.blk3 = ResBlock1D(int(128 / D), int(128 / D))
        self.AVG = nn.AdaptiveAvgPool1d(1)
        self.attheads = 8
        self.drop = 0.5
        self.mask_prob = 0.5
        self.max_len = length
        self.embedding = BERTEmbedding2(input_dim=input_shape, max_len=length)

        # self.conv1 = BasicBlock1D(inplanes=input_shape, planes=int(128 / D), kernel_size=7, padding=3)
        # self.MHA1 = MHA(int(64/D),self.attheads,self.drop)
        self.MHA1 = MHA(input_shape, 1, self.drop)

    def forward(self, x):
        batch_size = x.shape[0]
        if self.training:
            bernolliMatrix = torch.tensor([self.mask_prob]).float().cuda().repeat(self.max_len).unsqueeze(0).repeat(
                [batch_size, 1])
            self.bernolliDistributor = torch.distributions.Bernoulli(bernolliMatrix)
            sample = self.bernolliDistributor.sample()
            mask = (sample > 0).unsqueeze(1).repeat(1, sample.size(1), 1).unsqueeze(1)
        else:
            mask = torch.ones(batch_size, 1, self.max_len, self.max_len).cuda()

        x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.MHA1(x, mask)
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)

        x = self.AVG(x)
        return x


class Classifier_RESNET_MH3(nn.Module):

    def __init__(self, input_shape, D=1, length=96):
        super(Classifier_RESNET_MH3, self).__init__()
        self.out_shape = int(128 / D)
        self.blk1 = ResBlock1D(input_shape, int(64 / D))
        self.blk2 = ResBlock1D(int(64 / D), int(128 / D))
        self.blk3 = ResBlock1D(int(128 / D), int(128 / D))
        self.AVG = nn.AdaptiveAvgPool1d(1)
        self.attheads = 8
        self.drop = 0.5
        self.mask_prob = 0.5
        self.max_len = length
        self.embedding = BERTEmbedding2(input_dim=input_shape, max_len=length)

        # self.conv1 = BasicBlock1D(inplanes=input_shape, planes=int(128 / D), kernel_size=7, padding=3)
        self.MHA1 = MHA(int(128 / D), self.attheads, self.drop)
        clsToken = torch.zeros(1, 1, int(128 / D)).float().cuda()
        clsToken.require_grad = True
        self.clsToken = nn.Parameter(clsToken)
        torch.nn.init.normal_(clsToken, std=0.02)

    def forward(self, x):
        batch_size = x.shape[0]
        if self.training:
            bernolliMatrix = torch.cat((torch.tensor([1]).float().cuda(),
                                        (torch.tensor([self.mask_prob]).float().cuda()).repeat(self.max_len)),
                                       0).unsqueeze(0).repeat([batch_size, 1])
            self.bernolliDistributor = torch.distributions.Bernoulli(bernolliMatrix)
            sample = self.bernolliDistributor.sample()
            mask = (sample > 0).unsqueeze(1).repeat(1, sample.size(1), 1).unsqueeze(1)
        else:
            mask = torch.ones(batch_size, 1, self.max_len + 1, self.max_len + 1).cuda()

        x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = torch.cat((self.clsToken.repeat(batch_size, 1, 1), x.permute(0, 2, 1)), 1).permute(0, 2, 1)
        x = self.MHA1(x, mask)
        x = x.permute(0, 2, 1)

        # x=self.AVG(x)
        return x[:, 0, :]


class Classifier_RESNET_FTA1(nn.Module):

    def __init__(self, input_shape, D=1, length=96, ffh=16):
        super(Classifier_RESNET_FTA1, self).__init__()
        self.out_shape = int(128 / D)
        self.blk1 = ResBlock1D(input_shape, int(64 / D))
        self.blk2 = ResBlock1D(int(64 / D), int(128 / D))
        self.blk3 = ResBlock1D(int(128 / D), int(128 / D))
        self.AVG = nn.AdaptiveAvgPool1d(1)
        self.FTA0 = FTABlock2(channel=length)
        self.conv1 = BasicBlock1D(inplanes=input_shape, planes=int(128 / D), kernel_size=7, padding=3)
        self.FTA1 = FTABlock(channel=length, reduction=ffh)
        self.FTA2 = FTABlock(channel=length, reduction=ffh)
        self.FTA3 = FTABlock(channel=length, reduction=ffh)
        # self.FTA4 = FTABlock(channel=length)

    def forward(self, x):
        # x = self.FTA0(x, out)

        x = self.blk1(x)
        x = self.FTA1(x)

        x = self.blk2(x)
        x = self.FTA2(x)

        x = self.blk3(x)
        x = self.FTA3(x)

        x = self.AVG(x)
        return x


class Classifier_RESNET_FTA1_E(nn.Module):

    def __init__(self, input_shape, D=1, length=96):
        super(Classifier_RESNET_FTA1_E, self).__init__()
        self.out_shape = int(128 / D)
        self.blk1 = ResBlock1D(input_shape, int(64 / D))
        self.blk2 = ResBlock1D(int(64 / D), int(128 / D))
        self.blk3 = ResBlock1D(int(128 / D), int(128 / D))
        self.AVG = nn.AdaptiveAvgPool1d(1)
        # self.FTA0 = FTABlock2(channel=length)
        self.conv1 = BasicBlock1D(inplanes=input_shape, planes=int(128 / D), kernel_size=7, padding=3)
        # self.FTA1 = FTABlock(channel=length)
        # self.FTA2 = FTABlock(channel=length)
        self.FTA3 = FTABlock(channel=length)
        # self.FTA4 = FTABlock(channel=length)

    def forward(self, x):
        # x = self.FTA0(x, out)

        x = self.blk1(x)
        # x = self.FTA1(x)

        x = self.blk2(x)
        # x = self.FTA2(x)

        x = self.blk3(x)
        x = self.FTA3(x)

        x = self.AVG(x)
        return x


class Classifier_RESNET_FTA1_B(nn.Module):

    def __init__(self, input_shape, D=1, length=96, ffh=16):
        super(Classifier_RESNET_FTA1_B, self).__init__()
        self.out_shape = int(128 / D)
        self.blk1 = ResBlock1D(input_shape, int(64 / D))
        self.blk2 = ResBlock1D(int(64 / D), int(128 / D))
        self.blk3 = ResBlock1D(int(128 / D), int(128 / D))
        self.AVG = nn.AdaptiveAvgPool1d(1)
        # self.FTA0 = FTABlock2(channel=length)
        self.conv1 = BasicBlock1D(inplanes=input_shape, planes=int(128 / D), kernel_size=7, padding=3)
        self.FTA1 = FTABlockB(channel=int(64 / D), length=length, reduction=ffh)
        self.FTA2 = FTABlockB(channel=int(128 / D), length=length, reduction=ffh)
        self.FTA3 = FTABlockB(channel=int(128 / D), length=length, reduction=ffh)
        # self.FTA4 = FTABlock(channel=length)

    def forward(self, x):
        # x = self.FTA0(x, out)

        x = self.blk1(x)
        x = self.FTA1(x)

        x = self.blk2(x)
        x = self.FTA2(x)

        x = self.blk3(x)
        x = self.FTA3(x)

        x = self.AVG(x)
        return x


class Classifier_RESNET_FTA1_B_E(nn.Module):

    def __init__(self, input_shape, D=1, length=96):
        super(Classifier_RESNET_FTA1_B_E, self).__init__()
        self.out_shape = int(128 / D)
        self.blk1 = ResBlock1D(input_shape, int(64 / D))
        self.blk2 = ResBlock1D(int(64 / D), int(128 / D))
        self.blk3 = ResBlock1D(int(128 / D), int(128 / D))
        self.AVG = nn.AdaptiveAvgPool1d(1)
        # self.FTA0 = FTABlock2(channel=length)
        self.conv1 = BasicBlock1D(inplanes=input_shape, planes=int(128 / D), kernel_size=7, padding=3)
        # self.FTA1 = FTABlock(channel=length)
        # self.FTA2 = FTABlock(channel=length)
        self.FTA3 = FTABlockB(channel=int(128 / D), length=length)
        # self.FTA4 = FTABlock(channel=length)

    def forward(self, x):
        # x = self.FTA0(x, out)

        x = self.blk1(x)
        # x = self.FTA1(x)

        x = self.blk2(x)
        # x = self.FTA2(x)

        x = self.blk3(x)
        x = self.FTA3(x)

        x = self.AVG(x)
        return x


class Classifier_RESNET_FTA2(nn.Module):

    def __init__(self, input_shape, D=1, length=96):
        super(Classifier_RESNET_FTA2, self).__init__()
        self.out_shape = int(128 / D)
        self.blk1 = ResBlock1D(input_shape, int(64 / D))
        self.blk2 = ResBlock1D(int(64 / D), int(128 / D))
        self.blk3 = ResBlock1D(int(128 / D), int(128 / D))
        self.AVG = nn.AdaptiveAvgPool1d(1)
        self.FTA0 = FTABlock2(channel=length)
        self.conv1 = BasicBlock1D(inplanes=input_shape, planes=int(128 / D), kernel_size=7, padding=3)
        self.FTA1 = FTABlock(channel=length)
        self.FTA2 = FTABlock(channel=length)
        self.FTA3 = FTABlock(channel=length)
        # self.FTA4 = FTABlock(channel=length)

    def forward(self, x, out):
        x = self.FTA0(x, out)

        x = self.blk1(x)
        x = self.FTA1(x)

        x = self.blk2(x)
        x = self.FTA2(x)

        x = self.blk3(x)
        x = self.FTA3(x)

        x = self.AVG(x)
        return x
