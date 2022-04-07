import math
from bisect import bisect_left
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from TimeSeriesClass import BasicBlock1D, ResBlock1D
from TimeSeriesClass import FTABlock, FTABlockB
from inception import InceptionBlock

Records = {'FC_gama1': None, 'FC_gama2': None, 'Attn': None, 'Attn0': None}

class LearnedPositionalEmbedding2(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float().to(device='cuda')
        pe.require_grad = True
        pe = pe.unsqueeze(0)
        self.pe = nn.Parameter(pe)
        torch.nn.init.normal_(self.pe, std=0.02)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512, freq=64):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(freq) / d_model)).exp()
        if d_model % 2 == 1:
            div_term2 = div_term[:-1]
        else:
            div_term2 = div_term

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term2)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class BERTEmbedding2(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. PositionalEmbedding : adding positional information using sin, cos
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, input_dim, max_len, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.learnedPosition = LearnedPositionalEmbedding2(d_model=input_dim, max_len=max_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence):
        x = self.learnedPosition(sequence) + sequence
        return self.dropout(x)


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. PositionalEmbedding : adding positional information using sin, cos
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, input_dim, max_len, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.position = PositionalEmbedding(d_model=input_dim, max_len=max_len, freq=64)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence):
        x = self.position(sequence) + sequence
        return self.dropout(x)


class BERTEmbedding3(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. PositionalEmbedding : adding positional information using sin, cos
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, input_dim, max_len, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.position = PositionalEmbedding(d_model=input_dim, max_len=max_len, freq=64)
        self.lmbda = nn.Parameter(torch.tensor([0.5], dtype=torch.float), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence):
        x = self.lmbda * self.position(sequence) + sequence
        return self.dropout(x)


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        scores = torch.matmul(query, key.transpose(-2, -1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class Attention2(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, value, p_attn, dropout=None):
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttentionNeighbor(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1, Max_Len=64, pow=2, LrEnb=0, LrMo=0):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.pow = pow
        self.LrEnb = LrEnb
        self.LrMo = LrMo
        self.Max_Len = Max_Len
        self.pow = pow

        self.linear_layers1 = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])

        self.output_linear1 = nn.Linear(d_model, d_model)

        self.attention = Attention()
        self.attention2 = Attention2()

        self.dropout = nn.Dropout(p=dropout)
        self.lam1 = nn.Parameter(torch.tensor([0], dtype=torch.float), requires_grad=True)
        # self.lam2 = nn.Parameter(torch.tensor([0], dtype=torch.float), requires_grad=True)
        t = torch.arange(0, Max_Len, dtype=torch.float)
        t1 = t.repeat(Max_Len, 1)
        t2 = t1.permute([1, 0])

        if pow == 2:
            dis1 = torch.exp(-1 * torch.pow((t2 - t1), 2) / 2)
            self.dist = nn.Parameter(-1 * torch.pow((t2 - t1), 2) / 2, requires_grad=False)
        else:
            dis1 = torch.exp(-1 * torch.abs((t2 - t1) / 1))
            self.dist = nn.Parameter(-1 * torch.abs((t2 - t1) / 1), requires_grad=False)
        if LrEnb:
            self.adj1 = nn.Parameter(dis1)
            if LrMo == 1:
                self.FCgamma = nn.Linear(self.d_k, 1)
                torch.nn.init.xavier_uniform_(self.FCgamma.weight)
                self.FCgamma.bias.data.fill_(0.05)
            elif LrMo == 2:
                self.FCgamma = nn.Linear(self.d_k, 2)
                torch.nn.init.xavier_uniform_(self.FCgamma.weight)
                self.FCgamma.bias.data.fill_(0.05)
            else:
                self.gamma = nn.Parameter(torch.ones(Max_Len, dtype=torch.float), requires_grad=True)
                # gamma=torch.diag(torch.exp(1/torch.pow(gamma,2)))


        else:

            self.adj1 = nn.Parameter(torch.unsqueeze(torch.unsqueeze(dis1, dim=0), dim=0),
                                     requires_grad=False)

    def normalize(self, A, symmetric=True):
        # A = A+I
        # A = A + torch.eye(A.size(0))
        d = A.sum(-1) + 1e-9
        if symmetric:
            # D = D^-1/2
            D = torch.diag(torch.pow(d, -0.5))
            return D.mm(A).mm(D)
        else:
            # D=D^-1
            if len(A.shape) == 2:
                D = torch.diag(torch.pow(d, -1))
                return D.mm(A)
            else:
                D = torch.diag_embed(torch.pow(d, -1))
                return torch.matmul(D, A)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers1, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        _, attn = self.attention(query, key, value, mask=mask)
        if self.LrEnb == 0:
            adj = self.adj1
        else:
            if self.LrMo == 0:
                gamma = torch.diag(torch.pow(torch.pow(torch.abs(self.gamma), self.pow) + 1, -1))
                gau = torch.exp(gamma.mm(self.dist))
                # adj = gamma.mm(self.adj1)
                # adj = self.normalize(gau, symmetric=False).unsqueeze(0).unsqueeze(0)
                adj = gau.unsqueeze(0).unsqueeze(0)
                # attn += adj

            elif self.LrMo == 1:
                query = self.FCgamma(query).contiguous().view(batch_size * self.h, -1)
                # query = torch.pow(torch.pow(torch.abs(query), self.pow) + 1, -1)
                query = torch.pow(torch.pow(torch.abs(query) + 1, self.pow),
                                  -1)  #############################################################removed  +1
                query = torch.diag_embed(query)
                adj = torch.exp(torch.matmul(query, self.dist))
                # adj=self.normalize(adj, symmetric=False)
                adj = adj.contiguous().view(batch_size, self.h, self.Max_Len, self.Max_Len)
                query_o = 0
            elif self.LrMo == 2:
                query_o = self.FCgamma(query).contiguous().view(batch_size * self.h, self.Max_Len, -1)
                # query = torch.pow(torch.pow(torch.abs(query), self.pow) + 1, -1)
                query = torch.pow(torch.pow(torch.abs(query_o) + 1, self.pow),
                                  -1)  #############################################################removed  +1
                queryu = torch.diag_embed(query[..., 0])
                queryl = torch.diag_embed(query[..., 1])
                adj = torch.exp(torch.matmul(queryl, self.dist.tril(diagonal=-1)) + torch.matmul(queryu, self.dist.triu(
                    diagonal=0)))
                # adj=self.normalize(adj, symmetric=False)
                adj = adj.contiguous().view(batch_size, self.h, self.Max_Len, self.Max_Len)
            elif self.LrMo == 3:
                adj = 0
                query_o = 0
        if torch.is_tensor(query_o):
            Records['FC_gama1'] = query_o[..., 0]
            Records['FC_gama2'] = query_o[..., 1]
            # attn+=adj
        # attn= (torch.sigmoid(self.lam1)*attn+torch.sigmoid(self.lam2)*adj)/(torch.sigmoid(self.lam1)+torch.sigmoid(self.lam2))
        attn = attn / torch.max(attn, dim=-1)[0].unsqueeze(-1).expand_as(attn)
        Records['Attn0'] = attn
        # attn+=adj
        attn = (torch.sigmoid(self.lam1) * attn + (1 - torch.sigmoid(self.lam1)) * adj)
        attn = self.normalize(attn, symmetric=False)
        Records['Attn'] = attn
        output, _ = self.attention2(value, attn, self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear1(output)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class PositionwiseFeedForward0(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward0, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        # self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.dropout(self.activation(self.w_1(x)))


class TSCTransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, Max_Len, pow, LrEnb, LrMo):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttentionNeighbor(h=attn_heads, d_model=hidden, Max_Len=Max_Len, pow=pow,
                                                      LrEnb=LrEnb, LrMo=LrMo)
        if feed_forward_hidden == 0:
            self.feed_forward = PositionwiseFeedForward0(d_model=hidden, d_ff=hidden, dropout=dropout)
        else:
            self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class TSCBERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, input_dim, max_len, hidden=768, n_layers=12, attn_heads=12, dropout=0.4, mask_prob=0.8,
                 device='cuda', pow=2, LrEnb=1, LrMo=1, pos_enc=1, ffh=-4):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.max_len = max_len
        self.input_dim = input_dim
        self.mask_prob = mask_prob
        self.device = device
        # clsToken = torch.zeros(1, 1, self.input_dim).float().cuda()
        # clsToken.require_grad = True
        # self.clsToken = nn.Parameter(clsToken)
        # torch.nn.init.normal_(clsToken, std=0.02)
        if pos_enc == 1:
            self.embedding = BERTEmbedding(input_dim=input_dim, max_len=max_len)
        elif pos_enc == 2:
            self.embedding = BERTEmbedding2(input_dim=input_dim, max_len=max_len)
        elif pos_enc == 3:
            self.embedding = BERTEmbedding3(input_dim=input_dim, max_len=max_len)

        else:
            self.embedding = None
        # paper noted they used 4*hidden_size for ff_network_hidden_size
        if ffh == 4:
            self.feed_forward_hidden = hidden * 4
        elif ffh == 2:
            self.feed_forward_hidden = hidden * 2
        elif ffh == -2:
            self.feed_forward_hidden = hidden // 2
        elif ffh == -4:
            self.feed_forward_hidden = hidden // 4
        elif ffh == 0:
            self.feed_forward_hidden = 0

        self.transformer_blocks = nn.ModuleList(
            [TSCTransformerBlock(hidden, attn_heads, self.feed_forward_hidden, dropout, max_len, pow, LrEnb, LrMo) for _
             in range(n_layers)])

    def forward(self, input_vectors):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        batch_size = input_vectors.shape[0]
        sample = None
        if self.training:
            bernolliMatrix = (torch.tensor([self.mask_prob]).float().repeat(self.max_len)).unsqueeze(0).repeat(
                [batch_size, 1]).to(self.device)
            self.bernolliDistributor = torch.distributions.Bernoulli(bernolliMatrix)
            sample = self.bernolliDistributor.sample()
            mask = (sample > 0).unsqueeze(1).repeat(1, sample.size(1), 1).unsqueeze(1)
        else:
            mask = torch.ones(batch_size, 1, self.max_len, self.max_len).to(self.device)

        # embedding the indexed sequence to sequence of vectors
        # x = torch.cat((self.clsToken.repeat(batch_size, 1, 1), input_vectors), 1)
        x = input_vectors
        if self.embedding is not None:
            x = self.embedding(x)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x, sample


class Classifier_TSCBERT(nn.Module):

    def __init__(self, input_shape, hidden_size=512, D=1, length=24, device='cuda', pow=2, LrEnb=0, LrMo=0, adaD=0,
                 adaH=1, n_layers=1, pos_enc=1, ffh=-4):
        super(Classifier_TSCBERT, self).__init__()
        if adaD == 1:
            if input_shape >= hidden_size:
                # if hidden_size>=128:
                self.embedding = nn.Sequential(
                    nn.Linear(input_shape, input_shape // 2),
                    nn.ReLU(),
                    nn.Linear(input_shape // 2, hidden_size),
                    nn.ReLU(),
                )
                self.hidden_size = hidden_size
                self.adaD = 1
            else:
                self.embedding = nn.Sequential(
                    nn.Linear(input_shape, input_shape * 2),
                    nn.ReLU(),
                    nn.Linear(input_shape * 2, hidden_size),
                    nn.ReLU(),
                )
                self.hidden_size = hidden_size
                self.adaD = 1
        elif adaD == 2:
            # if input_shape >= hidden_size:

            self.embedding = BasicBlock1D(inplanes=input_shape, planes=hidden_size, kernel_size=7, padding=3)
            self.hidden_size = hidden_size
            self.adaD = 2

        elif adaD == 3:
            # if input_shape >= hidden_size:

            self.embedding = nn.Sequential(
                BasicBlock1D(inplanes=input_shape, planes=hidden_size, kernel_size=7, padding=3),
                BasicBlock1D(inplanes=hidden_size, planes=hidden_size * 2, kernel_size=5, padding=2),
                BasicBlock1D(inplanes=hidden_size * 2, planes=hidden_size, kernel_size=3, padding=1))
            self.hidden_size = hidden_size
            self.adaD = 2
        elif adaD == 4:
            # if input_shape >= hidden_size:

            self.embedding = nn.Sequential(ResBlock1D(input_shape, hidden_size),
                                           ResBlock1D(hidden_size, hidden_size * 2),
                                           ResBlock1D(hidden_size * 2, hidden_size * 2))
            self.hidden_size = hidden_size * 2
            self.adaD = 2
            # else:
            #    self.adaD = 0
            #    self.hidden_size = input_shape

        else:
            self.adaD = 0
            self.hidden_size = input_shape
        # self.adaD=adaD
        self.adaH = adaH
        if adaH == 1:
            mylist = self.factors(self.hidden_size)
            h = self.take_closest(mylist, 8)
            self.attn_heads = self.hidden_size // h
        else:
            self.attn_heads = 1
        self.length = length
        self.n_layers = n_layers
        self.bert = TSCBERT(self.hidden_size, self.length, hidden=self.hidden_size, n_layers=self.n_layers,
                            attn_heads=self.attn_heads, device=device, pow=pow, LrEnb=LrEnb, LrMo=LrMo, pos_enc=pos_enc,
                            ffh=ffh)
        self.out_shape = self.hidden_size
        self.avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):

        x = x.permute(0, 2, 1)
        if self.adaD == 1:
            x = self.embedding(x)
        elif self.adaD == 2:
            x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        input_vectors = x
        norm = input_vectors.norm(p=2, dim=-1, keepdim=True)
        input_vectors = input_vectors.div(norm)
        output, maskSample = self.bert(input_vectors)
        # classificationOut = output[:, 0, :]

        # sequenceOut = output[:, 1:, :]
        # norm = sequenceOut.norm(p=2, dim=-1, keepdim=True)
        # sequenceOut = sequenceOut.div(norm)
        # output = self.dp(classificationOut)
        # x = self.fc_action(output)
        clssificationOut = self.avg(output.transpose(1, 2))

        return clssificationOut

    def factors(self, n):
        return reduce(list.__add__,
                      ([n // i] for i in range(int(n ** 0.5), 0, -1) if n % i == 0))

    def take_closest(self, myList, myNumber):
        """
        Assumes myList is sorted. Returns closest value to myNumber.

        If two numbers are equally close, return the smallest number.
        """
        pos = bisect_left(myList, myNumber)
        if pos == 0:
            return myList[0]
        if pos == len(myList):
            return myList[-1]
        before = myList[pos - 1]
        after = myList[pos]
        if after - myNumber < myNumber - before:
            return after
        else:
            return before


class Classifier_FCN_MHAne(nn.Module):

    def __init__(self, input_shape, D=1, length=24, device='cuda', pow=2, LrEnb=0, LrMo=0, pos_enc=1, ffh=4):
        super(Classifier_FCN_MHAne, self).__init__()

        self.input_shape = input_shape
        self.out_shape = int(128 / D)
        self.mask_prob = 0.5
        self.max_len = length
        self.device = device
        if pos_enc == 1:
            self.embedding = BERTEmbedding(input_dim=input_shape, max_len=length)
        elif pos_enc == 2:
            self.embedding = BERTEmbedding2(input_dim=input_shape, max_len=length)
        elif pos_enc == 3:
            self.embedding = BERTEmbedding3(input_dim=input_shape, max_len=length)
        elif pos_enc == 0:
            self.embedding = None

        self.conv1 = BasicBlock1D(inplanes=input_shape, planes=int(128 / D), kernel_size=7, padding=3)
        self.FTA1 = TSCTransformerBlock(int(128 / D), 1, int(128 / D) * ffh, 0.4, length, pow, LrEnb, LrMo)

        self.conv2 = BasicBlock1D(inplanes=int(128 / D), planes=int(256 / D), kernel_size=5, padding=2)
        self.FTA2 = TSCTransformerBlock(int(256 / D), 1, int(256 / D) * ffh, 0.4, length, pow, LrEnb, LrMo)

        self.conv3 = BasicBlock1D(inplanes=int(256 / D), planes=int(128 / D), kernel_size=3, padding=1)
        self.FTA3 = TSCTransformerBlock(int(128 / D), 1, int(128 / D) * ffh, 0.4, length, pow, LrEnb, LrMo)

        self.AVG = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        batch_size = x.shape[0]
        sample = None
        if self.training:
            bernolliMatrix = (torch.tensor([self.mask_prob]).float().repeat(self.max_len)).unsqueeze(0).repeat(
                [batch_size, 1]).to(self.device)
            self.bernolliDistributor = torch.distributions.Bernoulli(bernolliMatrix)
            sample = self.bernolliDistributor.sample()
            mask = (sample > 0).unsqueeze(1).repeat(1, sample.size(1), 1).unsqueeze(1)
        else:
            mask = torch.ones(batch_size, 1, self.max_len, self.max_len).to(self.device)
        if self.embedding is not None:
            x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.conv1(x)
        x = self.FTA1(x.permute(0, 2, 1), mask).permute(0, 2, 1)
        x = self.conv2(x)
        x = self.FTA2(x.permute(0, 2, 1), mask).permute(0, 2, 1)
        x = self.conv3(x)
        x = self.FTA3(x.permute(0, 2, 1), mask).permute(0, 2, 1)
        x = self.AVG(x)
        return x


class Inception_TBO(InceptionBlock):
    def __init__(self, in_channels, type, length, device='cuda', pow=2, LrEnb=0, LrMo=0, pos_enc=1, ffh=4, n_filters=32,
                 **kwargs):
        super(Inception_TBO, self).__init__(in_channels, n_filters, **kwargs)
        self.device = device
        if pos_enc == 1:
            self.embedding = BERTEmbedding(input_dim=in_channels if not type == 8 else 4 * n_filters, max_len=length)
        elif pos_enc == 2:
            self.embedding = BERTEmbedding2(input_dim=in_channels if not type == 8 else 4 * n_filters, max_len=length)
        elif pos_enc == 3:
            self.embedding = BERTEmbedding3(input_dim=in_channels if not type == 8 else 4 * n_filters, max_len=length)
        elif pos_enc == 0:
            self.embedding = None

        if type == 1 or type == 4:
            self.FTA1 = FTABlock(channel=length, reduction=ffh) if type == 1 else None
            self.FTA2 = FTABlock(channel=length, reduction=ffh) if type == 1 else None
            self.FTA3 = FTABlock(channel=length, reduction=ffh)
            self.type = 25 if type == 1 else 5
        elif type == 2 or type == 5:
            self.FTA1 = FTABlockB(channel=4 * n_filters, length=length, reduction=ffh) if type == 2 else None
            self.FTA2 = FTABlockB(channel=4 * n_filters, length=length, reduction=ffh) if type == 2 else None
            self.FTA3 = FTABlockB(channel=4 * n_filters, length=length, reduction=ffh)
            self.type = 25 if type == 2 else 5
        elif type == 3 or type == 6 or type == 8:
            self.FTA1 = TSCTransformerBlock(4 * n_filters, 1, 4 * n_filters * ffh, 0.4, length, pow, LrEnb,
                                            LrMo) if type == 3 else None
            self.FTA2 = TSCTransformerBlock(4 * n_filters, 1, 4 * n_filters * ffh, 0.4, length, pow, LrEnb,
                                            LrMo) if type == 3 else None
            self.FTA3 = TSCTransformerBlock(4 * n_filters, 1, 4 * n_filters * ffh, 0.4, length, pow, LrEnb, LrMo)
            self.type = 2 if type == 3 else 4
            self.type = 8 if type == 8 else self.type

        # elif type==4:
        #     self.FTA3 = TSCTransformerBlock(4 * n_filters, 1, 4 * n_filters * ffh, 0.4, length, pow, LrEnb, LrMo)
        #     self.type = 4
        else:
            self.type = 7
        self.AVG = nn.AdaptiveAvgPool1d(1)
        self.out_shape = 4 * n_filters
        self.mask_prob = 0.5
        self.max_len = length

    def forward(self, X):
        if not (self.type == 7 or self.type == 8):
            if self.embedding is not None:
                Z = self.embedding(X.transpose(1, 2)).transpose(1, 2)
        if self.type % 2 == 0:
            batch_size = X.shape[0]
            sample = None
            if self.training:
                bernolliMatrix = (torch.tensor([self.mask_prob]).float().repeat(self.max_len)).unsqueeze(0).repeat(
                    [batch_size, 1]).to(self.device)
                self.bernolliDistributor = torch.distributions.Bernoulli(bernolliMatrix)
                sample = self.bernolliDistributor.sample()
                mask = (sample > 0).unsqueeze(1).repeat(1, sample.size(1), 1).unsqueeze(1)
            else:
                mask = torch.ones(batch_size, 1, self.max_len, self.max_len).to(self.device)

        if self.return_indices:
            Z, i1 = self.inception_1(X if self.type == 0 else Z)
            if self.type % 25 == 0:
                Z = self.FTA1(Z)
            elif self.type == 2:
                Z = self.FTA1(Z.permute(0, 2, 1), mask).permute(0, 2, 1)

            Z, i2 = self.inception_2(Z)
            if self.type % 25 == 0:
                Z = self.FTA2(Z)
            elif self.type == 2:
                Z = self.FTA2(Z.permute(0, 2, 1), mask).permute(0, 2, 1)

            Z, i3 = self.inception_3(Z)
            if self.type % 5 == 0:
                Z = self.FTA3(Z)
            elif self.type == 2 or self.type == 4:
                Z = self.FTA3(Z.permute(0, 2, 1), mask).permute(0, 2, 1)
        else:
            Z = self.inception_1(X)
            if self.type % 25 == 0:
                Z = self.FTA2(Z)
            elif self.type == 2:
                Z = self.FTA2(Z.permute(0, 2, 1), mask).permute(0, 2, 1)
            Z = self.inception_2(Z)
            if self.type % 25 == 0:
                Z = self.FTA2(Z)
            elif self.type == 2:
                Z = self.FTA2(Z.permute(0, 2, 1), mask).permute(0, 2, 1)
            Z = self.inception_3(Z)
            if self.type % 5 == 0:
                Z = self.FTA3(Z)
            elif self.type == 2 or self.type == 4:
                Z = self.FTA3(Z.permute(0, 2, 1), mask).permute(0, 2, 1)
        if self.use_residual:
            Z = Z + self.residual(X)
            Z = self.activation(Z)
        if self.type % 8 == 0:
            if self.embedding is not None:
                Z = self.embedding(Z.transpose(1, 2)).transpose(1, 2)
            Z = self.FTA3(Z.permute(0, 2, 1), mask).permute(0, 2, 1)

        Z = self.AVG(Z)
        if self.return_indices:
            return Z, [i1, i2, i3]
        else:
            return Z
