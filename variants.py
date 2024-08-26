import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter


class FusionLayer(nn.Module):
    def __init__(self, num_views, fusion_type, in_size, hidden_size=32):
        super(FusionLayer, self).__init__()
        self.fusion_type = fusion_type
        if self.fusion_type == 'weight':
            self.weight = nn.Parameter(torch.ones(num_views) / num_views, requires_grad=True)
        if self.fusion_type == 'attention':
            self.encoder = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 32, bias=False),
                nn.Tanh(),
                nn.Linear(32, 1, bias=False)
            )

    def forward(self, emb_list):
        if self.fusion_type == "average":
            common_emb = sum([value for key, value in emb_list.items()]) / len(emb_list)

        elif self.fusion_type == "weight":
            weight = F.softmax(self.weight, dim=0)
            common_emb = sum([w * e for e, w in zip(weight, emb_list)])
        elif self.fusion_type == 'attention':
            emb_list=[value for key, value in emb_list.items()]
            emb_ = torch.stack(emb_list, dim=1)
            w = self.encoder(emb_)
            weight = torch.softmax(w, dim=1)
            common_emb = (weight * emb_).sum(1)
        else:
            sys.exit("Please using a correct fusion type")
        return common_emb

class MLP(torch.nn.Module):

    def __init__(self, num_i, num_h, num_o,device):
        super(MLP, self).__init__()

        self.linear1 = torch.nn.Linear(num_i, num_h)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_h, num_h)  # 2个隐层
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(num_h, num_o)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        # uniform(x,y) 在 [x, y] 范围内随机生成一个实数
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    # forward是向前传播函数，网络向前传播的方式为：relu– > fropout– > gc2– > softmax
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class variant_MLP(nn.Module):
    def __init__(self, nfeats, n_view,n_classes, para, device,fusion_type='attention'):
        super(variant_MLP, self).__init__()
        self.n_classes = n_classes
        self.device=device
        self.n_view=n_view
        self.nfeats=nfeats
        self.Blocks=nn.ModuleList([MLP(feat,32,n_classes,device) for feat in nfeats])
        self.theta = nn.Parameter(torch.FloatTensor([para]), requires_grad=True).to(device)
        self.ZZ_init = nn.ModuleList([nn.Linear(feat,n_classes).to(device) for feat in nfeats])
        self.fusionlayer = FusionLayer(n_view, fusion_type, self.n_classes, hidden_size=64)
        self.device=device

    def forward(self, features):

        Z ={}
        for view in range(0,self.n_view):
            Z[view]=self.Blocks[view]( features[view])
        output_z=self.fusionlayer(Z)
        return output_z

class variant_GCN(nn.Module):
    def __init__(self, nfeats, n_view, n_classes, blocks, para, device, fusion_type='attention'):
        super(variant_GCN, self).__init__()
        self.n_classes = n_classes
        self.blocks = blocks
        self.device = device
        self.n_view = n_view
        self.nfeats = nfeats
        self.Blocks = nn.ModuleList([GCN(feat, 32, n_classes, 0.5) for feat in nfeats])
        self.theta = nn.Parameter(torch.FloatTensor([para]), requires_grad=True).to(device)
        self.ZZ_init = nn.ModuleList([nn.Linear(feat, n_classes).to(device) for feat in nfeats])
        self.fusionlayer = FusionLayer(n_view, fusion_type, self.n_classes, hidden_size=64)
        self.device = device

    def forward(self, features):
        Z = {}
        for view in range(0, self.n_view):
            Z[view] = self.Blocks[view](features[view],)
        output_z = self.fusionlayer(Z)
        return output_z