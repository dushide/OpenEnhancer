import torch
import torch.nn.functional as F
import torch.nn as nn

class Block(nn.Module):
    def __init__(self,  out_features, nfea,device):
        super(Block, self).__init__()
        self.S_norm = nn.BatchNorm1d(out_features, momentum=0.6).to(device)
        self.S = nn.Linear(out_features, out_features).to(device)

        self.U_norm = nn.BatchNorm1d(nfea, momentum=0.6).to(device)
        self.U = nn.Linear(nfea, out_features).to(device)

        self.device = device
    def forward(self, input, view):
        input1 = self.S(self.S_norm(input))
        input2 = self.U(self.U_norm(view))
        output = input1 + input2
        return output
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
            common_emb = sum([w * emb_list[e] for w, e in zip(weight, emb_list.keys())])
        elif self.fusion_type == 'attention':
            emb_list=[value for key, value in emb_list.items()]
            emb_ = torch.stack(emb_list, dim=1)
            w = self.encoder(emb_)
            weight = torch.softmax(w, dim=1)
            common_emb = (weight * emb_).sum(1)
        else:
            sys.exit("Please using a correct fusion type")
        return common_emb

class DBONet(nn.Module):
    def __init__(self, nfeats, n_view,n_classes, blocks, para, device,fusion_type='attention'):
        super(DBONet, self).__init__()
        self.n_classes = n_classes
        self.blocks = blocks
        self.device=device
        self.n_view=n_view
        self.nfeats=nfeats
        self.Blocks=nn.ModuleList([Block(n_classes,feat,device) for feat in nfeats])
        self.theta = nn.Parameter(torch.FloatTensor([para]), requires_grad=True).to(device)
        self.ZZ_init = nn.ModuleList([nn.Linear(feat,n_classes).to(device) for feat in nfeats])
        self.fusionlayer = FusionLayer(n_view, fusion_type, self.n_classes, hidden_size=64)
        self.device=device
    def soft_threshold(self, u):
        return F.selu(u - self.theta) - F.selu(-1.0 * u - self.theta)
    def forward(self, features):

        output_z = 0
        for j in range(self.n_view):
            output_z += self.ZZ_init[j](features[j] / 1.0)

        output_z=output_z/self.n_view
        for i in range(0, self.blocks):
            Z ={}
            for view in range(0,self.n_view):
                Z[view]=self.soft_threshold(self.Blocks[view](output_z, features[view]))
            output_z=self.fusionlayer(Z)
        return output_z

