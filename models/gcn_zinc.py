import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv


class GCNZinc(nn.Module):
    def __init__(self,
                g,
                num_feats,
                num_layers,
                num_hidden,
                num_atom_type,
                num_bond_type):
        super(GCNZinc, self).__init__()
        self.g = g
        
        self.num_atom_type = num_atom_type
        self.num_bond_type = num_bond_type
        self.gcn_layers = nn.ModuleList()
        self.BNs = nn.ModuleList()
        
        # atom_type embedding
        self.embed = nn.Embedding(num_atom_type, num_feats)

        self.gcn_layers.append(GraphConv(num_feats, num_hidden))
        self.BNs.append(nn.BatchNorm1d(num_hidden))
        
        for i in range(num_layers):
            self.gcn_layers.append(GraphConv(num_hidden, num_hidden))
            self.BNs.append(nn.BatchNorm1d(num_hidden))

        self.regressor1 = nn.Linear(num_hidden, num_hidden//2)
        self.regressor2 = nn.Linear(num_hidden//2, 1)

    def forward(self, x, e, snorm_n, snorm_e):
        h = self.embed(x)
        for layer, bn in zip(self.gcn_layers, self.BNs):
            h = layer(self.g, h)
            h = h * snorm_n
            h = bn(h)
            h = torch.tanh(h)
        
        self.g.ndata['h'] = h
        h = dgl.mean_nodes(self.g, 'h')
        h = torch.relu(h)
        
        h = self.regressor1(h)
        h = torch.relu(h)
        logits = self.regressor2(h)

        return logits