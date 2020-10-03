import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv


class GATZinc(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 num_atom_type,
                 num_bond_type):
        super(GATZinc, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.BNs = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.activation = activation
        self.num_atom_type = num_atom_type
        self.num_bond_type = num_bond_type
        
        # atom_type embedding
        self.embed = nn.Embedding(num_atom_type, in_dim)

        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, None))
        self.BNs.append(nn.BatchNorm1d(num_hidden * heads[0]))

        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
            self.BNs.append(nn.BatchNorm1d(num_hidden * heads[l]))

        hidden_dim = num_hidden * heads[-2]
        self.regressor1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.regressor2 = nn.Linear(hidden_dim//2, 1)

    def forward(self, x, e, snorm_n, snorm_e):
        h = self.embed(x)
        self.att_list = []
        for l in range(self.num_layers):
            h, att = self.gat_layers[l](self.g, h)
            h = h.flatten(1)
            self.att_list.append(att)
            h = h * snorm_n
            h = self.BNs[l](h)
            
            h = self.activation(h)
        
        self.g.ndata['h'] = h
        h = dgl.mean_nodes(self.g, 'h')
        h = self.activation(h)
        h = self.regressor1(h)
        h = torch.relu(h)
        logits = self.regressor2(h)

        return logits
    
    def get_factor(self):
        return self.att_list