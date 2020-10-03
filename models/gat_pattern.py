import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv


class GATSBMs(nn.Module):
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
                 n_cls=2):
        super(GATSBMs, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.BNs = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.activation = activation
        
        # atom_type embedding
        self.embed = nn.Embedding(200, in_dim)

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
        self.classifier1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.classifier2 = nn.Linear(hidden_dim//2, n_cls)

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
        
        h = self.classifier1(h)
        h = torch.relu(h)
        logits = self.classifier2(h)

        return logits
    
    def get_factor(self):
        return self.att_list