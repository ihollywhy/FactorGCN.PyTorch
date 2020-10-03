import torch
import torch.nn as nn
import torch.nn.functional as torch_fn
import dgl


class MLP(nn.Module):
    def __init__(self,
                g,
                in_dim,
                num_layers,
                num_hidden,
                num_classes):
        super(MLP, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, num_hidden))
        for _ in range(num_layers):
            self.layers.append(nn.Linear(num_hidden, num_hidden))
        self.layers.append(nn.Linear(num_hidden, num_classes))
        
    def forward(self, inputs):
        h = inputs
        for layer in self.layers[:-1]:
            h = layer(h)
            h = torch.relu(h)
        return self.layers[-1](h)

class MLPZinc(nn.Module):
    def __init__(self,
                g,
                in_dim,
                num_layers,
                num_hidden,
                num_atom_type,
                num_bond_type):
        super(MLPZinc, self).__init__()
        self.g = g

        self.num_atom_type = num_atom_type
        self.num_bond_type = num_bond_type
        self.layers = nn.ModuleList()
        self.BNs = nn.ModuleList()

        # atom_type embedding
        self.embed = nn.Embedding(num_atom_type, in_dim)

        self.layers.append(nn.Linear(in_dim, num_hidden))
        self.BNs.append(nn.BatchNorm1d(num_hidden))
        for _ in range(num_layers):
            self.layers.append(nn.Linear(num_hidden, num_hidden))
            self.BNs.append(nn.BatchNorm1d(num_hidden))
        
        self.regressor1 = nn.Linear(num_hidden, num_hidden//2)
        self.regressor2 = nn.Linear(num_hidden//2, 1)

    def forward(self, x, e, snorm_n, snorm_e):
        h = self.embed(x)
        for layer, bn in zip(self.layers, self.BNs):
            h = layer(h)
            h = h * snorm_n
            h = bn(h)
            h = torch.relu(h)

        self.g.ndata['h'] = h
        h = dgl.mean_nodes(self.g, 'h')
        h = torch.relu(h)
        
        h = self.regressor1(h)
        h = torch.relu(h)
        logits = self.regressor2(h)

        return logits