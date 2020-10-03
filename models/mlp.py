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

class MLPPool(nn.Module):
    def __init__(self,
                g,
                in_dim,
                num_layers,
                num_hidden,
                num_classes):
        super(MLPPool, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, num_hidden))
        
        for _ in range(num_layers):
            self.layers.append(nn.Linear(num_hidden, num_hidden))
        self.layers.append(nn.Linear(num_hidden, num_classes))
        self.classify = nn.Linear(num_hidden, num_classes)

    def forward(self, inputs):
        h = inputs
        self.feat_list = []
        for layer in self.layers[:-1]:
            h = layer(h)
            h = torch.relu(h)
            self.feat_list.append(h.detach().cpu().numpy())
            
        self.g.ndata['h'] = h
        h = dgl.mean_nodes(self.g, 'h')
        h = torch.relu(h)
        logits = self.classify(h)
        return logits
    
    def get_hidden_feature(self):
        return self.feat_list