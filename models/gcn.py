import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv


class GCNPool(nn.Module):
    def __init__(self,
                g,
                num_feats,
                num_layers,
                num_hidden,
                n_classes):
        super(GCNPool, self).__init__()
        self.g = g
        
        self.gcn_layers = nn.ModuleList()
        
        self.gcn_layers.append(GraphConv(num_feats, num_hidden))
        for i in range(num_layers):
            self.gcn_layers.append(GraphConv(num_hidden, num_hidden))

        self.classify = nn.Linear( num_hidden , n_classes)    
        
    def forward(self, inputs):
        h = inputs
        self.feat_list = []
        for layer in self.gcn_layers:
            h = layer(self.g, h)
            h = torch.tanh(h)
            self.feat_list.append(h.detach().cpu().numpy())
        
        self.g.ndata['h'] = h
        h = dgl.mean_nodes(self.g, 'h')
        h = torch.tanh(h)
        logits = self.classify(h)
        
        return logits
    
    def get_hidden_feature(self):
        return self.feat_list