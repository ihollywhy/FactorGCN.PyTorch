#!/usr/bin/env python3
#
# This is an alternative implementation of our model,
#   inspired by https://github.com/rusty1s/pytorch_geometric.
# This newer version is likely more stable, efficient, and scalable.
#
# However, I haven't have the time to test this version yet.
# So no guarantee on this version's correctness.
#
# This is NOT the version for producing the results reported in the paper.
#
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import dgl
import torch.autograd
import torch.nn as nn
import torch.nn.functional as fn


class NeibRoutLayer(nn.Module):
    def __init__(self, num_caps, niter, tau=1.0):
        super(NeibRoutLayer, self).__init__()
        self.k = num_caps
        self.niter = niter
        self.tau = tau

    #
    # x \in R^{n \times d}: d-dimensional node representations.
    #    It can be node features, output of the previous layer,
    #    or even rows of the adjacency matrix.
    #
    # src_trg \in R^{2 \times m}: a list that contains m edges.
    #    src means the source nodes of the edges, and
    #    trg means the target nodes of the edges.
    #
    def forward(self, x, src_trg):
        m, src, trg = src_trg.shape[1], src_trg[0], src_trg[1]
        n, d = x.shape
        k, delta_d = self.k, d // self.k
        x = fn.normalize(x.view(n, k, delta_d), dim=2).view(n, d)
        z = x[src].view(m, k, delta_d)
        u = x
        scatter_idx = trg.view(m, 1).expand(m, d)
        for clus_iter in range(self.niter):
            p = (z * u[trg].view(m, k, delta_d)).sum(dim=2)
            p = fn.softmax(p / self.tau, dim=1)
            scatter_src = (z * p.view(m, k, 1)).view(m, d)
            u = torch.zeros(n, d, device=x.device)
            u.scatter_add_(0, scatter_idx, scatter_src)
            u += x
            # noinspection PyArgumentList
            u = fn.normalize(u.view(n, k, delta_d), dim=2).view(n, d)
        # p: m * k, m is #edges
        return u, p


class DisenGCN(nn.Module):
    #
    # nfeat: dimension of a node's input feature
    # nclass: the number of target classes
    # hyperpm: the hyper-parameter configuration
    #    ncaps: the number of capsules/channels/factors per layer
    #    routit: routing iterations
    #
    def __init__(self, nfeat, nclass, hyperpm, split_mlp=False):
        super(DisenGCN, self).__init__()
        self.pca = SparseInputLinear(nfeat, hyperpm.ncaps * hyperpm.nhidden)
        conv_ls = []
        for i in range(hyperpm.nlayer):
            conv = NeibRoutLayer(hyperpm.ncaps, hyperpm.routit)
            self.add_module('conv_%d' % i, conv)
            conv_ls.append(conv)
        self.conv_ls = conv_ls
        if split_mlp:
            self.clf = SplitMLP(nclass, hyperpm.nhidden * hyperpm.ncaps,
                                nclass)
        else:
            self.clf = nn.Linear(hyperpm.nhidden * hyperpm.ncaps, nclass)
        self.dropout = hyperpm.dropout

    def _dropout(self, x):
        return fn.dropout(x, self.dropout, training=self.training)

    def forward(self, x, src_trg):
        x = self._dropout(fn.leaky_relu(self.pca(x)))
        for conv in self.conv_ls:
            x = self._dropout(fn.leaky_relu(conv(x, src_trg)))
        x = self.clf(x)
        return x

class DisenGCNPool(nn.Module):
    #
    # nfeat: dimension of a node's input feature
    # nclass: the number of target classes
    # hyperpm: the hyper-parameter configuration
    #    ncaps: the number of capsules/channels/factors per layer
    #    routit: routing iterations
    #
    def __init__(self, nfeat, nclass, hyperpm, split_mlp=False):
        super(DisenGCNPool, self).__init__()
        self.g = None
        self.pca = SparseInputLinear(nfeat, hyperpm.ncaps * hyperpm.nhidden)
        conv_ls = []
        for i in range(hyperpm.nlayer):
            conv = NeibRoutLayer(hyperpm.ncaps, hyperpm.routit)
            self.add_module('conv_%d' % i, conv)
            conv_ls.append(conv)
        self.conv_ls = conv_ls
        if split_mlp:
            self.clf = SplitMLP(nclass, hyperpm.nhidden * hyperpm.ncaps,
                                nclass)
        else:
            self.clf = nn.Linear(hyperpm.nhidden * hyperpm.ncaps, nclass)
        self.dropout = hyperpm.dropout

    def _dropout(self, x):
        return fn.dropout(x, self.dropout, training=self.training)

    def forward(self, x):
        self.ps = []
        
        src_trg = self.g.all_edges()
        src_trg = tuple((s.unsqueeze(0).cuda() for s in src_trg))
        src_trg = torch.cat(src_trg, dim=0)
        x = self._dropout(fn.leaky_relu(self.pca(x)))
        for conv in self.conv_ls:
            x, p = conv(x, src_trg)
            x = self._dropout(fn.leaky_relu(x))
            self.ps.append(p.detach().cpu().numpy())

        self.g.ndata['x'] = x
        x = dgl.mean_nodes(self.g, 'x')
        x = fn.leaky_relu(x)
        
        x = self.clf(x)
        return x
    
    def get_factor(self):
        self.gs = []
        for p in self.ps:
            g = self.g.local_var()
            g.edata['a'] = p
            self.gs.append(g)
        return self.gs


class DisenGCNZinc(nn.Module):
    #
    # nfeat: dimension of a node's input feature
    # nclass: the number of target classes
    # hyperpm: the hyper-parameter configuration
    #    ncaps: the number of capsules/channels/factors per layer
    #    routit: routing iterations
    #
    def __init__(self, nfeat, nclass, hyperpm, split_mlp=False, 
                num_atom_type = 28, num_bond_type = 4):
        super(DisenGCNZinc, self).__init__()
        self.g = None
        self.num_atom_type = num_atom_type
        self.num_bond_type = num_bond_type
        self.BNs = nn.ModuleList()

        # atom_type embedding
        self.embed = nn.Embedding(num_atom_type, nfeat)

        self.pca = SparseInputLinear(nfeat, hyperpm.ncaps * hyperpm.nhidden)
        conv_ls = []
        for i in range(hyperpm.nlayer):
            if i >=2:
                conv = NeibRoutLayer(hyperpm.ncaps//2, hyperpm.routit)
            else:
                conv = NeibRoutLayer(hyperpm.ncaps, hyperpm.routit)
            self.BNs.append(nn.BatchNorm1d(hyperpm.ncaps * hyperpm.nhidden))
            self.add_module('conv_%d' % i, conv)
            conv_ls.append(conv)
        self.conv_ls = conv_ls
        if split_mlp:
            self.clf = SplitMLP(nclass, hyperpm.nhidden * hyperpm.ncaps,
                                nclass)
        else:
            # self.clf = nn.Linear(hyperpm.nhidden * hyperpm.ncaps, nclass)
            hidden_dim = hyperpm.nhidden * hyperpm.ncaps
            self.regressor1 = nn.Linear(hidden_dim, hidden_dim//2)
            self.regressor2 = nn.Linear(hidden_dim//2, 1)
        self.dropout = hyperpm.dropout

    def _dropout(self, x):
        return fn.dropout(x, self.dropout, training=self.training)

    def forward(self, x, e, snorm_n, snorm_e):
        x = self.embed(x)
        
        self.ps = []
        src_trg = self.g.all_edges()
        src_trg = tuple((s.unsqueeze(0).cuda() for s in src_trg))
        src_trg = torch.cat(src_trg, dim=0)
        x = self._dropout(fn.leaky_relu(self.pca(x)))
        for conv, bn in zip(self.conv_ls, self.BNs):
            x, p = conv(x, src_trg)
            x = x * snorm_n
            x = bn(x)
            x = self._dropout(fn.leaky_relu(x))
            self.ps.append(p.detach().cpu().numpy())

        self.g.ndata['x'] = x
        x = dgl.mean_nodes(self.g, 'x')
        x = fn.leaky_relu(x)
        
        x = self.regressor1(x)
        x = fn.leaky_relu(x)
        x = self.regressor2(x)
        return x
    
    def get_factor(self):
        self.gs = []
        for p in self.ps:
            g = self.g.local_var()
            g.edata['a'] = p
            self.gs.append(g)
        return self.gs


# noinspection PyUnresolvedReferences
class SparseInputLinear(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(SparseInputLinear, self).__init__()
        weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
        weight = nn.Parameter(torch.from_numpy(weight))
        bias = np.zeros(out_dim, dtype=np.float32)
        bias = nn.Parameter(torch.from_numpy(bias))
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight, bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.shape[1])
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):  # *nn.Linear* does not accept sparse *x*.
        return torch.mm(x, self.weight) + self.bias


class SplitMLP(nn.Module):
    def __init__(self, n_mlp, d_inp, d_out):
        super(SplitMLP, self).__init__()
        assert d_inp >= n_mlp and d_inp % n_mlp == 0
        assert d_out >= n_mlp and d_out % n_mlp == 0
        self.mlps = nn.Conv1d(in_channels=n_mlp, out_channels=d_out,
                              kernel_size=d_inp // n_mlp, groups=n_mlp)
        self.n_mlp = n_mlp

    def forward(self, x):
        n = x.shape[0]
        x = x.view(n, self.n_mlp, -1)
        x = self.mlps(x)
        x = x.view(n, -1)
        return x