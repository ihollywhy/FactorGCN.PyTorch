import torch
import torch.nn as nn
import torch.nn.functional as torch_fn
import dgl
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GraphConv
import matplotlib.pyplot as plt


class FactorGNNSBMs(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_latent,
                 feat_drop,
                 residual,
                 n_cls=2):
        super(FactorGNNSBMs, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.BNs = nn.ModuleList()
        self.feat_drop = feat_drop
        
        self.activate = torch.nn.LeakyReLU(negative_slope=0.2)
        self.embed = nn.Embedding(200, in_dim)
        
        self.layers.append(DisentangleLayer(num_latent, in_dim, num_hidden, cat=True))
        self.BNs.append(nn.BatchNorm1d(num_hidden))
        self.layers.append(DisentangleLayer(max(num_latent // 2, 1), num_hidden, num_hidden, cat=True))
        self.BNs.append(nn.BatchNorm1d(num_hidden))
        self.layers.append(DisentangleLayer(max(num_latent // 2, 1), num_hidden, num_hidden, cat=True))
        self.BNs.append(nn.BatchNorm1d(num_hidden))
        # self.layers.append(DisentangleLayer(max(num_latent // 2 // 2, 1), num_hidden, num_hidden, cat=True))
        # self.BNs.append(nn.BatchNorm1d(num_hidden))
        self.layers.append(None)
        self.BNs.append(None)

        self.BN1 = nn.BatchNorm1d(num_hidden)
        self.classifier1 = nn.Linear(num_hidden, num_hidden//2)
        self.classifier2 = nn.Linear(num_hidden//2, n_cls)

    def forward(self, x, e, snorm_n, snorm_e):
        feat = self.embed(x)
        for layer, bn in zip(self.layers[:-1], self.BNs[:-1]):
            feat_prim = feat
            feat = torch_fn.dropout(feat, self.feat_drop)
            feat = layer(self.g, feat)

            feat = feat * snorm_n
            feat = bn(feat)
            
            ## seem residual does not help
            # if feat.shape[1] == feat_prim.shape[1]:
            #     feat = feat + feat_prim

            feat = self.activate(feat)
            # feat = torch.tanh(feat)
        
        feat = torch_fn.dropout(feat, self.feat_drop)
        h = feat
        h = torch.relu(h)
        h = self.classifier1(h)
        h = torch.relu(h)
        h = self.classifier2(h)
        return h

    def get_factor(self):
        # return factor graph at each disentangle layer as list
        factor_list = []
        for layer in self.layers:
            if isinstance(layer,DisentangleLayer):
                factor_list.append(layer.get_factor())
        return factor_list

    def compute_disentangle_loss(self):
        # compute disentangle loss at each layer
        # return: list of loss
        loss_list = []
        for layer in self.layers:
            if isinstance(layer,DisentangleLayer):
                loss_list.append(layer.compute_disentangle_loss())
        return loss_list

    @staticmethod
    def merge_loss(list_loss):
        total_loss = 0
        for loss in list_loss:
            discrimination_loss, distribution_loss = loss[0], loss[1]
            total_loss += discrimination_loss
            # total_loss += distribution_loss
        return total_loss
        

class DisentangleLayer(nn.Module):
    def __init__(self, n_latent, in_dim, out_dim, cat=True):
        super(DisentangleLayer, self).__init__()
        # init self.g as None, after forward step, it will be replaced
        self.g = None 
        
        self.n_latent = n_latent
        self.n_feat_latent = out_dim // self.n_latent if cat else out_dim
        self.cat = cat

        self.linear = nn.Linear(in_dim, self.n_feat_latent)
        self.att_ls = nn.ModuleList()
        self.att_rs = nn.ModuleList()
        for latent_i in range(self.n_latent):
            self.att_ls.append(nn.Linear(self.n_feat_latent, 1))
            self.att_rs.append(nn.Linear(self.n_feat_latent, 1))

        # define for the additional losses
        self.graph_to_feat = GraphEncoder(self.n_feat_latent, self.n_feat_latent // 2)
        self.classifier = nn.Linear(self.n_feat_latent, self.n_latent)
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, g, inputs):
        self.g = g.local_var()
        out_feats = []
        hidden = self.linear(inputs)
        self.hidden = hidden
        for latent_i in range(self.n_latent):
            # compute factor features of nodes
            a_l = self.att_ls[latent_i](hidden)
            a_r = self.att_rs[latent_i](hidden)
            self.g.ndata.update({f'feat_{latent_i}'  : hidden, 
                            f'a_l_{latent_i}'   : a_l, 
                            f'a_r_{latent_i}'   : a_r})
            self.g.apply_edges(fn.u_add_v(f'a_l_{latent_i}', f'a_r_{latent_i}', f"factor_{latent_i}"))
            self.g.edata[f"factor_{latent_i}"] = torch.sigmoid( 6.0 * self.g.edata[f"factor_{latent_i}"] )
            feat = self.g.ndata[f'feat_{latent_i}']
            
            # graph conv on the factor graph
            norm = torch.pow(self.g.in_degrees().float().clamp(min=1), -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp).to(feat.device)
            feat = feat * norm
            
            # generate the output features
            self.g.ndata['h'] = feat
            self.g.update_all(fn.u_mul_e('h', f"factor_{latent_i}", 'm'),
                             fn.sum(msg='m', out='h'))
            out_feats.append(self.g.ndata['h'].unsqueeze(-1))
            
        if self.cat:
            return torch.cat(tuple([rst.squeeze(-1) for rst in out_feats]), -1)
        else:
            return torch.mean(torch.cat(tuple(out_feats), -1), -1)

    def compute_disentangle_loss(self):
        assert self.g is not None, "compute disentangle loss need to be called after forward pass"
        
        # compute discrimination loss
        factors_feat = [self.graph_to_feat(self.g, self.hidden, f"factor_{latent_i}").squeeze() 
                        for latent_i in range(self.n_latent)] 

        labels = [torch.ones(f.shape[0])*i for i, f in enumerate(factors_feat)]
        labels = torch.cat(tuple(labels), 0).long().cuda()
        factors_feat = torch.cat(tuple(factors_feat), 0)

        pred = self.classifier(factors_feat)
        discrimination_loss = self.loss_fn(pred, labels)
        
        # list_num_edges = torch.tensor(self.g.batch_num_edges).unsqueeze(1)
        latent_mean = [dgl.mean_edges(self.g, f"factor_{latent_i}") for latent_i in range(self.n_latent)]
        latent_mean = torch.cat(tuple(latent_mean), dim=1)
        # list_num_edges = list_num_edges.to(latent_sum.device)
        # norm_latent_sum = latent_sum / list_num_edges)
        
        latent_mean_distrib = torch_fn.softmax(latent_mean, dim=1)
        latent_mean_entropy = torch.sum(latent_mean_distrib * torch.log(latent_mean_distrib), dim=1)
        
        uniform = torch_fn.softmax(torch.ones_like(latent_mean), dim=1)
        upper_bound = torch.sum(uniform * torch.log(uniform), dim=1)
        
        distribution_loss = (latent_mean_entropy - upper_bound)
        distribution_loss = torch.mean(distribution_loss) * 100.0

        return [discrimination_loss, distribution_loss]

    def get_factor(self):
        g = self.g.local_var()
        return g


class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(GraphEncoder, self).__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, in_dim)

    def forward(self, g, inputs, factor_key):
        g = g.local_var()
        # graph conv on the factor graph
        feat = self.linear1(inputs)
        norm = torch.pow(g.in_degrees().float().clamp(min=1), -0.5)
        shp = norm.shape + (1,) * (feat.dim() - 1)
        norm = torch.reshape(norm, shp).to(feat.device)
        feat = feat * norm

        g.ndata['h'] = feat
        g.update_all(fn.u_mul_e('h', factor_key, 'm'),
                            fn.sum(msg='m', out='h'))
        g.ndata['h'] = torch.tanh( g.ndata['h'] )

        # graph conv on the factor graph
        feat = self.linear2(g.ndata['h'])
        feat = feat * norm

        g.ndata['h'] = feat
        g.update_all(fn.u_mul_e('h', factor_key, 'm'),
                            fn.sum(msg='m', out='h'))
        g.ndata['h'] = torch.tanh( g.ndata['h'] )
        
        h = dgl.mean_nodes(g, 'h').unsqueeze(-1)
        h = torch.tanh(h)
        
        return h
