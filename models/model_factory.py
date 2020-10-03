import torch.nn.functional as F
import torch
from .factorgnn import FactorGNN, FactorGNNPool
from .factorgnn_zinc import FactorGNNZinc
from .factorgnn_pattern import FactorGNNSBMs
from .mlp import MLP, MLPPool
from .mlp_zinc import MLPZinc
from .gat import GAT, GATPool
from .gat_zinc import GATZinc
from .gat_pattern import GATSBMs
from .gcn import GCNPool
from .gcn_zinc import GCNZinc
from .disengcn import DisenGCN, DisenGCNPool, DisenGCNZinc, DisenGCNSBMs
import numpy as np


def get_model(dataset, args, mode = "multiclass"):
    if mode == "multiclass":
        model = get_model_multiclass(dataset, args)
    elif mode == "multilabel":
        model = get_model_multilabel(dataset, args)
    elif mode == "zinc":
        model = get_zinc_model(dataset, args)
    elif mode == 'sbms':
        model = get_sbms_model(dataset, args)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return model

def get_zinc_model(dataset, args):
    g, features, labels, train_mask, val_mask, test_mask, factor_graphs = dataset
    if args.model_name == 'GAT':
        heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        model = GATZinc(g, args.num_layers, args.in_dim, args.num_hidden,
                        heads, F.elu, args.in_drop, args.attn_drop, args.negative_slope,
                        args.residual, num_atom_type = 28, num_bond_type = 4)
    
    elif args.model_name == 'FactorGNN':
        model = FactorGNNZinc(g, args.num_layers, args.in_dim, args.num_hidden,
                            args.num_latent, args.in_drop, args.residual,
                            num_atom_type = 28, num_bond_type = 4)
    
    elif args.model_name == "MLP":
        model = MLPZinc(g, args.in_dim, args.num_layers, args.num_hidden,
                        num_atom_type = 28, num_bond_type = 4)
    elif args.model_name == "GCN":
        model = GCNZinc(g, args.in_dim, args.num_layers, args.num_hidden,
                        num_atom_type = 28, num_bond_type = 4)
    elif args.model_name == "DisenGCN":
        model = DisenGCNZinc(args.in_dim, 1,
                            args, split_mlp=False,
                            num_atom_type = 28, num_bond_type = 4)
    else:
        raise NameError(f'unknow format of model name: {args.model_name}')
        
    return model


def get_model_multilabel(dataset, args):
    g, features, labels, train_mask, val_mask, test_mask, factor_graphs = dataset

    num_feats = features.shape[1]
    n_classes = labels.shape[1]
    pooling = True if features.shape[0] != labels.shape[0] else False
    
    if args.model_name == "FactorGNN":
        if pooling:
            model = FactorGNNPool(g, args.num_layers, num_feats, args.num_hidden, 
                                n_classes, args.num_latent, args.in_drop, args.residual)
        else:
            model = FactorGNN(g, args.num_layers, num_feats, args.num_hidden,
                            n_classes, args.num_latent, args.in_drop, args.residual)

    elif args.model_name == "GAT":
        # create model
        heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        if pooling:
            model = GATPool(g, args.num_layers, num_feats, args.num_hidden,
                        n_classes, heads, F.elu, args.in_drop, args.attn_drop,
                        args.negative_slope, args.residual)
        else:
            model = GAT(g, args.num_layers, num_feats, args.num_hidden,
                        n_classes, heads, F.elu, args.in_drop,
                        args.attn_drop, args.negative_slope, args.residual)
    elif args.model_name == 'MLP':
        if pooling:
            model = MLPPool(g, num_feats, args.num_layers,
                        args.num_hidden, n_classes)
        else:
            model = MLP(g, num_feats, args.num_layers,
                        args.num_hidden, n_classes)
    elif args.model_name == "GCN":
        if pooling:
            model = GCNPool(g, num_feats, args.num_layers,
                            args.num_hidden, n_classes)
    elif args.model_name == "DisenGCN":
        if pooling:
            model = DisenGCNPool(num_feats, n_classes, 
                            args, split_mlp=False)
    else:
        raise NameError(f'unknow format of model name: {args.model_name}')

    return model


def get_model_multiclass(dataset, args):
    
    g, features, labels, train_mask, val_mask, test_mask, factor_graph = dataset

    num_feats = features.shape[1]
    n_classes = torch.max(labels).item() + 1
    
    if args.model_name == "FactorGNN":
        model = FactorGNN(g,
                 args.num_layers,
                 num_feats,
                 args.num_hidden,
                 n_classes,
                 args.num_latent,
                 args.in_drop,
                 args.residual)

    elif args.model_name == "GAT":
        # create model
        heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        model = GAT(g,
                    args.num_layers,
                    num_feats,
                    args.num_hidden,
                    n_classes,
                    heads,
                    F.elu,
                    args.in_drop,
                    args.attn_drop,
                    args.negative_slope,
                    args.residual)
    else:
        raise ValueError(f"unknow model name: {args.model_name}")
    return model


def get_sbms_model(dataset, args):
    g, features, labels, train_mask, val_mask, test_mask, factor_graphs = dataset
    n_classes = 2
    if args.model_name == 'FactorGNN':
        model = FactorGNNSBMs(g, args.num_layers, args.in_dim, args.num_hidden,
                            args.num_latent, args.in_drop, args.residual, n_classes)
    elif args.model_name == 'GAT':
        heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        model = GATSBMs(g, args.num_layers, args.in_dim, args.num_hidden,
                        heads, F.elu, args.in_drop, args.attn_drop, args.negative_slope,
                        args.residual)
    elif args.model_name == 'DisenGCN':
        model = DisenGCNSBMs(args.in_dim, 1,
                            args, split_mlp=False)
    else:
        raise NameError(f'unknow format of model name: {args.model_name}')
    
    return model