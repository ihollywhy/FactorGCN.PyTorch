import torch
import dgl
from dgl.data import register_data_args, load_data
from dgl import DGLGraph
import networkx as nx
from .generate_synthetic import synthetic_graph_cls
from .zinc_dataset import MoleculeDatasetDGL
from .pattern_dataset import SBMsDatasetDGL
import numpy as np


def load_dataset(args=None, dataset=None):
    # if args is not provide, use dataset
    if args is not None:
        dataset_name = args.dataset.lower()
    else:
        dataset_name = dataset.lower()

    if dataset_name == 'synthetic_graph_cls':
        return load_synthetic_graph_cls(args)
    elif dataset_name == 'zinc':
        zinc_data = MoleculeDatasetDGL()
        return zinc_data
    elif 'sbms' in dataset_name:
        name = dataset_name.split('_')[-1]
        sbms_dataset = SBMsDatasetDGL(name)
        return sbms_dataset
    else:
        raise NameError(f'unknow dataset name {dataset_name}')


def load_synthetic_graph_cls(args):
    graph_generator = synthetic_graph_cls(args)
    samples = graph_generator.gen_union_graph()
    return samples
    