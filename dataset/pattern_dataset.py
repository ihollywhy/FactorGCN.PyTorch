
import time
import os
import pickle
import numpy as np

import dgl
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sbms_dir = os.path.join(cur_dir, '..', "data/sbms/")

class load_SBMsDataSetDGL(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 name,
                 split):

        self.split = split
        self.is_test = split.lower() in ['test', 'val'] 
        with open(os.path.join(data_dir, name+f'_{split}_list_v2.pkl'), 'rb') as f:
            self.dataset = pickle.load(f)
        
        self.node_labels = []
        self.graph_lists = []
        self.n_samples = len(self.dataset)
        self._prepare()
    
    def _prepare(self):
        self.graph_lists = [self.dataset[i][0] for i in range(self.n_samples)]
        self.node_labels = [self.dataset[i][1] for i in range(self.n_samples)]

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.node_labels[idx]


class SBMsDatasetDGL(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            TODO
        """
        start = time.time()
        print("[I] Loading data ...")
        self.name = name
        data_dir = sbms_dir
        self.train = load_SBMsDataSetDGL(data_dir, name, split='train')
        self.test = load_SBMsDataSetDGL(data_dir, name, split='test')
        self.val = load_SBMsDataSetDGL(data_dir, name, split='val')
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.cat(labels).long()
        tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        snorm_n = torch.cat(tab_snorm_n).sqrt()  
        tab_sizes_e = [ graphs[i].number_of_edges() for i in range(len(graphs))]
        tab_snorm_e = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e ]
        snorm_e = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels, snorm_n, snorm_e


def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']
        
        
        This function is called inside a function in SBMsDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']
    
    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)
    
    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g


class SBMsDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading SBM datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        data_dir = sbms_dir
        with open(os.path.join(data_dir, name+'.pkl'),"rb") as f:
            f = pickle.load(f)
            self.train = f[0]
            self.val = f[1]
            self.test = f[2]
        print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.cat(labels).long()
        tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        snorm_n = torch.cat(tab_snorm_n).sqrt()  
        tab_sizes_e = [ graphs[i].number_of_edges() for i in range(len(graphs))]
        tab_snorm_e = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e ]
        snorm_e = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels, snorm_n, snorm_e

    def _add_self_loops(self):
        
        # function for adding self loops
        # this function will be called only if self_loop flag is True
            
        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]


if __name__ == "__main__":
    sbms_dataset = SBMsDatasetDGL('cluster')
    from torch.utils.data import DataLoader
    sbms_dataloader = DataLoader(sbms_dataset.train, batch_size=100, shuffle=True,
            collate_fn=sbms_dataset.collate, num_workers=4)

    for i, item in enumerate(sbms_dataloader):
        batch_graphs, batch_targets, batch_snorm_n, batch_snorm_e = item
        print(batch_graphs)
        print(batch_targets.size())
        break