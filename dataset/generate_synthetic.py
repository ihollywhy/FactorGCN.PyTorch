import os, collections
import networkx as nx
import numpy as np
import torch, dgl
import random, pickle


class synthetic_graph_cls:
    # generate several graphs with different paterns and union them
    # labels is whether the union graph contains speicific pattern
    def __init__(self, args):
        self.args = args
        self.saved_file = f'./data/synthetic/synthetic_graph_cls_data_{args.num_factors}.pkl'
        os.makedirs(os.path.dirname(self.saved_file), exist_ok=True)
        
    def gen_union_graph(self, graph_size=15, num_graph=20000):
        if os.path.isfile(self.saved_file):
            print(f"load synthetic graph cls data from {self.saved_file}")
            with open(self.saved_file, 'rb') as f:
                return pickle.load(f)
        
        graph_list = synthetic_graph_cls.get_graph_list(self.args.num_factors)
        samples = []

        for _ in range(num_graph):
            union_adj = np.zeros((graph_size, graph_size))
            factor_adjs = []
            labels = np.zeros((1, len(graph_list)))
            
            id_index = list(range(len(graph_list)))
            random.shuffle(id_index)
            
            for i in range((len(id_index)+1)//2): # get random half adj
                id = id_index[i]
                labels[0, id] = 1

                single_adj = graph_list[id]
                padded_adj = np.zeros((graph_size, graph_size))
                padded_adj[:single_adj.shape[0], :single_adj.shape[0]] = single_adj
                
                random_index = np.arange(padded_adj.shape[0])
                np.random.shuffle(random_index)
                padded_adj = padded_adj[random_index]
                padded_adj = padded_adj[:, random_index]
                
                union_adj += padded_adj
                factor_adjs.append((padded_adj, id))

            g = dgl.DGLGraph()
            g.from_networkx(nx.DiGraph(union_adj))
            g = dgl.transform.add_self_loop(g)
            g.ndata['feat'] = torch.tensor(union_adj)
            labels = torch.tensor(labels)
            samples.append((g, labels, factor_adjs))
        with open(self.saved_file, 'wb') as f:
            pickle.dump(samples, f)
            print(f"dataset saved to {self.saved_file}")
            
        return samples

    @staticmethod
    def get_graph_list(num_factors):
        graph_list = []
        # 2, 3 bipartite graph
        g = nx.turan_graph(n=5, r=2)
        graph_list.append(nx.to_numpy_array(g))

        g = nx.house_x_graph()
        graph_list.append(nx.to_numpy_array(g))
        
        g = nx.balanced_tree(r=3, h=2)
        graph_list.append(nx.to_numpy_array(g))
        
        g = nx.grid_2d_graph(m=3, n=3)
        graph_list.append(nx.to_numpy_array(g))

        g = nx.hypercube_graph(n=3)
        graph_list.append(nx.to_numpy_array(g))

        g = nx.octahedral_graph()
        graph_list.append(nx.to_numpy_array(g))
        return graph_list[:num_factors]
