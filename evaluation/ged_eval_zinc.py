import os
import pickle
import torch
import collections
import numpy as np
from dataset.load_dataset import load_dataset
from torch.utils.data import DataLoader
from utils.utils import set_seed
from models.model_factory import get_model
import dgl
import scipy
import tqdm
from dgl import DGLGraph
import networkx as nx
from dataset.visualization import vis_graph, plot
from scipy.optimize import linear_sum_assignment
import collections


print(dgl.__file__)

class compute_GED():
    def __init__(self, mode="same"):
        self.mode = mode
    
    def match_num_edges(self, gt_adj, pred_adj):
        ## match the number of edges for gt_adj and pred_adj
        # remove self_loop
        np.fill_diagonal(gt_adj, 0.0)
        np.fill_diagonal(pred_adj, 0.0)
        n_edges = int(np.sum(gt_adj))
        
        pred_adj = pred_adj.flatten()
        # get the index of top n_edges 
        
        if self.mode == "same":
            idx = np.argpartition(-pred_adj, int(n_edges*1.0), axis=-1)
            idx = idx[:n_edges]
        else:
            # mean edges
            idx = np.where(pred_adj > np.mean(pred_adj))[0]

        pred_adj *= 0.0
        pred_adj[idx] = pred_adj[idx] + 1.0
        pred_adj = pred_adj.reshape((gt_adj.shape[0],gt_adj.shape[1]))
        return gt_adj, pred_adj

    def get_GED(self, gt, pred):
        gt = self.convert_to_nx(gt)
        pred = self.convert_to_nx(pred)
        
        gt_adj = nx.to_numpy_array(gt)
        pred_adj = nx.to_numpy_array(pred)
        
        # remove diagonal
        np.fill_diagonal(gt_adj, 0.0)
        np.fill_diagonal(pred_adj, 0.0)

        gt_adj, pred_adj = self.match_num_edges(gt_adj, pred_adj)

        sum_adj = gt_adj + pred_adj
        sum_adj = sum_adj.reshape((-1,1))
        # TODO not safe to use ==
        indices = np.where( sum_adj == 1.0 )[0] 
        return indices.shape[0]
        # optimal GED is very slow
        # return nx.graph_edit_distance(gt, pred)

    def convert_to_nx(self, g):
        if isinstance(g, nx.Graph):
            pass
        elif isinstance(g, dgl.DGLGraph):
            g = g.to_networkx()
        elif isinstance(g, np.ndarray):
            g = nx.DiGraph(g)
        else:
            raise NameError('unknow format of input graph')
        return g

    def hungarian_match(self, gt_list, pred_list):
        gt_list, gt_ids = gt_list
        gt_len = len(gt_list)
        pred_len = len(pred_list)
        cost = np.zeros((gt_len, pred_len))
        for gt_i, gt in enumerate(gt_list):
            for pred_i, pred in enumerate(pred_list):
                cost[gt_i, pred_i] = self.get_GED(
                                gt,
                                pred)
        
        row_ind, col_ind = linear_sum_assignment(cost)
        
        factor_map = collections.defaultdict(list)
        for r, c in zip(row_ind, col_ind):
            edge_id = gt_ids[r]
            factor_map[edge_id].append(c)

        total_ED = cost[row_ind, col_ind].sum()
        return total_ED, factor_map
        

def translate_facgorGNN_format(factor_g):
    output_gs = []
    for i in range(16):
        if f'factor_{i}' not in factor_g.edata:
            break
        
        n_node = factor_g.number_of_nodes()
        srt, dst = factor_g.edges()
        
        edge_factor = factor_g.edata[f'factor_{i}'].squeeze()
        edge_factor = edge_factor.detach().cpu().numpy()
        
        ## operate in the matrix form
        srt, dst = srt.detach().cpu().numpy(), dst.detach().cpu().numpy()
        org_g = np.zeros((n_node, n_node))
        org_g[srt, dst] = edge_factor
        
        # bigraph to graph
        org_g += np.transpose(org_g)
        org_g /= 2.0
        
        # # np.fill_diagonal(org_g, 0.0)
        # indices = np.where(org_g > np.mean(org_g))
        # output_g = np.zeros_like(org_g)
        # output_g[indices] = 1.0
        
        output_gs.append(org_g)
    
    return output_gs


def translate_gat_format(factor_g):
    output_gs = []
    atts = factor_g.edata['a']  # n * att_head * 1
    atts = atts.squeeze(-1).detach().cpu().numpy()
    for att_i in range(atts.shape[1]):
        n_node = factor_g.number_of_nodes()
        srt, dst = factor_g.edges()
        
        edge_factor = atts[:, att_i]
        
        ## operate in the matrix form
        srt, dst = srt.detach().cpu().numpy(), dst.detach().cpu().numpy()
        org_g = np.zeros((n_node, n_node))
        org_g[srt, dst] = edge_factor
        
        # bigraph to graph
        org_g += np.transpose(org_g)
        org_g /= 2.0
        
        output_gs.append(org_g)
    
    return output_gs


def generate_adj_factor_graph(factor_g):
    if 'factor_0' in factor_g.edata:
        out = translate_facgorGNN_format(factor_g)
    elif 'a' in factor_g.edata:
        out = translate_gat_format(factor_g)
    return out


def translate_gt_graph_to_adj(gt_graph):
    gt_adjs = []
    gt_g_list = dgl.unbatch(gt_graph)
    for gt_g in gt_g_list:
        gt_list = []
        gt_ids = []

        n_node = gt_g.number_of_nodes()
        srt, dst = gt_g.edges()
        srt, dst = srt.detach().cpu().numpy(), dst.detach().cpu().numpy()
        
        edge_factor = gt_g.edata['feat'].detach().cpu().numpy()
        assert srt.shape[0] == edge_factor.shape[0]


        for edge_id in set(edge_factor):
            ## operate in the matrix form
            org_g = np.zeros((n_node, n_node))
            edge_factor_edge_id = np.zeros_like(edge_factor)
            idx = np.where(edge_factor == edge_id)[0] 
            edge_factor_edge_id[idx] = 1.0
            org_g[srt, dst] = edge_factor_edge_id
            gt_list.append(org_g)
            gt_ids.append(edge_id)

        gt_adjs.append((gt_list, gt_ids))

    return gt_adjs


def compute_consistant(total_factor_map):
    scores = []
    for edge_id in total_factor_map.keys():
        inds = total_factor_map[edge_id]
        most_id = max(set(inds), key = inds.count)
        scores.append(float(inds.count(most_id)) / len(inds))
    return np.mean(scores)


def forward_model(best_model, method):
    args = best_model['args']
    
    torch.cuda.set_device(args.gpu)
    set_seed(args)

    # load and preprocess dataset
    zinc_data = load_dataset(args)
    train_loader = DataLoader(zinc_data.train, batch_size=1000, shuffle=True,
                         collate_fn=zinc_data.collate, num_workers=4)
    val_loader = DataLoader(zinc_data.val, batch_size=1000, shuffle=False,
                         collate_fn=zinc_data.collate)
    test_loader = DataLoader(zinc_data.test, batch_size=1000, shuffle=False,
                         collate_fn=zinc_data.collate)

    # placeholder of dataset
    dataset = (None, None, None, None, None, None, None)
    # create model
    model = get_model(dataset, args, mode='zinc').cuda()
    model.load_state_dict(best_model['model_state_dict'])
    model.eval()

    batch_graphs, batch_targets, batch_snorm_n, batch_snorm_e = next(iter(train_loader))
    batch_x = batch_graphs.ndata['feat'].cuda()  # num x feat
    batch_e = batch_graphs.edata['feat'].cuda()
    batch_snorm_e = batch_snorm_e.cuda()
    batch_targets = batch_targets.cuda()
    batch_snorm_n = batch_snorm_n.cuda()         # num x 1
    
    model.g = batch_graphs
    batch_scores = model.forward(batch_x, batch_e, batch_snorm_n, batch_snorm_e)
    
    factors = model.get_factor()

    batch_g = factors[1]
    unbatch_g = dgl.unbatch(batch_g)
    
    gt_adjs = translate_gt_graph_to_adj(batch_graphs)

    ged_ins = compute_GED()
    
    total_ged = []
    total_factor_map = collections.defaultdict(list)
    for gt_list, pred_g in tqdm.tqdm(zip(gt_adjs, unbatch_g)):
        # dgl graph to adj
        pred_list = generate_adj_factor_graph(pred_g)
        ged, factor_map = ged_ins.hungarian_match(gt_list, pred_list)
        
        for edge_id in factor_map.keys():
            total_factor_map[edge_id] = total_factor_map[edge_id] + factor_map[edge_id]

        total_ged.append(ged / len(gt_list))
    
    c_score = compute_consistant(total_factor_map)

    print(f"{method}| loss {best_model['loss']:.3f} | c_score {c_score:.3f} | ged: {np.mean(total_ged):.2f} $\pm$ {np.std(total_ged):.2f}")


if __name__ == '__main__':
    
    res_root = "./data/run_log"
    dataset = 'zinc'
    
    methods = ['factorgnn']
    runs = ["run0001"]

    for method, run in zip(methods, runs):
        run_file = os.path.join(res_root, dataset, method, run)
        best_model = torch.load(os.path.join(run_file, 'best_model.pt'))
        forward_model(best_model, method)
    