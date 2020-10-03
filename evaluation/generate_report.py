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
from evaluation.correlation_plot import plot_corr


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels, gt_adjs = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.cat(tuple(labels), 0), gt_adjs


def get_mae_report(res_root = "./data/run_log",
                    dataset = 'zinc',
                    methods = ['factorgnn']
                ):
    report = collections.defaultdict(list)

    for method in methods:
        run_files_dir = os.path.join(res_root, dataset, method)
        for run_file in os.listdir(run_files_dir):
            run_file = os.path.join(run_files_dir, run_file)
            best_model = torch.load(os.path.join(run_file, 'best_model.pt'))
            
            report[method].append(best_model['loss'])


    for m in report.keys():
        res = report[m]
        print(m, f"{np.mean(res):.3f}", f"$\pm$ {np.std(res):.3f}")


def get_acc_report(res_root = "./data/run_log", 
                    dataset = 'synthetic_graph_cls',
                    methods = ['factorgnn']
                ):
    report = collections.defaultdict(dict)

    for method in methods:
        run_files_dir = os.path.join(res_root, dataset, method)
        for run_file in os.listdir(run_files_dir):
            run_file = os.path.join(run_files_dir, run_file)
            best_model = torch.load(os.path.join(run_file, 'best_model.pt'))
            
            n_f = best_model['args'].num_factors
            if n_f not in report[method]:
                report[method][n_f] = []
            report[method][n_f].append(best_model['loss'])


    for m in report.keys():
        for n_f in report[m].keys():
            res = report[m][n_f]
            print(m, n_f, f"{np.mean(res):.3f}", f"$\pm$ {np.std(res):.3f}")


def get_10fold_curve_report(res_root = "./data/run_log", 
                                dataset='COLLAB', 
                                methods=['factorgnn']
                        ):
    report = collections.defaultdict(list)

    curves = []
    for method in methods:
        run_files_dir = os.path.join(res_root, dataset, method)
        for run_file in os.listdir(run_files_dir):
            if run_file[:3] != "run": continue
            run_file = os.path.join(run_files_dir, run_file)
            log_file = os.path.join(run_file, 'log.txt')
            
            curve = get_curve_from_file(log_file)
            curves.append(curve)
    
    curves = np.array(curves)
    mean = np.mean(curves, axis=0)
    index = np.argmax(mean)
    print(index, np.max(mean), np.std(curves[:, index]))


def get_curve_from_file(file_path=None):
    import re
    val_accs = []
    prog = re.compile("val acc [\d.]*")
    
    with open(file_path, "r") as f:
        line = f.readline()
        while line:
            result = prog.search(line)
            if result is not None:
                start, end = result.span()
                start += 8
                val_accs.append( float(line[start:end]) )
            line = f.readline()
    
    return val_accs


def forward_model(best_model, method):
    args = best_model['args']
    
    torch.cuda.set_device(args.gpu)
    set_seed(args)

    # load and preprocess dataset
    all_data = load_dataset(args)
    training = all_data[:int(len(all_data)*0.7)]
    validation = all_data[int(len(all_data)*0.7) : int(len(all_data)*0.8)]
    testing = all_data[int(len(all_data)*0.8):]

    train_loader = DataLoader(training, batch_size=1000, shuffle=True,
                         collate_fn=collate)
    val_loader = DataLoader(validation, batch_size=1000, shuffle=True,
                         collate_fn=collate)
    test_loader = DataLoader(testing, batch_size=4000, shuffle=False,
                         collate_fn=collate)

    dataset = (None, np.zeros((15, 15)), np.zeros((1, args.num_factors)), None, None, None, None)
    # create model
    model = get_model(dataset, args, mode='multilabel').cuda()
    
    for step, (g, labels, gt_adjs) in enumerate(test_loader):
        model.load_state_dict(best_model['model_state_dict'])
        model.eval()

        # update the new graph
        model.g = g
        
        features = g.ndata['feat'].float().cuda()
        labels = labels.cuda()
        logits = model(features) #.view(-1, n_class, n_latent)

        hidden = model.get_hidden_feature()
        matrix = hidden[0]   # #sample x dim
        correlation = np.zeros((matrix.shape[1], matrix.shape[1]))
        for i in range(matrix.shape[1]):
            for j in range(matrix.shape[1]):
                cof = scipy.stats.pearsonr(matrix[:, i], matrix[:, j])[0]
                correlation[i][j] = cof

        plot_corr(np.abs(correlation), save=f'{method}.png')


if __name__ == '__main__':
    get_acc_report()
    get_mae_report()
    get_10fold_curve_report()
    