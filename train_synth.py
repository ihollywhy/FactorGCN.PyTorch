import os
import sys
import argparse
import numpy as np
import networkx as nx
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dgl import DGLGraph
import dgl
from dgl.data import register_data_args, load_data, CoraFull
from models.model_factory import get_model
from models.utils import EarlyStopping, evaluate, accuracy, evaluate_multilabel, evaluate_f1
from utils.utils import set_seed, make_log_dir
from dataset.load_dataset import load_dataset
from dataset.visualization import vis_graph


def test(model, data_loder):
    with torch.no_grad():
        logits_all = []
        labels_all = []
        model.eval()
        for step, (g, labels, gt_adjs) in enumerate(data_loder):
            model.g = g
            features = g.ndata['feat'].float().cuda()
            labels = labels.cuda()
            logits = model(features)
            logits_all.append(logits.detach())
            labels_all.append(labels.detach())
        
        logits_all = torch.cat(tuple(logits_all), 0)
        labels_all = torch.cat(tuple(labels_all), 0)
        micro_f1 = evaluate_f1(logits_all, labels_all)
    return micro_f1

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels, gt_adjs = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.cat(tuple(labels), 0), gt_adjs


def main(args):
    torch.cuda.set_device(args.gpu)
    set_seed(args)

    log_dir = make_log_dir(args.model_name, args.dataset, args.log_subdir)
    
    log_file = os.path.join(log_dir, "log.txt")
    sys.stdout = open(log_file, 'w')
    backup_model = f"cp -r ./models {log_dir}"
    os.system(backup_model)

    # load and preprocess dataset
    all_data = load_dataset(args)
    training = all_data[:int(len(all_data)*0.7)]
    validation = all_data[int(len(all_data)*0.7) : int(len(all_data)*0.8)]
    testing = all_data[int(len(all_data)*0.8):]

    train_loader = DataLoader(training, batch_size=1000, shuffle=True,
                         collate_fn=collate)
    val_loader = DataLoader(validation, batch_size=1000, shuffle=True,
                         collate_fn=collate)
    test_loader = DataLoader(testing, batch_size=1000, shuffle=False,
                         collate_fn=collate)

    dataset = (None, np.zeros((15, 15)), np.zeros((1, args.num_factors)), None, None, None, None)
    # create model
    model = get_model(dataset, args, mode='multilabel').cuda()
    
    print(model)

    # define loss func
    loss_fcn = torch.nn.BCEWithLogitsLoss()

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=args.lr, 
                                weight_decay=args.weight_decay)

    best_val_f1 = 0
    best_test_f1 = 0
    dur= []
    for epoch in range(args.epochs):
        for step, (g, labels, gt_adjs) in enumerate(train_loader):
            model.train()

            # update the new graph
            model.g = g
            
            t0 = time.time()
            features = g.ndata['feat'].float().cuda()
            labels = labels.cuda()
            logits = model(features) #.view(-1, n_class, n_latent)
            loss = loss_fcn(logits, labels)
            
            if args.model_name == 'FactorGNN' and args.dis_weight > 0.0:
                losses = model.compute_disentangle_loss()
                dis_loss = model.merge_loss(losses) * args.dis_weight
                loss = loss + dis_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dur.append(time.time() - t0)
        
        val_micro_f1 = test(model, val_loader)
        test_micro_f1 = test(model, val_loader)
        
        if val_micro_f1 > best_val_f1:
            best_val_f1 = val_micro_f1
            best_test_f1 = test_micro_f1
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_test_f1,
                        'args': args}, 
                        os.path.join(log_dir, 'best_model.pt') )

        print(  f"time {np.mean(dur):.2f} epoch {epoch:03d} | " + 
                f"val ({val_micro_f1:.4f}) | "+
                f"test ({test_micro_f1:.4f}) | "+
                f"best: {best_test_f1:.4f}")
        
        sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    parser.add_argument("--model-name", type=str, default="FactorGNN",
                        help="FactorGNN, GAT, MLP, GCN, DisenGCN")
    parser.add_argument("--gpu", type=int, default=1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=80,
                        help="number of training epochs")

    parser.add_argument("--num_factors", type=int, default=4,
                        help="number of factor graph in the dataset")

    parser.add_argument("--num-latent", type=int, default=4,
                        help="number of training epochs")
    parser.add_argument("--num-hidden", type=int, default=32,
                        help="number of hidden units")
    parser.add_argument("--dis-weight", type=float, default=1.0,
                        help="weight of disentangle")

    parser.add_argument("--ncaps", type=int, default=4,
                        help="")
    parser.add_argument("--nhidden", type=int, default=8,
                        help="")
    parser.add_argument("--routit", type=int, default=8,
                        help="")
    parser.add_argument("--nlayer", type=int, default=2,
                        help="")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="")

    parser.add_argument("--log-subdir", type=str, default="run0000",
                        help="the subdir name of log, eg. run0000")

    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--residual", action="store_true", default=True,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0.0,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0.0,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-5,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--seed', type=int, default=100,
                        help="set seed")
    args = parser.parse_args()
    print(args)

    main(args)