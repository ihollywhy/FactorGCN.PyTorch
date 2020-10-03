import os
import sys
import argparse
import numpy as np
import networkx as nx
import time
import torch
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data, CoraFull
from models.model_factory import get_model
from models.utils import EarlyStopping, evaluate, accuracy
from utils.utils import set_seed, make_log_dir
from dataset.gin_dataset import load_gin_dataset
import copy
import matplotlib.pyplot as plt
from dgl.data import GINDataset
from torch.utils.data import DataLoader


def eval_net(args, model, dataloader, criterion):
    model.eval()

    total = 0
    total_loss = 0
    total_correct = 0

    for data in dataloader:
        graphs, labels = data
        model.g = graphs

        if args.dataset in ["IMDBBINARY", "COLLAB"]:
            if args.dataset == "IMDBBINARY": in_dim = 150
            if args.dataset == "COLLAB": in_dim = 500
            y = graphs.in_degrees().long().unsqueeze(-1)
            y_onehot = torch.FloatTensor(graphs.number_of_nodes(), in_dim)
            y_onehot.zero_()
            y_onehot.scatter_(1, y, 1)
            feat = y_onehot.float().cuda()
        else:
            feat = graphs.ndata['attr'].float().cuda()
        
        labels = labels.cuda()
        total += len(labels)
        outputs = model(feat)
        _, predicted = torch.max(outputs.data, 1)

        total_correct += (predicted == labels.data).sum().item()
        loss = criterion(outputs, labels)
        # crossentropy(reduce=True) for default
        total_loss += loss.item() * len(labels)

    loss, acc = 1.0*total_loss / total, 1.0*total_correct / total

    model.train()

    return loss, acc


def main(args):
    torch.cuda.set_device(args.gpu)
    set_seed(args)
    
    if args.log_subdir != "":
        log_dir = make_log_dir(args.model_name, args.dataset, args.log_subdir)
        
        log_file = os.path.join(log_dir, "log.txt")
        sys.stdout = open(log_file, 'w')
        backup_model = f"cp -r ./models {log_dir}"
        os.system(backup_model)

    # load and preprocess dataset
    train_loader, val_loader = load_gin_dataset(args)
    
    # num_feats = features.shape[1]
    # n_classes = torch.max(labels).item() + 1)
    
    # create model
    sample = next(iter(train_loader))
    n_class_dict = {'MUTAG':2, "IMDBBINARY":2, "COLLAB": 3}
    
    in_dim = sample[0].ndata['attr'].shape[1]
    if args.dataset == "IMDBBINARY": in_dim = 150
    if args.dataset == "COLLAB": in_dim = 500
    feat = torch.ones(1, in_dim)
    dataset = (None, feat, 
                torch.tensor([n_class_dict[args.dataset]-1]), None, None, None, None)
    model = get_model(dataset, args).cuda()

    # define loss func
    loss_fcn = torch.nn.CrossEntropyLoss()

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=args.lr, 
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    dur = []
    best_acc = 0
    for epoch in range(args.epochs):
        total_loss = []
        # total_edges = 0
        # total_nodes = 0
        for step, (g, labels) in enumerate(train_loader):
            
            # total_nodes += g.number_of_nodes()
            # total_edges += g.number_of_edges()
            # continue
            model.train()
            # update the new graph
            model.g = g
            # print(max(g.in_degrees()))
            t0 = time.time()
            
            if args.dataset in ["IMDBBINARY", "COLLAB"]:
                if args.dataset == "IMDBBINARY": in_dim = 150
                if args.dataset == "COLLAB": in_dim = 500
                y = g.in_degrees().long().unsqueeze(-1)
                y_onehot = torch.FloatTensor(g.number_of_nodes(), in_dim)
                y_onehot.zero_()
                y_onehot.scatter_(1, y, 1)
                features = y_onehot.float().cuda()
            else:
                features = g.ndata['attr'].float().cuda()
            
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

            total_loss.append(loss.item())

            dur.append(time.time() - t0)
        
        loss, acc = eval_net(args, model, train_loader, loss_fcn)
        val_loss, val_acc = eval_net(args, model, val_loader, loss_fcn)
        if val_acc > best_acc:
            best_acc = val_acc
            if args.log_subdir != "":
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_acc,
                        'args': args}, 
                        os.path.join(log_dir, 'best_model.pt') )
        print(f"epoch {epoch:03d} | train_loss {np.mean(total_loss):.3f} | train acc {acc:.3f} | val acc {val_acc:.3f} | best {best_acc:.3f}")
        if args.log_subdir != "":
            sys.stdout.flush()

        scheduler.step()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    register_data_args(parser)
    parser.add_argument("--model-name", type=str, default="FactorGNN",
                        help="which model to use. GAT, FactorGNN")
    # common params
    parser.add_argument("--gpu", type=int, default=1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fold-idx", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500,
                        help="number of training epochs")
    
    parser.add_argument("--in-drop", type=float, default=0.2,
                        help="input feature dropout")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=64,
                        help="number of hidden units")
    parser.add_argument("--num-latent", type=int, default=4,
                        help="number of latent factors")
    parser.add_argument("--dis-weight", type=float, default=0.2,
                        help="weight of addtional loss")

    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")

    parser.add_argument("--log-subdir", type=str, default="",
                        help="the subdir name of log, eg. run0000")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--seed', type=int, default=100,
                        help="set seed")
    args = parser.parse_args()
    print(args)
    
    set_seed(args)
    main(args)