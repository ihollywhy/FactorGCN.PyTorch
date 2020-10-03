import sys
import os
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
from models.utils import EarlyStopping
from utils.utils import set_seed, make_log_dir
from dataset.load_dataset import load_dataset
from dataset.visualization import vis_graph
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import tqdm


n_classes = 6
def loss_fcn(pred, label):

    # calculating label weights for weighted loss computation
    V = label.size(0)
    label_count = torch.bincount(label)
    label_count = label_count[label_count.nonzero()].squeeze()
    cluster_sizes = torch.zeros(n_classes).long().cuda()
    cluster_sizes[torch.unique(label)] = label_count
    weight = (V - cluster_sizes).float() / V
    weight *= (cluster_sizes>0).float()
    
    # weighted cross-entropy for unbalanced classes
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    loss = criterion(pred, label)

    return loss


def accuracy(scores, targets):
    S = targets.cpu().numpy()
    C = np.argmax( torch.nn.Softmax(dim=1)(scores).cpu().detach().numpy() , axis=1 )
    CM = confusion_matrix(S,C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets==r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r,r]/ float(cluster.shape[0])
            if CM[r,r]>0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = 100.* np.sum(pr_classes)/ float(nb_non_empty_classes)
    return acc


def test(model, data_loader):
    model.eval()
    loss = 0
    acc = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].cuda()  # num x feat
            batch_e = batch_graphs.edata['feat'].cuda()
            batch_snorm_e = batch_snorm_e.cuda()
            batch_targets = batch_targets.cuda()
            batch_snorm_n = batch_snorm_n.cuda()         # num x 1
            
            model.g = batch_graphs
            batch_scores = model.forward(batch_x, batch_e, batch_snorm_n, batch_snorm_e)
            
            loss = loss_fcn(batch_scores, batch_targets)
            iter_loss = loss.item()
            iter_acc = accuracy(batch_scores, batch_targets)
            loss += iter_loss
            acc += iter_acc
        
    loss /= (iter + 1)
    acc /= (iter + 1)
    return loss, acc


def main(args):
    torch.cuda.set_device(args.gpu)
    set_seed(args)
    global n_classes
    if args.dataset == 'sbms_cluster':
        n_classes = 6 
    if args.dataset == 'sbms_pattern':
        n_classes = 2
    
    log_dir = make_log_dir(args.model_name, args.dataset, args.log_subdir)
    
    log_file = os.path.join(log_dir, "log.txt")
    sys.stdout = open(log_file, 'w')
    backup_model = f"cp -r ./models {log_dir}"
    os.system(backup_model)
    
    # load and preprocess dataset
    sbms_data = load_dataset(args)
    
    train_loader = DataLoader(sbms_data.train, batch_size=64, shuffle=True,
                         collate_fn=sbms_data.collate, num_workers=4)
    val_loader = DataLoader(sbms_data.val, batch_size=500, shuffle=False,
                         collate_fn=sbms_data.collate)
    test_loader = DataLoader(sbms_data.test, batch_size=500, shuffle=False,
                         collate_fn=sbms_data.collate)

    # placeholder of dataset
    dataset = (None, None, None, None, None, None, None)
    # create model
    model = get_model(dataset, args, mode='sbms').cuda()
    
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=args.lr, 
                                weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5,
                                                     patience=50,
                                                     verbose=True)

    best_val_loss = sys.maxsize
    best_test_acc = 0
    dur = []
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_train_acc = 0
        t0 = time.time()
        for iter, (batch_graphs, batch_targets, batch_snorm_n, batch_snorm_e) in tqdm.tqdm(enumerate(train_loader)):
            batch_x = batch_graphs.ndata['feat'].cuda()  # num x feat
            batch_e = batch_graphs.edata['feat'].cuda()
            batch_snorm_e = batch_snorm_e.cuda()
            batch_targets = batch_targets.cuda()
            batch_snorm_n = batch_snorm_n.cuda()         # num x 1
            
            optimizer.zero_grad()
            
            model.g = batch_graphs
            batch_scores = model.forward(batch_x, batch_e, batch_snorm_n, batch_snorm_e)
            
            loss = loss_fcn(batch_scores, batch_targets)
            
            if args.model_name == "FactorGNN" and args.dis_weight > 0.0:
                losses = model.compute_disentangle_loss()
                dis_loss = model.merge_loss(losses) * args.dis_weight
                loss = loss + dis_loss
            if args.model_name == "IPDGN":
                HSIC = model.compute_HSIC()
                loss = loss + HSIC * 0.1

            loss.backward()
            optimizer.step()
            
            iter_loss = loss.item()
            iter_acc = accuracy(batch_scores, batch_targets)
            epoch_loss += iter_loss
            epoch_train_acc += iter_acc
        
        dur.append(time.time() - t0)
        epoch_loss /= (iter + 1)
        epoch_train_acc /= (iter + 1)
        # print(f"loss {epoch_loss:.4f}, mae {epoch_train_acc:.4f}")
        val_loss, val_acc = test(model, val_loader)
        test_loss, test_acc = test(model, test_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_test_acc = test_acc
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_test_acc,
                        'args': args}, 
                        os.path.join(log_dir, 'best_model.pt') )

        print(  f"time {np.mean(dur):.2f} epoch {epoch:03d} | " + 
                f"train ({epoch_loss:.4f}, {epoch_train_acc:.4f}) | "+
                f"val ({val_loss:.4f}, {val_acc:.4f}) | "+
                f"test ({test_loss:.4f}, {test_acc:.4f}) | "+
                f"best: {best_test_acc:.4f}")
        
        sys.stdout.flush()
        
        if optimizer.param_groups[0]['lr'] > 1e-5:
            scheduler.step(val_loss)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    register_data_args(parser)
    parser.add_argument("--model-name", type=str, default="FactorGNN",
                        help="FactorGNN, GAT")
    parser.add_argument("--gpu", type=int, default=1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=15,
                        help="number of training epochs")
    parser.add_argument("--log-subdir", type=str, default="run0000",
                        help="the subdir name of log, eg. run0000")


    parser.add_argument("--ncaps", type=int, default=4,
                        help="")
    parser.add_argument("--nhidden", type=int, default=32,
                        help="")
    parser.add_argument("--routit", type=int, default=5,
                        help="")
    parser.add_argument("--nlayer", type=int, default=4,
                        help="")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="")


    parser.add_argument("--in-dim", type=int, default=36,
                        help="dim of embedded feature")
    parser.add_argument("--num-latent", type=int, default=8,
                        help="number of training epochs")
    parser.add_argument("--num-hidden", type=int, default=18*8,
                        help="number of hidden units")
    parser.add_argument("--dis-weight", type=float, default=0.5,
                        help="weight of disentangle")

    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=3,
                        help="number of hidden layers")
    parser.add_argument("--residual", action="store_true", default=True,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0.0,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0.0,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0.0,
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