import random
import numpy as np
import torch
import dgl
import os


def set_seed(args=None):
    seed = 1 if not args else args.seed
    
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dgl.random.seed(seed)


def make_log_dir(model_name, dataset, subdir):
    # make and return
    model_name = model_name.lower()
    log_dir = os.path.join(f"./data/run_log/{dataset}", model_name, subdir)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir
