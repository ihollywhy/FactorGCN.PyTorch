# PyTorch implementation of FactorGCN

Paper: [Factorizable Graph Convolutional Networks](), NeurIPS'20

# Dependencies

See [requirment](requirment.txt) file for more information
about how to install the dependencies.

# Usage

## 1, Prerequisites

### Datasets

We provide [here](https://drive.google.com/file/d/1WeSVyzftUCcLm6q_U-g5Wzka5WUB-TIy/view?usp=sharing) 
datasets that are ready to use
within this project.
Download the datasets and unzip it into
**./data** dir.
These datasets can also be either 
downloaded from their official websites 
or generated on the fly.

### Models

We use [DGL](https://www.dgl.ai/) to 
implement all the GCN models. 
In order to evaluate the disentanglement performance of GAT model, 
you need to modify the last line of 

> dgl/nn/pytorch/conv/gatconv.py

from
`return rst`
to
`return rst, graph.local_var(), graph.edata['a']`

## 2, Training

The `train_*.sh` scripts contains the training codes for corresponding datasets and methods. 

> **train_synth.sh** for Synthetic dataset;

> **train_zinc.sh** for ZINC dataset;

> **train_gin.sh** for IMDB-B, COLLAB, and MUTAG datasets.

The model as well as the training log
will be saved to the corresponding dir in **./data** for evaluation.

## 3, Evaluation

The **evaluation** dir contains the codes and examples for evaluating the performance of
both the task-related performance and the disentanglement performance.

> **generate_report.get_acc_report** reports the accuracy on the 
Synthetic dataset;

> **generate_report.get_mae_report** reports the MAE on the ZINC dataset;

> **generate_report.get_10fold_curve_report** report the 10-fold cross-validation performance on IMDB-B, COLLAB, and MUTAG datasets.

The performances of disentanglement are evaluted
based the ![](./asserts/ged_e.gif) and C-score.

> **ged_eval_synth** generates the disentanglement performance
on the Synthetic dataset;

> **ged_eval_zinc** generates the disentanglement performance
on the ZINC dataset.

# License

FactorGCN is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.
