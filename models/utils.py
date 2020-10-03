import numpy as np
import torch
from sklearn.metrics import f1_score
from ogb.nodeproppred import Evaluator

evaluator = Evaluator(name = "ogbn-proteins")
# print(evaluator.expected_input_format) 
# print(evaluator.expected_output_format) 

def evaluate_multilabel(model, feature, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = torch.sigmoid(model(features))
        logits = logits[mask]
        labels = labels[mask]
        return evaluator({'y_true': labels, 'y_pred': logits})


def evaluate_f1(logits, labels):
    # logits = torch.sigmoid(logits)
    y_pred = torch.where(logits > 0.0, torch.ones_like(logits), torch.zeros_like(logits))
    y_pred = y_pred.detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()
    return f1_score(y_true, y_pred, average='micro')


def accuracy(logits, labels):
    
    
    _, indices = torch.max(logits, dim=1)
    if len(indices.shape) > 1:
        indices = indices.view(-1,)
        labels = labels.view(-1,)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

loss_fcn = torch.nn.CrossEntropyLoss()
def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        if len(labels.shape) > 1:
            logits = logits[mask].view(-1, torch.max(labels) + 1, labels.shape[1])
        else:
            logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels), loss_fcn(logits, labels).item()


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score <= self.best_score:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'es_checkpoint.pt')
    
    def load_checkpoint(self, model):
        model.load_state_dict( torch.load('es_checkpoint.pt') )
        model.eval()