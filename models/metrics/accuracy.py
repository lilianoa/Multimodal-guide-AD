import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

class Accuracy(nn.Module):

    def __init__(self):
        super(Accuracy, self).__init__()

    def __call__(self, net_out, batch):
        out = {}
        logits = net_out['logits'].data.cpu()
        label_id = batch['label_id'].data.cpu()
        acc_out = accuracy(logits, label_id)
        out['accuracy'] = acc_out
        return out

def accuracy(output, target):
    """Computes the accuracy"""
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        correct = pred.eq(target.view_as(pred))
        correct = correct.view(-1).float().sum(0)
        acc = correct.mul_(100.0 / batch_size)
        return acc

class Accuracy_AUROC(nn.Module):

    def __init__(self):
        super(Accuracy, self).__init__()

    def __call__(self, net_out, batch):
        out = {}
        logits = net_out['logits'].data.cpu()
        label_id = batch['label_id'].data.cpu()
        acc_out = accuracy(logits, label_id)
        out['accuracy'] = acc_out
        pr_ = F.softmax(logits, dim=-1)
        pr = torch.chunk(pr_, 2, dim=-1)[-1]
        out['AUROC'] = roc_auc_score(label_id, pr)
        return out
