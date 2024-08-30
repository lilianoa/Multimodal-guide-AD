import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, average_precision_score, roc_curve, roc_auc_score
import numpy as np


def image_level_metrics(gt, pr, metric):
    gt = np.array(gt)
    pr = np.array(pr)
    if metric == 'image-auroc':
        performance = roc_auc_score(gt, pr)
    elif metric == 'image-ap':
        performance = average_precision_score(gt, pr)

    return performance


class Metric_test(nn.Module):
    def __init__(self):
        super(Metric_test, self).__init__()

    def __call__(self, net_out, batch):
        out = {}
        logits = net_out['logits'].data.cpu()
        _, pres = torch.max(logits, dim=1)
        label_id = batch['label_id'].data.cpu()
        acc, sen, spe, f1, roc, ap = metric_test(pres, label_id)
        out['acc'] = torch.tensor(acc)
        out['sen'] = torch.tensor(sen)
        out['spe'] = torch.tensor(spe)
        out['f1'] = torch.tensor(f1)
        out['roc'] = torch.tensor(roc)
        out['ap'] = torch.tensor(ap)
        return out

def metric_test(output, target):
    with torch.no_grad():
        cm = confusion_matrix(target, output)
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tp = cm[1, 1]
        acc = (tp + tn) / (tp + tn + fp + fn)
        sen = tp / (tp + fn)
        spe = tn / (tn + fp)
        f1 = 2*tp / (2*tp + fp + fn)   # F1 score
        roc = roc_auc_score(target, output)
        ap = average_precision_score(target, output)
        return acc*100, sen*100, spe*100, f1*100, roc, ap

