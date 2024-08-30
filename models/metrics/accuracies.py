import torch
import torch.nn as nn
from .accuracy import Accuracy
import numpy as np
from bootstrap.lib.logger import Logger


class Accuracies(nn.Module):
    def __init__(self,
                 engine=None,
                 mode='val',
                 dir_exp=''):
        super(Accuracies, self).__init__()
        self.engine = engine
        self.mode = mode
        self.dir_exp = dir_exp
        self.dataset = engine.dataset[mode]
        self.results = None
        if self.dataset.split == 'train' or "val":
            self.accuracy = Accuracy()
        elif self.dataset.split == 'test':
            self.accuracy = Accuracy()
        else:
            self.accuracy = None

        engine.register_hook('{}_on_start_epoch'.format(mode), self.reset)
        engine.register_hook('{}_on_end_epoch'.format(mode), self.compute_accuracy)

    def reset(self):
        self.results = []

    def forward(self, net_out, batch):
        out = {}
        if self.accuracy is not None:
            out = self.accuracy(net_out, batch)
        self.results.append(out['accuracy'])
        return out


    def compute_accuracy(self):
        accuracy = float(100*np.mean(self.results))
        Logger()('Overall Accuracy is {:.2f}'.format(accuracy))
        Logger().log_value('{}_epoch.accuracy'.format(self.mode), accuracy.numpy(), should_print=False)


