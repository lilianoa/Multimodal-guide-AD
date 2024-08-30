from bootstrap.lib.options import Options
from .accuracies import Accuracies
from .accuracy import Accuracy
from .metric_test import Metric_test

def factory(engine, mode):
    name = Options()['model.metric.name']

    metric = None
    if name == 'accuracies':
        if mode == 'train':
            split = engine.dataset[mode].split
            if split == 'train':
                metric = Accuracy()
            elif split == 'trainval':
                metric = None
            else:
                raise ValueError(split)
        elif mode == 'eval':
            split = engine.dataset[mode].split
            if split == 'val':
                metric = Accuracy()
            elif split == 'test':
                metric = Accuracy()
        else:
            metric = None

    else:
        raise ValueError(name)
    return metric



