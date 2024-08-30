import os
from bootstrap.lib.options import Options
from datasets.dataset import GasDataset, ADDataset
from modules.clip.clip import _transform

def factory(engine=None):
    opt = Options()['dataset']
    dataset = {}  # datasets is a dictionary that contains all the needed datasets indexed by modes, example: train, eval
    if opt.get('train_split', None):
        dataset['train'] = factory_split(opt['train_split'])
    if opt.get('eval_split', None):
        dataset['eval'] = factory_split(opt['eval_split'])
    return dataset

def factory_split(split):
    opt = Options()['dataset']
    dataroot = opt['dir']
    assert opt['name'] in ['Gas', 'AD']
    transform = _transform(224)
    if opt['name'] == 'Gas':
        dataset = GasDataset(split=split, dataroot=dataroot, transform=transform)
    elif opt['name'] == 'AD':
        dataset = ADDataset(split=split, dataroot=dataroot, transform=transform, class_names=opt['cls_names'])
    else:
        raise RuntimeError(f"Dataset {opt['name']} not found.")
    return dataset


