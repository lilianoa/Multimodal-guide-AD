import os
import sys
import traceback
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options
from bootstrap import engines
from bootstrap import datasets
from bootstrap import models
from tabulate import tabulate

from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

# metrics
def image_level_metrics(results, obj, metric):
    gt = results[obj]['gt_sp']
    pr = results[obj]['pr_sp']
    gt = np.array(gt)
    pr = np.array(pr)
    if metric == 'image-auroc':
        performance = roc_auc_score(gt, pr)
    elif metric == 'image-ap':
        performance = average_precision_score(gt, pr)

    return performance


def run(exp_dir=None):

    path_opts = os.path.join(exp_dir, 'options.yaml')
    # first call to Options() load the options yaml file from --path_opts command line argument if path_opts=None
    Options(path_opts)

    if torch.cuda.is_available():
        cudnn.benchmark = True

    # engine can save and load the model and optimizer
    engine = engines.factory()

    # dataset is a dictionary that contains all the needed datasets indexed by modes
    # (example: dataset.keys() -> ['train','eval'])
    engine.dataset = datasets.factory(engine)

    # model includes a network, a criterion and a metric
    # model can register engine hooks (begin epoch, end batch, end batch, etc.)
    # (example: "calculate mAP at the end of the evaluation epoch")
    # note: model can access to datasets using engine.dataset

    engine.model = models.factory(engine)

    Options()['dataset']['train_split'] = None
    # load the model from a checkpoint
    if Options()['exp']['resume']:
        engine.resume()

    engine.model.eval()

    obj_list = engine.dataset['eval'].obj_list
    results = {}
    for obj in obj_list:
        results[obj] = {}
        results[obj]['gt_sp'] = []
        results[obj]['pr_sp'] = []


        Options()['dataset']['cls_names'] = [obj]
        engine.dataset = datasets.factory(engine)
        dataloader = DataLoader(
            dataset=engine.dataset['eval'],
            batch_size=Options()['dataset']['batch_size'],
            shuffle=False,
            num_workers=Options()['dataset']['nb_threads'],
            pin_memory=True
        )
        for batch in dataloader:
            label = batch['label_id']
            batch['img'] = batch['img'].cuda()
            results[obj]['gt_sp'].extend(label)
            with torch.no_grad():
                net_out = engine.model.network(batch)
                logits = net_out['logits']

            probs = logits.softmax(-1)
            results[obj]['pr_sp'].extend(probs[..., -1].detach().cpu().numpy())

    table_ls = []
    image_auroc_list = []
    image_ap_list = []
    for obj in obj_list:
        table = []
        table.append(obj)
        image_auroc = image_level_metrics(results, obj, "image-auroc")
        image_ap = image_level_metrics(results, obj, "image-ap")
        print("------------", obj, "-------------")
        print("image-auroc", image_auroc)
        print("image-ap", image_ap)
        table.append(str(np.round(image_auroc * 100, decimals=1)))
        table.append(str(np.round(image_ap * 100, decimals=1)))
        image_auroc_list.append(image_auroc)
        image_ap_list.append(image_ap)
        table_ls.append(table)

    # logger
    table_ls.append(['mean',
                     str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                     str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
    results = tabulate(table_ls, headers=['objects', 'image_auroc', 'image_ap'], tablefmt="pipe")
    Logger()("\n", results)


def reset_options_instance():
    Options._Options__instance = None
    Logger._instance = None
    sys.argv = [sys.argv[0]]   # reset command line args

def main(exp_dir=None, run=None):
    reset_options_instance()
    sys.argv += [
        '--dataset.eval_split', 'test',
        '--misc.cuda', 'True',
        '--exp.resume', 'last'  # best_accuracy
    ]
    try:
        run(exp_dir=exp_dir)
    # to avoid traceback for -h flag in arguments line
    except SystemExit:
        pass
    except:
        # to be able to write the error trace to exp_dir/logs.txt
        try:
            Logger()(traceback.format_exc(), Logger.ERROR)
        except:
            pass


if __name__ == '__main__':
    main(exp_dir='./logs/visa/sca/kd_rn50', run=run)  # './logs/visa/clip_rn50'
