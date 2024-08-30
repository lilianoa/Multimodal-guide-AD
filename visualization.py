import os
import sys
import traceback
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from bootstrap.lib import utils
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options
from bootstrap import engines
from bootstrap import datasets
from bootstrap import models

def visualize_feature(features, n_components=2):
    img_feat = features
    x_data = []
    cls_names = []
    y_labels_1 = []  # y_labels_1: cls
    y_labels_2 = []  # y_label_2: normal/abnormal
    for cls in img_feat.keys():
        cls_names.append(cls)
        for k in img_feat[cls].keys():
            for item in img_feat[cls][k]:
                x_data.append(item)
                y_labels_1.append(cls)
                y_labels_2.append(k)

    model = TSNE(n_components=n_components)
    feat_proj = model.fit_transform(x_data)
    X_tsne = pd.DataFrame(feat_proj).rename(columns={0:'dim1', 1:'dim2'})
    y_label_1 = pd.DataFrame(y_labels_1).rename(columns={0:'label_cls'})
    y_label_2 = pd.DataFrame(y_labels_2).rename(columns={0:'label_ab'})
    data_tsne = pd.concat([X_tsne, y_label_1, y_label_2], axis=1)
    matplotlib.use('TkAgg')
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=data_tsne, x='dim1', y='dim2', hue='label_cls', style='label_ab')
    plt.show()


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
    mode = 'test'
    epoch = engine.epoch-1

    # dataset separation
    indices = [7, 9, 10]
    obj_list = [engine.dataset['eval'].obj_list[i] for i in indices]
    cls_data = {cls: {} for cls in obj_list}
    N_Image = 20
    for cls in obj_list:
        num_normal = 0
        num_abnormal = 0
        img_feat_normal = []
        img_feat_abnormal = []
        Options()['dataset']['cls_names'] = [cls]
        engine.dataset = datasets.factory(engine)

        batch_loader = DataLoader(
            dataset=engine.dataset['eval'],
            batch_size=Options()['dataset']['batch_size'],
            shuffle=True,
            num_workers=Options()['dataset']['nb_threads'],
            pin_memory=True
        )

        for batch in batch_loader:
            with torch.no_grad():
                img_feat = engine.model.network.encode_image(batch['img'].cuda())
                label = batch['label_id']
                for idx, i in enumerate(label):
                    if i:
                        if num_abnormal < N_Image:
                            img_feat_abnormal.append(img_feat[idx].cpu().numpy())  # img_feat[0]
                            num_abnormal += 1
                    else:
                        if num_normal < N_Image:
                            img_feat_normal.append(img_feat[idx].cpu().numpy())  # img_feat[0]
                            num_normal += 1

                if num_normal == N_Image and num_abnormal == N_Image:
                    break

        cls_data[cls]['abnormal'] = img_feat_abnormal
        cls_data[cls]['normal'] = img_feat_normal

    visualize_feature(features=cls_data)


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
    main(exp_dir='./logs/visa/sca/kd_rn50', run=run)   # './logs/visa/clip_rn50'



