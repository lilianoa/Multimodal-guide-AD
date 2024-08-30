import torch
import torch.nn as nn
from modules import resnet
from modules.classifier import SimpleClassifier
import os

_MODELS_cnn = {'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'}

def factory_img_enc(opt):
    if opt['name'] == 'CNN' or 'cnn':
        if opt['type'] in _MODELS_cnn:
            model = getattr(resnet, opt['type'])(pretrained=opt['pre_trained'])
            img_enc = model
        else:
            raise ValueError("Make sure the network type is correct.")
    else:
        raise ValueError("Make sure the network name is correct.")

    return img_enc

def factory_classif(opt):
    input_dim = opt['input_dim']
    m_dim = opt['m_dim']
    num_classes = 2
    dropout = opt['dropout']
    classif = SimpleClassifier(input_dim, m_dim, num_classes, dropout)
    return classif


class StudentNet_cnn(nn.Module):
    def __init__(self,
                 visual={},
                 classif={}):
        super(StudentNet_cnn, self).__init__()
        self.visual = factory_img_enc(visual)
        self.classif = factory_classif(classif)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def forward(self, batch):  # image
        image = batch['img']
        x, feats = self.encode_image(image)
        logits = self.classif(x)
        out = {'logits': logits, 'feats': feats}
        return out
