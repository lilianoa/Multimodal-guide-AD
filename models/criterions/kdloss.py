import torch
import torch.nn as nn
import torch.nn.functional as F

class KDLoss(nn.Module):

    def __init__(self, temp):
        super(KDLoss, self).__init__()
        self.temp = temp
        self.loss = nn.KLDivLoss()

    def forward(self, net_out, teacher_out):
        out = self.loss(
            F.log_softmax(net_out / self.temp, dim=1),
            F.softmax(teacher_out / self.temp, dim=1)) * (self.temp * self.temp)
        return out


class CE_KDLoss(nn.Module):
    def __init__(self, temp, alpha):
        super(CE_KDLoss, self).__init__()
        self.temp = temp
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kd_loss = KDLoss(self.temp)

    def forward(self, net_out, teacher_out, batch):
        out = {}
        ce_loss = self.ce_loss(net_out['logits'], batch['label_id'])
        out['ce_loss'] = ce_loss
        kd_loss = self.kd_loss(net_out['logits'], teacher_out['logits'])
        out['kd_loss'] = kd_loss
        out['loss'] = (1-self.alpha) * ce_loss + self.alpha * kd_loss
        return out

class CE_MSELoss(nn.Module):
    def __init__(self, alpha):
        super(CE_MSELoss, self).__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, net_out, teacher_out, batch):
        out = {}
        mse_loss = 0
        ce_loss = self.ce_loss(net_out['logits'], batch['label_id'])
        out['ce_loss'] = ce_loss
        for i, f in enumerate(net_out['feats']):
            '''
            print("student feats", f.size())
            print("student feats requires_grad", f.requires_grad)
            print("teacher feats", teacher_out['feats'][i+1].size())
            print("teacher feats requires_grad", teacher_out['feats'][i+1].requires_grad)
            '''
            mse_loss += self.mse_loss(f, teacher_out['feats'][i+1])
        mse_loss = mse_loss / len(net_out['feats'])
        out['mse_loss'] = mse_loss
        out['loss'] = (1-self.alpha) * ce_loss + self.alpha * mse_loss
        return out

class CE_KD_MSELoss(nn.Module):
    def __init__(self, temp, alpha, beta, lamda):
        super(CE_KD_MSELoss, self).__init__()
        self.temp = temp
        self.alpha = alpha
        self.beta = beta
        self.lamda = lamda
        self.ce_loss = nn.CrossEntropyLoss()
        self.kd_loss = KDLoss(self.temp)
        self.mse_loss = nn.MSELoss()

    def forward(self, net_out, teacher_out, batch):
        out = {}
        mse_loss = 0
        ce_loss = self.ce_loss(net_out['logits'], batch['label_id'])
        out['ce_loss'] = ce_loss
        kd_loss = self.kd_loss(net_out['logits'], teacher_out['logits'])
        out['kd_loss'] = kd_loss
        for i, f in enumerate(net_out['feats']):
            mse_loss += self.mse_loss(f, teacher_out['feats'][i + 1])
        mse_loss = mse_loss / len(net_out['feats'])
        out['mse_loss'] = mse_loss
        out['loss'] = self.lamda * ce_loss + self.alpha * kd_loss + self.beta * mse_loss
        return out