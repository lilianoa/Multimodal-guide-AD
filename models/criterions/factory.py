from bootstrap.lib.options import Options
from .cross_entropy import CrossEntropyLoss
from .kdloss import CE_KDLoss, CE_MSELoss, CE_KD_MSELoss

def factory(engine, mode):
    name = Options()['model.criterion.name']
    split = engine.dataset[mode].split

    if name == 'cross_entropy':
        if split == 'test':
            return None
        criterion = CrossEntropyLoss()
    elif name == 'cross_entropy+kdloss':
        temp = Options()['model.criterion.temp']
        alpha = Options()['model.criterion.alpha']
        if split == 'test':
            return None
        criterion = CE_KDLoss(temp, alpha)
    elif name == 'cross_entropy+mseloss':
        alpha = Options()['model.criterion.alpha']
        if split == 'test':
            return None
        criterion = CE_MSELoss(alpha)
    elif name == 'cross_entropy+kd+mseloss':
        temp = Options()['model.criterion.temp']
        alpha = Options()['model.criterion.alpha']
        beta = Options()['model.criterion.beta']
        lamda = Options()['model.criterion.lamda']
        if split == 'test':
            return None
        criterion = CE_KD_MSELoss(temp, alpha, beta, lamda)

    else:
        raise ValueError(name)

    return criterion