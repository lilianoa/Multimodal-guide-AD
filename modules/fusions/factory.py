import copy
from bootstrap.lib.logger import Logger
from .bilinearfusions import ConcatMLP, LinearSum, MLB, Mutan, Block
from .SAFfusion import SAF
from .BAFfusion import BAF

def factory(opt):
    opt = copy.copy(opt)
    ftype = opt.pop('type', None) # rm type from dict

    if ftype == 'concat':
        fusion = ConcatMLP(**opt)
    elif ftype == 'sum':
        fusion = LinearSum(**opt)
    elif ftype == 'mlb':
        fusion = MLB(**opt)
    elif ftype == 'mutan':
        fusion = Mutan(**opt)
    elif ftype == 'block':
        fusion = Block(**opt)
    elif ftype == 'saf':
        fusion = SAF(**opt)
    elif ftype == 'baf':
        fusion = BAF(**opt)
    else:
        raise ValueError()

    Logger().log_value('nb_params_fusion', fusion.n_params, should_print=True)
    return fusion
