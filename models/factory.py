from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger

from .model import DefaultModel,KDModel

def factory(engine=None):
    if Options()['model']['name'] == 'default':
        model = DefaultModel(engine)
    elif Options()['model']['name'] == 'KDmodel':
        model = KDModel(engine)
    else:
        raise ValueError()

    # TODO
    # if data_parallel is not None:
    #     if not cuda:
    #         raise ValueError
    #     model = nn.DataParallel(model).cuda()
    #     model.save = lambda x: x.module.save()

    if Options()['misc']['cuda']:
        model.cuda()

    return model