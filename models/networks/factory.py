import torch
from bootstrap.lib.options import Options
from bootstrap.models.networks.data_parallel import DataParallel
from .studentnet import StudentNet
from .teachernet import TeacherNet
from .studentnet_cnn import StudentNet_cnn
from .textonly import TextonlyNet

def factory(engine):
    opt = Options()['model.network']

    if opt['name'] == 'student':
        net = StudentNet(
            visual=opt['img_enc'],
            classif=opt['classif'])
    elif opt['name'] == 'teacher':
        net = TeacherNet(
            visual=opt['img_enc'],
            textual=opt['text_enc'],
            fusion=opt['fusion'],
            classif=opt['classif'])
    elif opt['name'] == 'KDstudent':
        net = {}
        s_opt = opt['studentnet']
        t_opt = opt['teachernet']
        net['student'] = StudentNet_cnn(
                visual=s_opt['img_enc'],
                classif=s_opt['classif'])
        if t_opt['name'] == 'teacher':
            net['teacher'] = TeacherNet(
                visual=t_opt['img_enc'],
                textual=t_opt['text_enc'],
                fusion=t_opt['fusion'],
                classif=t_opt['classif'])

        else:
            raise ValueError(t_opt['name'])

    elif opt['name'] == 'textonly':
        net = TextonlyNet(
                        textual=opt['text_enc'],
                        classif=opt['classif'])
    else:
        raise ValueError(opt['name'])

    if torch.cuda.device_count() > 1:
        net = DataParallel

    return net
