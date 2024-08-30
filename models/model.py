import torch.nn as nn
from bootstrap.datasets import transforms
from .networks.factory import factory as net_factory
from .criterions.factory import factory as cri_factory
from .metrics.factory import factory as met_factory

class Model(nn.Module):
    """ Model contains a network, two criterions (train, eval) and two metrics.
    """

    def __init__(self,
            engine=None,
            cuda_tf=transforms.ToCuda,
            detach_tf=transforms.ToDetach,
            network=None,
            criterions={},
            metrics={}):
        super(Model, self).__init__()
        self.cuda_tf = cuda_tf
        self.detach_tf = detach_tf
        self.network = network
        self.criterions = criterions
        self.metrics = metrics
        self.is_cuda = False
        self.eval()

    def eval(self):
        """ Activate evaluation mode
        """
        super(Model, self).train(mode=False)
        self.mode = 'eval'

    def train(self):
        """ Activate training mode
        """
        super(Model, self).train(mode=True)
        self.mode = 'train'

    def cuda(self, device_id=None):
        """ Moves all model parameters and buffers to the GPU.

            Args:
                device_id (int, optional): if specified, all parameters will be
                    copied to that device
        """
        self.is_cuda = True
        return self._apply(lambda t: t.cuda(device_id))

    def cpu(self):
        """ Moves all model parameters and buffers to the CPU.
        """
        self.is_cuda = False
        return self._apply(lambda t: t.cpu())

    def prepare_batch(self, batch):
        """ Prepare a batch with two functions: cuda_tf and detach_tf (only in eval mode)
        """
        if self.is_cuda:
            batch = self.cuda_tf()(batch)
        if self.mode == 'eval':
            batch = self.detach_tf()(batch)
        return batch

    def forward(self, batch):
        """ Prepare the batch and feed it to the network, criterion and metric.

            Returns:
                out (dict): a dictionary of outputs
        """
        batch = self.prepare_batch(batch)
        net_out = self.network(batch)

        cri_out = {}
        if self.mode in self.criterions:
            cri_tmp = self.criterions[self.mode](net_out, batch)
            if cri_tmp is not None:
                cri_out = cri_tmp

        met_out = {}
        if self.mode in self.metrics:
            met_tmp = self.metrics[self.mode](net_out, batch)
            if met_tmp is not None:
                met_out = met_tmp

        out = {}
        if type(net_out) is dict:
            for key, value in net_out.items():
                out[key] = value
        if type(cri_out) is dict:
            for key, value in cri_out.items():
                out[key] = value
        if type(met_out) is dict:
            for key, value in met_out.items():
                out[key] = value
        return out

    def state_dict(self, *args, **kwgs):
        """
        """
        state = {}
        state['network'] = self.network.state_dict(*args, **kwgs)
        state['criterions'] = {}
        for mode, criterion in self.criterions.items():
            if hasattr(criterion, '__parameters'):
                state['criterions'][mode] = criterion.state_dict(*args, **kwgs)
        state['metrics'] = {}
        for mode, metric in self.metrics.items():
            if hasattr(metric, '__parameters'):
                state['metrics'][mode] = metric.state_dict(*args, **kwgs)
        return state

    def load_state_dict(self, state, *args, **kwgs):
        """
        """
        self.network.load_state_dict(state['network'], *args, **kwgs)
        for mode, criterion in self.criterions.items():
            if hasattr(criterion, '__parameters'):
                criterion.load_state_dict(state['criterions'][mode], *args, **kwgs)
        for mode, metric in self.metrics.items():
            if hasattr(metric, '__parameters'):
                metric.load_state_dict(state['metrics'][mode], *args, **kwgs)

class DefaultModel(Model):
    """ An extension of Model that relies on factory calls
    """

    def __init__(self, engine=None,
                 cuda_tf=transforms.ToCuda,
                 detach_tf=transforms.ToDetach):
        super(DefaultModel, self).__init__(
            engine=engine,
            cuda_tf=cuda_tf,
            detach_tf=detach_tf)
        self.network = self._init_network(engine=engine)
        self.criterions = self._init_criterions(engine=engine)
        self.metrics = self._init_metrics(engine=engine)
        self.eval()

    def _init_network(self, engine=None):
        """ Create the network using the bootstrap network factory
        """
        return net_factory(engine)

    def _init_criterions(self, engine=None):
        """ Create the two criterions using the bootstrap criterion factory
        """
        # by default all modes have criterions
        if engine:
            modes = list(engine.dataset.keys()) # [train, val] for mnist
        else:
            modes = ['train', 'eval']

        criterions = {}
        for mode in modes:
            tmp_cri = cri_factory(engine, mode)
            if tmp_cri is not None:
                criterions[mode] = tmp_cri
        return criterions

    def _init_metrics(self, engine=None):
        """ Create the two metrics using the bootstrap metric factory
        """
        # by default all modes have metrics
        if engine:
            modes = list(engine.dataset.keys())
        else:
            modes = ['train', 'eval']

        metrics = {}
        for mode in modes:
            tmp_met = met_factory(engine, mode)
            if tmp_met is not None:
                metrics[mode] = tmp_met
        return metrics


class KDModel(Model):

    """Specially designed for knowledge distillation
    """

    def __init__(self, engine=None,
                 cuda_tf=transforms.ToCuda,
                 detach_tf=transforms.ToDetach):
        super(KDModel, self).__init__(
            engine=engine,
            cuda_tf=cuda_tf,
            detach_tf=detach_tf)
        self.network, self.teachernet = self._init_networks(engine=engine)
        self.criterions = self._init_criterions(engine=engine)
        self.metrics = self._init_metrics(engine=engine)
        self.eval()

    def _init_networks(self, engine=None):
        """ Create the network using the bootstrap network factory
        """
        net = net_factory(engine)
        if type(net) is dict:
            studentnet = net['student']
            teachernet = net['teacher']
            for param in teachernet.parameters():
                param.requires_grad = False
        else:
            raise ValueError("The network is not a knowledge distillation model. Make sure the network name is correct.")
        return studentnet, teachernet

    def _init_criterions(self, engine=None):
        """ Create the two criterions using the bootstrap criterion factory
        """
        # by default all modes have criterions
        if engine:
            modes = list(engine.dataset.keys()) # [train, val] for mnist
        else:
            modes = ['train', 'eval']

        criterions = {}
        for mode in modes:
            tmp_cri = cri_factory(engine, mode)
            if tmp_cri is not None:
                criterions[mode] = tmp_cri
        return criterions

    def _init_metrics(self, engine=None):
        """ Create the two metrics using the bootstrap metric factory
        """
        # by default all modes have metrics
        if engine:
            modes = list(engine.dataset.keys())
        else:
            modes = ['train', 'eval']

        metrics = {}
        for mode in modes:
            tmp_met = met_factory(engine, mode)
            if tmp_met is not None:
                metrics[mode] = tmp_met
        return metrics


    def forward(self, batch):
        """ Prepare the batch and feed it to the network, criterion and metric.

            Returns:
                out (dict): a dictionary of outputs
        """
        batch = self.prepare_batch(batch)

        net_out = self.network(batch)
        teacher_out = self.teachernet(batch)
        teacher_out['logits'] = teacher_out['logits'].detach()
        '''
        for i, f in enumerate(teacher_out['feats']):
            teacher_out['feats'][i] = f.detach()
        '''
        '''
        for f in net_out['feats']:
            print("student feats", f.size())
        for f in teacher_out['feats']:
            print("teacher feats", f.size())
        '''


        cri_out = {}
        if self.mode in self.criterions:
            cri_tmp = self.criterions[self.mode](net_out, teacher_out, batch)
            if cri_tmp is not None:
                cri_out = cri_tmp


        met_out = {}
        if self.mode in self.metrics:
            met_tmp = self.metrics[self.mode](net_out, batch)
            if met_tmp is not None:
                met_out = met_tmp

        out = {}
        if type(net_out) is dict:
            for key, value in net_out.items():
                out[key] = value
        if type(cri_out) is dict:
            for key, value in cri_out.items():
                out[key] = value
        if type(met_out) is dict:
            for key, value in met_out.items():
                out[key] = value
        return out

    def state_dict(self, *args, **kwgs):
        """
        """
        state = {}
        state['network'] = self.network.state_dict(*args, **kwgs)
        state['teachernet'] = self.teachernet.state_dict(*args, **kwgs)
        state['criterions'] = {}
        for mode, criterion in self.criterions.items():
            if hasattr(criterion, '__parameters'):
                state['criterions'][mode] = criterion.state_dict(*args, **kwgs)
        state['metrics'] = {}
        for mode, metric in self.metrics.items():
            if hasattr(metric, '__parameters'):
                state['metrics'][mode] = metric.state_dict(*args, **kwgs)
        return state

    def load_state_dict(self, state, *args, **kwgs):
        """
        """
        self.network.load_state_dict(state['network'], *args, **kwgs)
        self.teachernet.load_state_dict(state['teachernet'], *args, **kwgs)
        for mode, criterion in self.criterions.items():
            if hasattr(criterion, '__parameters'):
                criterion.load_state_dict(state['criterions'][mode], *args, **kwgs)
        for mode, metric in self.metrics.items():
            if hasattr(metric, '__parameters'):
                metric.load_state_dict(state['metrics'][mode], *args, **kwgs)


    def net_state_dict(self, *args, **kwgs):
        """
        """
        state = {}
        state['network'] = self.network.state_dict(*args, **kwgs)
        state['teachernet'] = self.teachernet.state_dict(*args, **kwgs)
        state['criterions'] = {}
        for mode, criterion in self.criterions.items():
            if hasattr(criterion, '__parameters'):
                state['criterions'][mode] = criterion.state_dict(*args, **kwgs)
        state['metrics'] = {}
        for mode, metric in self.metrics.items():
            if hasattr(metric, '__parameters'):
                state['metrics'][mode] = metric.state_dict(*args, **kwgs)
        return state

    def net_load_state_dict(self, state, *args, **kwgs):
        """
        """
        self.network.load_state_dict(state['network'], *args, **kwgs)
        self.teachernet.load_state_dict(state['teachernet'], *args, **kwgs)
        for mode, criterion in self.criterions.items():
            if hasattr(criterion, '__parameters'):
                criterion.load_state_dict(state['criterions'][mode], *args, **kwgs)
        for mode, metric in self.metrics.items():
            if hasattr(metric, '__parameters'):
                metric.load_state_dict(state['metrics'][mode], *args, **kwgs)