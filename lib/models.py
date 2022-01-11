import torch
import torch.nn as nn

from lib.resnet import resnet18_1d, resnet34_1d, resnet50_1d, resnet101_1d, resnet152_1d, wide_resnet50_1d, wide_resnet101_1d, resnext50_32x4d_1d, resnext101_32x8d_1d
from lib.densenet import densenet121_1d, densenet161_1d, densenet169_1d, densenet201_1d
from lib.hannunnet import HannunNet
from lib.rnn import RNN_LSTM, RNN_GRU


def load_model(cfg, load_model_only=False):
    model = ECGModel(cfg)
    dp = cfg['training']['data_parallel']

    if dp:
        model = nn.DataParallel(model).to(cfg['training']['device'])
        m = model.module
    else:
        m = model

    if load_model_only:
        return model

    if modelpath := cfg['resume'].get('path', None):
        state = torch.load(modelpath)
        m.load_state_dict(state['state_dict'])
        starting_epoch = state['epoch']
        if conf_epoch := cfg['resume'].get('epoch', None):
            print(
                f"WARNING: Loaded model trained for {starting_epoch - 1} epochs but config explicitly overrides to {conf_epoch}")
            starting_epoch = conf_epoch
    else:
        starting_epoch = 1
        state = {}

    return model, starting_epoch, state


class ECGModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        n_classes_train = len(cfg['data']['beat_types']['train'])
        n_classes_test = len(cfg['data']['beat_types']['test'])
        self.n_classes = max(n_classes_train, n_classes_test)
        self.arch = cfg['training']['arch']

        if n_classes_train != n_classes_test:
            print(f"NB Network has {self.n_classes} output classes, as training data has {n_classes_train} classes but test has {n_classes_test}")

        self.model = self.load_model()

    def load_model(self):
        n_inputs = 24 if self.cfg['data']['stack_intrinsic'] else 12
        dropout = self.cfg['training'].get('dropout', False)
        if self.arch == 'resnet18_1d':
            return resnet18_1d(n_inputs, self.n_classes, dropout=dropout)
        elif self.arch == 'resnet34_1d':
            return resnet34_1d(n_inputs, self.n_classes, dropout=dropout)
        elif self.arch == 'resnet50_1d':
            return resnet50_1d(n_inputs, self.n_classes, dropout=dropout)
        elif self.arch == 'resnet101_1d':
            return resnet101_1d(n_inputs, self.n_classes, dropout=dropout)
        elif self.arch == 'resnet152_1d':
            return resnet152_1d(n_inputs, self.n_classes, dropout=dropout)

        elif self.arch == 'wide_resnet50_1d':
            return wide_resnet50_1d(n_inputs, self.n_classes, dropout=dropout)
        elif self.arch == 'wide_resnet101_1d':
            return wide_resnet101_1d(n_inputs, self.n_classes, dropout=dropout)

        elif self.arch == 'resnext50_32x4d_1d':
            return resnext50_32x4d_1d(n_inputs, self.n_classes, dropout=dropout)
        elif self.arch == 'resnext101_32x8d_1d':
            return resnext101_32x8d_1d(n_inputs, self.n_classes, dropout=dropout)

        elif self.arch == 'densenet121':
            return densenet121_1d(n_inputs, self.n_classes)
        elif self.arch == 'densenet161':
            return densenet161_1d(n_inputs, self.n_classes)
        elif self.arch == 'densenet169':
            return densenet169_1d(n_inputs, self.n_classes)
        elif self.arch == 'densenet201':
            return densenet201_1d(n_inputs, self.n_classes)

        elif self.arch == 'hannunnet':
            return HannunNet(n_inputs, self.n_classes)

        elif self.arch == 'rnn_lstm':
            return RNN_LSTM(n_inputs, self.n_classes)
        elif self.arch == 'rnn_gru':
            return RNN_GRU(n_inputs, self.n_classes)

        else:
            raise ValueError()

    def forward(self, x):
        return self.model(x)
