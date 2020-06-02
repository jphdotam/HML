import torch
import torch.nn as nn

from lib.resnet import resnet50_1d, resnet18_1d, resnet34_1d


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
        if self.arch == 'resnet18_1d':
            return resnet18_1d(n_inputs, self.n_classes)
        elif self.arch == 'resnet34_1d':
            return resnet34_1d(n_inputs, self.n_classes)
        elif self.arch == 'resnet50_1d':
            return resnet50_1d(n_inputs, self.n_classes)

    def forward(self, x):
        return self.model(x)
