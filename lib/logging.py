import os
from torch.utils.tensorboard import SummaryWriter


def get_summary_writer(cfg, log_dir):
    if not cfg['output']['use_tensorboard']:
        return None
    else:
        return SummaryWriter(log_dir=log_dir)
