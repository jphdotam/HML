import os
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import savgol_filter
from matplotlib.colors import LinearSegmentedColormap

import torch

from lib.datasets import LEADS
from lib.models import load_model

blues_cdict = {'red': [(0.0, 1, 1), (0.125, 0.8705882352941177, 0.8705882352941177),
                       (0.25, 0.7764705882352941, 0.7764705882352941),
                       (0.375, 0.6196078431372549, 0.6196078431372549),
                       (0.5, 0.4196078431372549, 0.4196078431372549),
                       (0.625, 0.25882352941176473, 0.25882352941176473),
                       (0.75, 0.12941176470588237, 0.12941176470588237),
                       (0.875, 0.03137254901960784, 0.03137254901960784),
                       (1.0, 0.03137254901960784, 0.03137254901960784)],
               'green': [(0.0, 1, 1), (0.125, 0.9215686274509803, 0.9215686274509803),
                         (0.25, 0.8588235294117647, 0.8588235294117647),
                         (0.375, 0.792156862745098, 0.792156862745098),
                         (0.5, 0.6823529411764706, 0.6823529411764706),
                         (0.625, 0.5725490196078431, 0.5725490196078431),
                         (0.75, 0.44313725490196076, 0.44313725490196076),
                         (0.875, 0.3176470588235294, 0.3176470588235294),
                         (1.0, 0.18823529411764706, 0.18823529411764706)],
               'blue': [(0.0, 1.0, 1.0), (0.125, 0.9686274509803922, 0.9686274509803922),
                        (0.25, 0.9372549019607843, 0.9372549019607843),
                        (0.375, 0.8823529411764706, 0.8823529411764706),
                        (0.5, 0.8392156862745098, 0.8392156862745098),
                        (0.625, 0.7764705882352941, 0.7764705882352941),
                        (0.75, 0.7098039215686275, 0.7098039215686275),
                        (0.875, 0.611764705882353, 0.611764705882353),
                        (1.0, 0.4196078431372549, 0.4196078431372549)],
               'alpha': [(0.0, 1, 1), (0.125, 1, 1), (0.25, 1, 1), (0.375, 1, 1), (0.5, 1, 1), (0.625, 1, 1),
                         (0.75, 1, 1), (0.875, 1, 1), (1.0, 1, 1)]}
cmap = LinearSegmentedColormap('cm_map', segmentdata=blues_cdict, N=256)


def plot_ecg(ecg, title, channels=(0, 12), n_cols=2, figsize=(4, 12), channel_names=LEADS, saliency=None, savepath=None, smooth_saliency=True):
    channels = list(range(*channels))
    n_rows = math.ceil(len(channels)/n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex='col')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    fig.suptitle(title)

    # Plot ECG
    for i_channel in channels:
        col = i_channel//n_rows
        row = i_channel%n_rows
        ax = axes[row][col]
        ax.plot(ecg[i_channel])
        ax.title.set_text(channel_names[i_channel])
        ax.set_axis_off()
        if saliency is not None:
            if saliency.shape[0] == len(channels):
                sal = np.expand_dims(saliency[i_channel], axis=0)
                if smooth_saliency:
                    sal = savgol_filter(sal, 31, 3)
                ax.pcolorfast(ax.get_xlim(), ax.get_ylim(), sal, cmap=cmap, alpha=0.3)
            else:
                raise NotImplementedError("Not yet implemented lead-averaged saliency")

    # Save
    if savepath:
        fig.savefig(savepath)
    return fig, axes


def vis(statepaths, dataloader, experiment_id, cfg):
    device = cfg['training']['device']
    n_folds = cfg['data']['n_folds']
    assert len(statepaths) == (n_folds - 1), f"If using a hold out test set, we should have n_folds-1 statepaths"

    model = load_model(cfg, load_model_only=True)
    model = model.to(device)

    for i_fold, state in enumerate(tqdm(statepaths)):
        # Load model for fold
        state = torch.load(state)
        model.module.load_state_dict(state['model'])
        model.eval()

        # Test model
        for i_batch, (x, y_true, filename) in enumerate(dataloader):
            assert len(x) == 1, "batch size for testing should be 1"

            x = x.to(device, non_blocking=True)
            x.requires_grad_()  # Specify we want gradient back to original image, not just first conv layer
            y_pred = model(x)

            cls_pred = y_pred.argmax().detach()  # If we don't detach we get a backprop error on pytorch 1.6
            score_pred = y_pred[0, cls_pred]  # Activation of highest class, element 0 (BS 1)
            score_pred.backward()

            saliency_eachlead = x.grad.data.abs()[0]
            saliency_allleads, _ = torch.max(x.grad.data.abs()[0], dim=0)

            # Standardise saliency maps between 0 and 1
            saliency_eachlead = ((saliency_eachlead - saliency_eachlead.min()) / saliency_eachlead.max()).detach().cpu().numpy()
            saliency_allleads = ((saliency_allleads - saliency_allleads.min()) / saliency_allleads.max()).detach().cpu().numpy()

            # Plot
            filename = filename[0]
            classid_true, classid_pred = int(y_true[0]), int(cls_pred)
            classname_true, classname_pred = cfg['data']['beat_types'][classid_true], cfg['data']['beat_types'][classid_pred]
            correct = 'CORRECT' if classname_true == classname_pred else 'INCORRECT'

            title = f"Case {os.path.basename(os.path.dirname(filename))}\n" \
                    f"Ground truth:{classname_true}\n" \
                    f"Predicted:{classname_pred}"

            savepath = f"./output/vis/{experiment_id}/{os.path.basename(filename).split('.')[0]}_{correct}.png"
            plot_ecg(x[0].detach().cpu().numpy(), title=title, saliency=saliency_eachlead, savepath=savepath)