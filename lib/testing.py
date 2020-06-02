import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix

import torch

from lib.models import load_model


def test(statepaths, dataloader, cfg):
    n_classes_test = len(set(cfg['data']['beat_types']['test'].values()))

    device = cfg['training']['device']
    n_test = len(dataloader)
    n_folds = cfg['data']['n_folds']
    assert len(statepaths) == (n_folds-1), f"If using a hold out test set, we should have n_folds-1 statepaths"

    predictions = np.zeros((n_test, n_classes_test, n_folds-1))
    targets = np.zeros(n_test)
    filenames = []

    model = load_model(cfg, load_model_only=True)
    model = model.to(device)

    for i_fold, state in enumerate(tqdm(statepaths)):
        # Load model for fold
        state = torch.load(state)
        if cfg['training']['data_parallel']:
            model.module.load_state_dict(state['model'])
        else:
            model.load_state_dict(state['model'])
        model.eval()

        # Test model
        for i_batch, (x, y_true, filename) in enumerate(dataloader):
            assert len(x) == 1, "batch size for testing should be 1"

            x = x.to(device, non_blocking=True)
            y_true = y_true.to(device, non_blocking=True)

            with torch.no_grad():
                y_pred = model(x)
                y_pred = y_pred[:, :n_classes_test]

            predictions[i_batch, :, i_fold] = y_pred[0].detach().float().cpu()
            if i_fold == 0:
                targets[i_batch] = y_true[0].detach().cpu()
                filenames.append(filename[0])

    ensembled_preds = np.mean(predictions, axis=-1)
    predicted_classes = np.argmax(ensembled_preds, axis=-1)
    kappa = cohen_kappa_score(targets, predicted_classes)
    accuracy = accuracy_score(targets, predicted_classes)
    cm = confusion_matrix(targets, predicted_classes)

    print(f"Cohen's Kappa: {kappa}")
    print(f"Accuracy:      {accuracy}")
    print(f"Confusion matrix:\n{cm}")

    return targets, predicted_classes, filenames, kappa, accuracy, cm