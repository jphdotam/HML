import os
import math
import numpy as np
import scipy.stats
from glob import glob

from torch.utils.data import DataLoader

from lib.config import load_config
from lib.datasets import ECGDataset, collate_longest_in_batch
from lib.testing import test
from lib.plots import plot_confusion_matrix

CONFIG = "./experiments/010.yaml"
MODEL_DIR = "./output/models/010"
TEST_FOLD = 4

if __name__ == "__main__":
    cfg, model_dir, vis_dir, log_dir = load_config(CONFIG)
    state_paths = glob(os.path.join(MODEL_DIR, "*.pt"))

    # Data
    ds_test = ECGDataset(cfg, 'test', fold=TEST_FOLD, final_test=True)
    dl_test = DataLoader(ds_test, 1, shuffle=False, num_workers=cfg['training']['n_workers'], pin_memory=True, collate_fn=collate_longest_in_batch)

    # Test
    targets, predicted_classes, filenames, kappa, accuracy, cm = test(state_paths, dl_test, cfg)

    targets2 = [t if t != 2 else 1 for t in targets]
    predicted_classes2 = [t if t != 2 else 1 for t in predicted_classes]

    # Write results
    with open(f"./output/results/{os.path.basename(MODEL_DIR)}.txt", 'w') as f:
        f.write(f"Cohen's Kappa: {kappa}\n")
        f.write(f"Accuracy:      {accuracy}\n")
        f.write(f"Confusion matrix:\n{cm}")

    # Sort by filename
    filenames, targets, predicted_classes = map(list, zip(*sorted(zip(filenames, targets, predicted_classes))))

    with open(f"./output/results/{os.path.basename(MODEL_DIR)}.csv", 'w') as f:
        f.write(f"case,filename,target_id,predicted_id,target_name,predicted_name\n")
        for filename, target, predicted in zip(filenames, targets, predicted_classes):
            target_name = list(cfg['data']['beat_types']['test'].keys())[int(target)]
            predicted_name = list(cfg['data']['beat_types']['test'].keys())[int(predicted)]
            case = os.path.basename(os.path.dirname(filename))
            f.write(f"{case},{os.path.basename(filename)},{int(target)},{int(predicted)},{target_name},{predicted_name}\n")

    # Kappa calculations
    true = np.array(targets)
    pred = np.array(predicted_classes)

    n_agree = np.sum(true == pred)
    p_observed = n_agree / len(true)

    # Expected for classes 0, 1, 2
    p_expected_0 = np.sum(true == 0) / len(true) * np.sum(pred == 0) / len(true)
    p_expected_1 = np.sum(true == 1) / len(true) * np.sum(pred == 2) / len(true)
    p_expected_2 = np.sum(true == 1) / len(true) * np.sum(pred == 2) / len(true)
    p_expected = p_expected_0 + p_expected_1 + p_expected_2

    # Kappa
    kappa = (p_observed - p_expected) / (1 - p_expected)
    kappa_se = math.sqrt((p_observed * (1 - p_observed)) / (len(true) * (1 - p_observed)**2))
    kappa_lci = kappa - (1.96 * kappa_se)
    kappa_uci = kappa + (1.96 * kappa_se)
    z_score = kappa/kappa_se
    p_val = scipy.stats.norm.sf(z_score)*2  # * 2 as 2 sided
    print(f"Kappa: {kappa} {kappa_lci:.5f} to {kappa_uci:.5f} (p {p_val:.8f})")

    plot_confusion_matrix(cm,
                          #classes=["Myocardial capture", "Non-selective HBP", "Selective HBP"],
                          classes=["MOC", "NS-HBP", "S-HBP"],
                          title=None)
