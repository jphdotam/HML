import os
from glob import glob

from torch.utils.data import DataLoader

from lib.config import load_config
from lib.datasets import ECGDataset, collate_longest_in_batch
from lib.vis import vis

# tmp
import torch
from lib.models import load_model
from tqdm import tqdm
#

EXPERIMENT_ID = '010'
CONFIG = f"./experiments/{EXPERIMENT_ID}.yaml"
MODEL_DIR = f"./output/models/{EXPERIMENT_ID}"
TEST_FOLD = 4

if __name__ == "__main__":
    cfg, model_dir, vis_dir, log_dir = load_config(CONFIG)
    state_paths = glob(os.path.join(MODEL_DIR, "*.pt"))

    # Data
    ds_test = ECGDataset(cfg, 'test', fold=TEST_FOLD)
    dl_test = DataLoader(ds_test, 1, shuffle=False, num_workers=cfg['training']['n_workers'], pin_memory=True, collate_fn=collate_longest_in_batch)

    # Vis
    vis(state_paths, dl_test, EXPERIMENT_ID, cfg)

