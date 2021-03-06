import os

from torch.utils.data import DataLoader

from lib.models import load_model
from lib.config import load_config
from lib.losses import load_criterion
from lib.optimizers import load_optimizer
from lib.training import cycle, save_state
from lib.logging import get_summary_writer
from lib.datasets import ECGDataset, collate_longest_in_batch

CONFIG = "./experiments/001.yaml"

if __name__ == "__main__":
    cfg, model_dir, vis_dir, log_dir = load_config(CONFIG)

    for fold in range(1,cfg['data']['n_folds']):

        # Data
        ds_train = ECGDataset(cfg, 'train', fold)
        ds_test = ECGDataset(cfg, 'test', fold)
        dl_train = DataLoader(ds_train, cfg['training']['batch_size'], shuffle=True, num_workers=cfg['training']['n_workers'], pin_memory=True, collate_fn=collate_longest_in_batch)
        dl_test = DataLoader(ds_test, cfg['training']['batch_size'], shuffle=False, num_workers=cfg['training']['n_workers'], pin_memory=True, collate_fn=collate_longest_in_batch)

        # Model
        model, starting_epoch, state = load_model(cfg)
        optimizer, scheduler = load_optimizer(model, cfg, state, steps_per_epoch=(len(dl_train)))
        train_criterion, test_criterion = load_criterion(cfg)

        # Train
        writer = get_summary_writer(cfg, log_dir)
        best_loss, best_path, last_save_path = 1e10, None, None
        n_epochs = cfg['training']['n_epochs']

        for epoch in range(starting_epoch, n_epochs + 1):
            print(f"\nEpoch {epoch} of {n_epochs}")

            # Cycle
            train_loss, train_kappa = cycle('train', model, dl_train, epoch, train_criterion, optimizer, cfg, scheduler, writer)
            test_loss, test_kappa = cycle('test', model, dl_test, epoch, test_criterion, optimizer, cfg, scheduler, writer)

            # Save state if required
            model_weights = model.module.state_dict() if cfg['training']['data_parallel'] else model.state_dict()
            state = {'epoch': epoch + 1,
                     'model': model_weights,
                     'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler}
            save_path = os.path.join(model_dir, f"f{fold}_{epoch}_{test_kappa:.05f}.pt")
            best_loss, last_save_path = save_state(state, save_path, test_loss, best_loss, cfg, last_save_path, lowest_best=True)