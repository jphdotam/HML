import os
import torch
from collections import deque
from sklearn.metrics import cohen_kappa_score, accuracy_score


class Am:
    "Simple average meter which stores progress as a running average"

    def __init__(self, n_for_running_average=100):  # n is in samples not batches
        self.n_for_running_average = n_for_running_average
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.running = deque(maxlen=self.n_for_running_average)
        self.running_average = -1

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.running.extend([val] * n)
        self.count += n
        self.avg = self.sum / self.count
        self.running_average = sum(self.running) / len(self.running)

def cycle(train_or_test, model, dataloader, epoch, criterion, optimizer, cfg, scheduler=None, writer=None):
    n_classes_train = len(set(cfg['data']['beat_types']['train'].values()))
    n_classes_test = len(set(cfg['data']['beat_types']['test'].values()))

    log_freq = cfg['output']['log_freq']
    device = cfg['training']['device']
    meter_loss = Am()
    preds, targets = [], []

    model = model.to(device)

    if train_or_test == 'train':
        model.train()
        training = True
    elif train_or_test == 'test':
        model.eval()
        training = False

    else:
        raise ValueError(f"train_or_test must be 'train', or 'test', not {train_or_test}")

    for i_batch, (x, y_true, _filename) in enumerate(dataloader):
        # Forward pass
        optimizer.zero_grad()

        x = x.to(device, non_blocking=True)
        y_true = y_true.to(device, non_blocking=True)

        # Forward pass
        if training:
            y_pred = model(x)
            if n_classes_train != n_classes_test:
                y_pred = y_pred[:, :n_classes_train]
            loss = criterion(y_pred, y_true)
        else:
            with torch.no_grad():
                y_pred = model(x)
                if n_classes_train != n_classes_test:
                    y_pred = y_pred[:, :n_classes_test]
                loss = criterion(y_pred, y_true)

        # Backward pass
        if training:
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

        meter_loss.update(loss, x.size(0))

        # For Cohen's Kappa
        preds.append(y_pred.detach().float().cpu())  # Cast as float as doesn't work otherwise with mixed precision
        targets.append(y_true.detach().cpu())

        # Loss intra-epoch printing
        if (i_batch+1) % log_freq == 0:
            p = torch.argmax(torch.cat(preds,0),1)
            t = torch.cat(targets)
            kappa = cohen_kappa_score(t, p)

            print(f"{train_or_test.upper(): >5} [{i_batch+1:04d}/{len(dataloader):04d}]"
                  f"\t\tLOSS: {meter_loss.running_average:.5f}\t\tKAPPA: {kappa:.5f}")

            if train_or_test == 'train':
                i_iter = ((epoch - 1) * len(dataloader)) + i_batch+1
                writer.add_scalar(f"LossIter/{train_or_test}", meter_loss.running_average, i_iter + 1)
                writer.add_scalar(f"KappaIter/{train_or_test}", kappa, i_iter + 1)

    loss = float(meter_loss.avg.detach().cpu().numpy())
    p = torch.argmax(torch.cat(preds,0),1)
    t = torch.cat(targets)
    kappa = cohen_kappa_score(t, p)
    accuracy = accuracy_score(t, p)

    print(f"{train_or_test.upper(): >5} Complete!"
          f"\t\t\tLOSS: {meter_loss.avg:.5f}\t\tKAPPA: {kappa:.5f}\t\tACCU: {accuracy:.5f}")

    if writer:
        writer.add_scalar(f"LossEpoch/{train_or_test}", loss, epoch)
        writer.add_scalar(f"KappaEpoch/{train_or_test}", kappa, epoch)

    return loss, kappa


def save_state(state, save_path, test_metric, best_metric, cfg, last_save_path, lowest_best=True):
    save = cfg['output']['save']
    if save == 'all':
        torch.save(state, save_path)
    elif (test_metric < best_metric) == lowest_best:
        print(f"{test_metric:.5f} better than {best_metric:.5f} -> SAVING")
        if save == 'best':  # Delete previous best if using best only; otherwise keep previous best
            if last_save_path:
                try:
                    os.remove(last_save_path)
                except FileNotFoundError:
                    print(f"Failed to find {last_save_path}")
        best_metric = test_metric
        torch.save(state, save_path)
        last_save_path = save_path
    else:
        print(f"{test_metric:.5g} not improved from {best_metric:.5f}")
    return best_metric, last_save_path
