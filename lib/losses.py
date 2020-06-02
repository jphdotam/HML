import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Thanks to"""
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return self.linear_combination(loss / n, nll, self.epsilon)

    @staticmethod
    def reduce_loss(loss, reduction='mean'):
        return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

    @staticmethod
    def linear_combination(x, y, epsilon):
        return epsilon * x + (1 - epsilon) * y


class OHEMCrossEntropy(nn.Module):
    """Combination of above & written by JH"""
    def __init__(self, ohem_rate=0.9, labelsmoothing_epsilon=0.1, reduction='mean'):
        print(f"Using OHEM crossentropy with OHEM rate {ohem_rate} and LS epsilon of {labelsmoothing_epsilon}")
        super().__init__()
        self.ohem_rate = ohem_rate
        self.labelsmoothing_epsilon = labelsmoothing_epsilon
        self.reduction = reduction

    def forward(self, preds, targets):
        n = preds.size()[-1]
        elementwise_losses = F.cross_entropy(preds, targets, reduction='none', ignore_index=-1)
        sorted_losses, idx = torch.sort(elementwise_losses, descending=True)
        keep_num = min(sorted_losses.size()[0], int(len(preds) * self.ohem_rate))
        keep_ids = idx[:keep_num]

        preds = preds[keep_ids]
        targets = targets[keep_ids]

        if self.labelsmoothing_epsilon:
            log_preds = F.log_softmax(preds, dim=-1)
            loss = self.reduce_loss(-log_preds.sum(dim=-1), self.reduction)
            nll = F.nll_loss(log_preds, targets, reduction=self.reduction)
            return self.linear_combination(loss / n, nll, self.labelsmoothing_epsilon)

        else:
            elementwise_losses = elementwise_losses[keep_ids]
            loss = elementwise_losses.sum() / keep_num
            return loss

    @staticmethod
    def reduce_loss(loss, reduction='mean'):
        return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

    @staticmethod
    def linear_combination(x, y, epsilon):
        return epsilon * x + (1 - epsilon) * y


def load_criterion(cfg):
    def get_criterion(name, class_weights=False, label_smoothing=False, ohem_rate=None):

        if class_weights:
            class_weights = torch.tensor(class_weights).float().to(cfg['training']['device'])

        if name == 'crossentropy':
            assert not class_weights or not label_smoothing, "Can't specify label smoothing and class weights"
            if label_smoothing:
                return LabelSmoothingCrossEntropy(epsilon=label_smoothing)
            else:
                return nn.CrossEntropyLoss(weight=class_weights)
        elif name == 'ohemcrossentropy':
            return OHEMCrossEntropy(ohem_rate=ohem_rate, labelsmoothing_epsilon=label_smoothing)

        elif name == 'mse':
            return nn.MSELoss()
        else:
            raise ValueError()

    crittype_train = cfg['training']['train_criterion']
    crittype_test = cfg['training']['test_criterion']
    class_weights_train = cfg['training'].get('class_weights_train', None)
    class_weights_test = cfg['training'].get('class_weights_test', None)
    label_smoothing = cfg['training'].get('label_smoothing', None)
    ohem_rate = cfg['training'].get('ohem_rate', None)

    if class_weights_train:
        print(f"Using class weights {class_weights_train} for training")

    if class_weights_test:
        print(f"Using class weights {class_weights_test} for testing")

    train_criterion = get_criterion(crittype_train, class_weights=class_weights_train, label_smoothing=label_smoothing, ohem_rate=ohem_rate)
    test_criterion = get_criterion(crittype_test, class_weights=class_weights_test, label_smoothing=label_smoothing, ohem_rate=ohem_rate)

    return train_criterion, test_criterion
