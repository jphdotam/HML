import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Adapted by James Howard from https://www.nature.com/articles/s41591-018-0268-3

To make this work with our single ECG snippets, the filter size is 3 by default rather than 16 (they used much
longer ECG strips).

It's also not clear how they downsampled in alternating blocks - I have assumed with a stride.
If so, it's also not clear which of the 2 convolutional layers did this. I've assumed the second.
Also, when they increase the number of filters, it's unclear how residual connections cope with this.
I've assumed by using a conv layer instead of a maxpool.
"""

def conv1d(in_planes, out_planes, kernel_size, downsample=False):
    """Width 3 convolution with padding"""
    assert kernel_size % 2, f"Kernel size must be an odd numbers, not {kernel_size}"
    padding = kernel_size % 2
    stride = 2 if downsample else 1
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

class HannunNet(nn.Module):
    def __init__(self, n_inputs, n_outputs, filter_size=3, n_blocks=16):
        super(HannunNet, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.filter_size = filter_size
        self.n_blocks = n_blocks

        self.conv1 = nn.Sequential(
            conv1d(n_inputs, 32, self.filter_size),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True))

        # This first block must downsample (paper says alternates, but shows a maxpool on the 1st block)
        self.block0 = nn.Sequential(
            conv1d(32, 32, self.filter_size, downsample=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            conv1d(32, 32, self.filter_size, downsample=True))

        self.maxpool = nn.MaxPool1d(2)

        blocks = []
        for i_block in range(1, n_blocks):  # Start at 1, as block0 is the first block
            k_this = i_block // 4  # layers have 32*2k filters; starts at k 0 and +=1 every 4th block
            k_last = (i_block - 1) // 4
            downsample = not (i_block % 2)  # The block0 downsamples, 1 does not
            input_filters = int(32 * math.pow(2, k_last))
            output_filters = int(32 * math.pow(2, k_this))
            print(f"{i_block} {k_last} -> {k_this}; {input_filters} - {output_filters} - {downsample}")
            blocks.append(HannunBlock(input_filters, output_filters, self.filter_size, downsample))
        self.blocks = nn.Sequential(*blocks)

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(output_filters),
            nn.ReLU(inplace=True),
            nn.Linear(output_filters, n_outputs)
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        if x.shape[-1] % 64 or x.shape[-1] < 256:
            # This implementation requires inputs to be divisible by 64 to ensure residual connections work properly
            newlen = max(math.ceil(x.shape[2]/64)*64, 256)
            new = torch.zeros(x.shape[0], x.shape[1], newlen)
            new[:, :, :x.shape[-1]] = x
            x = new.to(x.device)

        # initial conv-bn-relu
        x = self.conv1(x)

        # pre-block (down samples)
        inner = self.block0(x)
        outer = self.maxpool(x)
        x = inner + outer

        # blocks
        x = self.blocks(x)

        # avg pol
        x = self.avg_pool(x).squeeze(-1)

        # classifier (excluding softmax)
        x = self.classifier(x)

        return x


class HannunBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, filter_size, downsample):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.filter_size = filter_size
        self.downsample = downsample
        super(HannunBlock, self).__init__()
        self.inner = nn.Sequential(
            nn.BatchNorm1d(n_inputs),
            nn.ReLU(inplace=True),
            conv1d(n_inputs, n_outputs, filter_size, downsample=False),
            nn.BatchNorm1d(n_outputs),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            conv1d(n_outputs, n_outputs, filter_size, downsample=downsample)
        )
        self.outer = self.get_outer(n_inputs, n_outputs)

    def get_outer(self, n_inputs, n_outputs):
        if n_inputs != n_outputs:
            return conv1d(n_inputs, n_outputs, self.filter_size, downsample=True)
        else:
            return nn.MaxPool1d(2)

    def forward(self, x):
        inner = self.inner(x)
        if self.downsample:
            outer = self.outer(x)
            return inner + outer
        else:
            return inner


if __name__ == "__main__":
    def get_n_params(model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    import torch
    net = HannunNet(12, 12)
    pred = net(torch.zeros(4, 12, 512))
    print(get_n_params(net))
