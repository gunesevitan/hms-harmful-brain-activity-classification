import torch
import torch.nn as nn
import torch.nn.functional as F

from heads import ClassificationHead


class WaveBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dilation_rates, kernel_size):

        super(WaveBlock, self).__init__()

        self.dilation_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2 ** i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=int((dilation_rate * (kernel_size - 1)) / 2),
                dilation=dilation_rate)
            )
            self.gate_convs.append(nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=int((dilation_rate * (kernel_size - 1)) / 2),
                dilation=dilation_rate)
            )
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.dilation_rates):
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            x = self.convs[i + 1](x)
            res = res + x
        return res


class WaveNet(nn.Module):

    def __init__(self, in_channels, kernel_size, pooling_type, dropout_rate, head_args):

        super(WaveNet, self).__init__()

        self.wave_block1 = WaveBlock(in_channels, 16, 12, kernel_size)
        self.wave_block2 = WaveBlock(16, 32, 8, kernel_size)
        self.wave_block3 = WaveBlock(32, 64, 4, kernel_size)
        self.wave_block4 = WaveBlock(64, 128, 1, kernel_size)

        self.pooling_type = pooling_type
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.head = ClassificationHead(input_dimensions=128, **head_args)

    def forward(self, x):

        x = self.wave_block1(x)
        x = self.wave_block2(x)
        x = self.wave_block3(x)
        x = self.wave_block4(x)

        if self.pooling_type == 'avg':
            x = F.adaptive_avg_pool1d(x, output_size=(1,)).view(x.size(0), -1)
        elif self.pooling_type == 'max':
            x = F.adaptive_max_pool1d(x, output_size=(1,)).view(x.size(0), -1)
        elif self.pooling_type == 'concat':
            x = torch.cat([
                F.adaptive_avg_pool1d(x, output_size=(1,)).view(x.size(0), -1),
                F.adaptive_max_pool1d(x, output_size=(1,)).view(x.size(0), -1)
            ], dim=-1)

        x = self.dropout(x)
        output = self.head(x)

        return output
