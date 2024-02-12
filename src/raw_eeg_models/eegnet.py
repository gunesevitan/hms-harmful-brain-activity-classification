import torch
import torch.nn as nn

from heads import ClassificationHead


class GatedConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(1,), stride=(1,), padding=(2,), padding_mode='replicate'):

        super(GatedConv1d, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, bias=True),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

        self.gate = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):

        return self.conv(x) * self.gate(x)


class ResNet1DBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, downsampling):

        super(ResNet1DBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.relu = nn.LeakyReLU(inplace=False)
        self.dropout = nn.Dropout(p=0.0, inplace=False)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.downsampling = downsampling

    def forward(self, x):

        identity = x

        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.maxpool(x)

        identity = self.downsampling(identity)
        outputs = x + identity

        return outputs


class EEGNet(nn.Module):

    def __init__(self, kernels, in_channels, fixed_kernel_size, head_args):

        super(EEGNet, self).__init__()

        self.kernels = kernels
        self.planes = 24
        self.parallel_conv = nn.ModuleList()
        self.in_channels = in_channels

        for i, kernel_size in enumerate(list(self.kernels)):
            sep_conv = nn.Conv1d(in_channels=in_channels, out_channels=self.planes, kernel_size=kernel_size, stride=1, padding=0, bias=False)
            self.parallel_conv.append(sep_conv)

        self.bn1 = nn.BatchNorm1d(num_features=self.planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv1d(in_channels=self.planes, out_channels=self.planes, kernel_size=fixed_kernel_size, stride=2, padding=2, bias=False)
        self.block = self._make_resnet_layer(kernel_size=fixed_kernel_size, stride=1, padding=fixed_kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(num_features=self.planes)
        self.avgpool = nn.AvgPool1d(kernel_size=6, stride=6, padding=2)
        self.head = ClassificationHead(input_dimensions=224, **head_args)

    def _make_resnet_layer(self, kernel_size, stride, blocks=9, padding=0):

        layers = []

        for i in range(blocks):
            downsampling = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
            layers.append(ResNet1DBlock(
                in_channels=self.planes,
                out_channels=self.planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                downsampling=downsampling
            ))

        return nn.Sequential(*layers)

    def forward(self, x):

        out_sep = []

        for i in range(len(self.kernels)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)

        out = torch.cat(out_sep, dim=2)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.block(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.avgpool(out)

        out = out.reshape(out.shape[0], -1)

        result = self.head(out)

        return result
