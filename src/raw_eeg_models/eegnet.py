import torch
import torch.nn as nn

from heads import ClassificationHead


class ResNet1DBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, downsampling):

        super(ResNet1DBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.activation = nn.LeakyReLU(inplace=False)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.pooling = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.downsampling = downsampling

    def forward(self, x):

        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pooling(x)

        identity = self.downsampling(identity)
        outputs = x + identity
        outputs = self.activation(outputs)

        return outputs


class EEGNet(nn.Module):

    def __init__(self, in_channels, kernels, fixed_kernel_size, head_args):

        super(EEGNet, self).__init__()

        self.in_channels = in_channels
        self.kernels = kernels
        self.planes = 128
        self.parallel_conv = nn.ModuleList()

        for i, kernel_size in enumerate(self.kernels):
            sep_conv = nn.Conv1d(in_channels=in_channels, out_channels=self.planes, kernel_size=kernel_size, stride=1, padding=0, bias=False)
            self.parallel_conv.append(sep_conv)

        self.bn1 = nn.BatchNorm1d(num_features=self.planes)
        self.activation = nn.LeakyReLU(inplace=False)
        self.conv1 = nn.Conv1d(in_channels=self.planes, out_channels=self.planes, kernel_size=fixed_kernel_size, stride=2, padding=2, bias=False)
        self.res_blocks = self._make_resnet_layer(kernel_size=fixed_kernel_size, stride=1, padding=fixed_kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(num_features=self.planes)
        self.pooling = nn.AvgPool1d(kernel_size=6, stride=6, padding=2)
        self.head = ClassificationHead(input_dimensions=256, **head_args)

    def _make_resnet_layer(self, kernel_size, stride, blocks=11, padding=0):

        layers = []

        for i in range(blocks):
            downsampling = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
            res_block = ResNet1DBlock(
                in_channels=self.planes,
                out_channels=self.planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                downsampling=downsampling
            )
            layers.append(res_block)

        return nn.Sequential(*layers)

    def forward(self, x):

        out_sep = []

        for i in range(len(self.kernels)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)

        x = torch.cat(out_sep, dim=2)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv1(x)

        x = self.res_blocks(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pooling(x)

        x = x.reshape(x.shape[0], -1)
        outputs = self.head(x)

        return outputs
