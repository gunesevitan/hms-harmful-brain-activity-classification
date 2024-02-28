import torch
import torch.nn as nn
import torch.nn.functional as F

from heads import ClassificationHead


class CSNBottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, channels, stride=1):

        super(CSNBottleneck, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False, groups=channels)
        self.bn2 = nn.BatchNorm3d(channels)

        self.conv3 = nn.Conv3d(channels, channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(channels * self.expansion)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(channels * self.expansion)
            )

    def forward(self, x):
        shortcut = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += shortcut
        out = self.relu(out)

        return out


class CSN(nn.Module):

    def __init__(self, block=CSNBottleneck, layers=(3, 4, 6, 3), inp_channels=3):

        super(CSN, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv3d(inp_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.layer1 = self._make_layer(block, 64, layers[0], stride=(1, 1, 1))
        self.layer2 = self._make_layer(block, 128, layers[1], stride=(1, 2, 2))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=(1, 2, 2))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=(1, 2, 2))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, channels, n_blocks, stride):
        layers = [block(self.in_channels, channels, stride)]
        self.in_channels = channels * block.expansion
        for i in range(1, n_blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.max_pool(out)

        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        return out4


class CSNModel(nn.Module):

    def __init__(self, backbone_args, pooling_type, dropout_rate, head_args):

        super(CSNModel, self).__init__()

        self.encoder = CSN(**backbone_args)

        self.pooling_type = pooling_type
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.head = ClassificationHead(input_dimensions=2048, **head_args)

    def forward(self, x):

        x = self.encoder(x)

        if self.pooling_type == 'avg':
            x = F.adaptive_avg_pool3d(x, output_size=(1, 1, 1)).view(x.size(0), -1)
        elif self.pooling_type == 'max':
            x = F.adaptive_max_pool3d(x, output_size=(1, 1, 1)).view(x.size(0), -1)
        elif self.pooling_type == 'concat':
            x = torch.cat([
                F.adaptive_avg_pool3d(x, output_size=(1, 1, 1)).view(x.size(0), -1),
                F.adaptive_max_pool3d(x, output_size=(1, 1, 1)).view(x.size(0), -1)
            ], dim=-1)

        x = self.dropout(x)
        output = self.head(x)

        return output
