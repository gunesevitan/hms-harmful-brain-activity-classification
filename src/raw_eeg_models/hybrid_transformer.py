import numpy as np
import torch
import torch.nn as nn

from heads import ClassificationHead


def positional_encoding(length, embed_dim):

    dim = embed_dim // 2

    position = np.arange(length)[:, np.newaxis]
    dim = np.arange(dim)[np.newaxis, :] / dim

    angle = 1 / (10000 ** dim)
    angle = position * angle

    pos_embed = np.concatenate([np.sin(angle), np.cos(angle)], axis=-1)
    pos_embed = torch.from_numpy(pos_embed).float()

    return pos_embed


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), padding_mode='zeros', bias=True)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), padding_mode='zeros', bias=True)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), padding_mode='zeros', bias=True)
        self.activation = nn.LeakyReLU(inplace=False)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)

        x = self.conv3(x)

        return x


class MLP(nn.Module):

    def __init__(self, embed_dim, hidden_dim):

        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_dim, out_features=embed_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class TransformerBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, out_dim):
        super(TransformerBlock, self).__init__()

        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.mlp = MLP(embed_dim=embed_dim, hidden_dim=out_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(out_dim)

    def forward(self, x):

        x = self.ln1(x)
        x = self.attention(x, x, x)[0]
        x = self.ln2(x)
        x = self.mlp(x)

        return x


class HybridTransformer(nn.Module):

    def __init__(self, in_channels, hidden_size, num_heads, num_blocks, pooling_type, dropout_rate, head_args):

        super(HybridTransformer, self).__init__()

        self.conv_block = ConvBlock(in_channels=1, out_channels=hidden_size)
        self.encoder = nn.ModuleList([
            TransformerBlock(
                embed_dim=hidden_size,
                num_heads=num_heads,
                out_dim=hidden_size,
            ) for _ in range(num_blocks)
        ])

        self.positional_embeddings = torch.nn.Parameter(positional_encoding(in_channels, hidden_size))
        self.cls_token = nn.Parameter(torch.zeros((1, hidden_size)))

        self.pooling_type = pooling_type
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.head = ClassificationHead(input_dimensions=hidden_size, **head_args)

    def forward(self, x):

        # Add channel dimension and pass it to conv 2d block
        x = x.unsqueeze(dim=1).permute(0, 1, 3, 2)
        x = self.conv_block(x)

        # Average features along time dimension
        x = torch.mean(x, dim=2).permute(0, 2, 1)

        # Add positional embeddings and concatenate cls token
        x += self.positional_embeddings
        x = torch.cat([
            self.cls_token.unsqueeze(0).repeat(x.size(0), 1, 1),
            x
        ], 1)

        # Pass it to transformer encoder
        for block in self.encoder:
            x = block(x)

        if self.pooling_type == 'avg':
            x = torch.mean(x, dim=1)
        elif self.pooling_type == 'max':
            x = torch.max(x, dim=1)[0]
        elif self.pooling_type == 'cls':
            x = x[:, 0, :]

        x = self.dropout(x)
        output = self.head(x)

        return output
