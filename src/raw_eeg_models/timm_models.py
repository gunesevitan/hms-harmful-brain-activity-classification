import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from heads import ClassificationHead


class EfficientNet(nn.Module):

    def __init__(self, model_name, pretrained, backbone_args, pooling_type, dropout_rate, head_args):

        super(EfficientNet, self).__init__()

        self.backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            **backbone_args
        )

        input_features = self.backbone.get_classifier().in_features

        self.pooling_type = pooling_type
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.head = ClassificationHead(input_dimensions=input_features, **head_args)

    def forward(self, x):

        x = self.backbone.forward_features(x)

        if self.pooling_type == 'avg':
            x = F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
        elif self.pooling_type == 'max':
            x = F.adaptive_max_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
        elif self.pooling_type == 'concat':
            x = torch.cat([
                F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.size(0), -1),
                F.adaptive_max_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
            ], dim=-1)

        x = self.dropout(x)
        output = self.head(x)

        return output


class NFNet(nn.Module):

    def __init__(self, model_name, pretrained, backbone_args, pooling_type, dropout_rate, head_args):

        super(NFNet, self).__init__()

        self.backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            **backbone_args
        )

        input_features = self.backbone.get_classifier().in_features

        self.pooling_type = pooling_type
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.head = ClassificationHead(input_dimensions=input_features, **head_args)

    def forward(self, x):

        x = self.backbone.forward_features(x)

        if self.pooling_type == 'avg':
            x = F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
        elif self.pooling_type == 'max':
            x = F.adaptive_max_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
        elif self.pooling_type == 'concat':
            x = torch.cat([
                F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.size(0), -1),
                F.adaptive_max_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
            ], dim=-1)

        x = self.dropout(x)
        output = self.head(x)

        return output


class ConvNeXt(nn.Module):

    def __init__(self, model_name, pretrained, backbone_args, pooling_type, dropout_rate, head_args):

        super(ConvNeXt, self).__init__()

        self.backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            **backbone_args
        )

        input_features = self.backbone.get_classifier().in_features

        self.pooling_type = pooling_type
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.head = ClassificationHead(input_dimensions=input_features, **head_args)

    def forward(self, x):

        x = self.backbone.forward_features(x)

        if self.pooling_type == 'avg':
            x = F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
        elif self.pooling_type == 'max':
            x = F.adaptive_max_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
        elif self.pooling_type == 'concat':
            x = torch.cat([
                F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.size(0), -1),
                F.adaptive_max_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
            ], dim=-1)

        x = self.dropout(x)
        output = self.head(x)

        return output


class InceptionNeXt(nn.Module):

    def __init__(self, model_name, pretrained, backbone_args, pooling_type, dropout_rate, head_args):

        super(InceptionNeXt, self).__init__()

        self.backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            **backbone_args
        )

        input_features = self.backbone.get_classifier().in_features
        self.backbone.head = nn.Identity()
        self.pooling_type = pooling_type
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.head = ClassificationHead(input_dimensions=int(input_features / 3), **head_args)

    def forward(self, x):

        x = self.backbone.forward_features(x)

        if self.pooling_type == 'avg':
            x = F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
        elif self.pooling_type == 'max':
            x = F.adaptive_max_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
        elif self.pooling_type == 'concat':
            x = torch.cat([
                F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.size(0), -1),
                F.adaptive_max_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
            ], dim=-1)

        x = self.dropout(x)
        output = self.head(x)

        return output


class CoaT(nn.Module):

    def __init__(self, model_name, pretrained, backbone_args, dropout_rate, head_args):

        super(CoaT, self).__init__()

        self.backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            **backbone_args
        )

        input_features = self.backbone.get_classifier().in_features
        self.backbone.head_drop = nn.Identity()
        self.backbone.head = nn.Identity()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.head = ClassificationHead(input_dimensions=input_features, **head_args)

    def forward(self, x):

        x = self.backbone(x)
        x = self.dropout(x)
        output = self.head(x)

        return output


class GCViT(nn.Module):

    def __init__(self, model_name, pretrained, backbone_args, pooling_type, dropout_rate, head_args):

        super(GCViT, self).__init__()

        self.backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            **backbone_args
        )

        input_features = self.backbone.get_classifier().in_features
        self.pooling_type = pooling_type
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.head = ClassificationHead(input_dimensions=input_features, **head_args)

    def forward(self, x):

        x = self.backbone.forward_features(x)

        if self.pooling_type == 'avg':
            x = F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
        elif self.pooling_type == 'max':
            x = F.adaptive_max_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
        elif self.pooling_type == 'concat':
            x = torch.cat([
                F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.size(0), -1),
                F.adaptive_max_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
            ], dim=-1)

        x = self.dropout(x)
        output = self.head(x)

        return output


class CoAtNet(nn.Module):

    def __init__(self, model_name, pretrained, backbone_args, pooling_type, dropout_rate, head_args):

        super(CoAtNet, self).__init__()

        self.backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            **backbone_args
        )

        input_features = self.backbone.get_classifier().in_features
        self.pooling_type = pooling_type
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.head = ClassificationHead(input_dimensions=input_features, **head_args)

    def forward(self, x):

        x = self.backbone.forward_features(x)

        if self.pooling_type == 'avg':
            x = F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
        elif self.pooling_type == 'max':
            x = F.adaptive_max_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
        elif self.pooling_type == 'concat':
            x = torch.cat([
                F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.size(0), -1),
                F.adaptive_max_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
            ], dim=-1)

        x = self.dropout(x)
        output = self.head(x)

        return output


class NextViT(nn.Module):

    def __init__(self, model_name, pretrained, backbone_args, pooling_type, dropout_rate, head_args):

        super(NextViT, self).__init__()

        self.backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            **backbone_args
        )

        input_features = self.backbone.get_classifier().in_features
        self.pooling_type = pooling_type
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.head = ClassificationHead(input_dimensions=input_features, **head_args)

    def forward(self, x):

        x = self.backbone.forward_features(x)

        if self.pooling_type == 'avg':
            x = F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
        elif self.pooling_type == 'max':
            x = F.adaptive_max_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
        elif self.pooling_type == 'concat':
            x = torch.cat([
                F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.size(0), -1),
                F.adaptive_max_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
            ], dim=-1)

        x = self.dropout(x)
        output = self.head(x)

        return output


class SwinTransformer(nn.Module):

    def __init__(self, model_name, pretrained, backbone_args, pooling_type, dropout_rate, head_args):

        super(SwinTransformer, self).__init__()

        self.backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            **backbone_args
        )

        input_features = self.backbone.get_classifier().in_features
        self.pooling_type = pooling_type
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.head = ClassificationHead(input_dimensions=input_features, **head_args)

    def forward(self, x):

        x = self.backbone.forward_features(x)
        x = x.permute(0, -1, 1, 2)

        if self.pooling_type == 'avg':
            x = F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
        elif self.pooling_type == 'max':
            x = F.adaptive_max_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
        elif self.pooling_type == 'concat':
            x = torch.cat([
                F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.size(0), -1),
                F.adaptive_max_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
            ], dim=-1)

        x = self.dropout(x)
        output = self.head(x)

        return output


def load_timm_model(model_directory, model_file_names, device):

    """
    Load model and pretrained weights from the given model directory

    Parameters
    ----------
    model_directory: pathlib.Path
        Path of the model directory

    model_file_names: list
        List of names of the model weights files

    device: torch.device
        Location of the model

    Returns
    -------
    model: dict
        Dictionary of models with pretrained weights loaded

    config: dict
        Dictionary of configurations
    """

    config = yaml.load(open(model_directory / 'config.yaml', 'r'), Loader=yaml.FullLoader)
    config['model']['model_args']['pretrained'] = False

    models = {}

    for model_file_name in model_file_names:
        model = eval(config['model']['model_class'])(**config['model']['model_args'])
        model.load_state_dict(torch.load(model_directory / model_file_name))
        model.to(device)
        model.eval()
        models[model_file_name] = model

    return models, config
