import sys
import yaml
import json
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from mne.filter import filter_data, notch_filter

import preprocessing
import torch_datasets
sys.path.append('..')
import settings
import metrics
import visualization


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, conv_bias, batch_norm):

        super(_DenseLayer, self).__init__()

        if batch_norm:
            self.add_module('norm1', nn.BatchNorm1d(num_input_features)),
        self.add_module('elu1', nn.ELU()),
        self.add_module('conv1', nn.Conv1d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=conv_bias))
        if batch_norm:
            self.add_module('norm2', nn.BatchNorm1d(bn_size * growth_rate)),
        self.add_module('elu2', nn.ELU()),
        self.add_module('conv2', nn.Conv1d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=conv_bias))
        self.drop_rate = drop_rate

    def forward(self, x):

        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, conv_bias, batch_norm):

        super(_DenseBlock, self).__init__()

        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate, conv_bias, batch_norm)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features, conv_bias, batch_norm):

        super(_Transition, self).__init__()

        if batch_norm:
            self.add_module('norm', nn.BatchNorm1d(num_input_features))

        self.add_module('elu', nn.ELU())
        self.add_module('conv', nn.Conv1d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=conv_bias))
        self.add_module('pool', nn.AvgPool1d(kernel_size=2, stride=2))


class DenseNetEnconder(nn.Module):

    def __init__(self, growth_rate=32, block_config=(4, 4, 4, 4, 4, 4, 4), in_channels=16, num_init_features=64, bn_size=4, drop_rate=0.2, conv_bias=True, batch_norm=False):

        super(DenseNetEnconder, self).__init__()

        first_conv = OrderedDict([('conv0', nn.Conv1d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=conv_bias))])

        if batch_norm:
            first_conv['norm0'] = nn.BatchNorm1d(num_init_features)

        first_conv['elu0'] = nn.ELU()
        first_conv['pool0'] = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.densenet = nn.Sequential(first_conv)

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, conv_bias=conv_bias, batch_norm=batch_norm)
            self.densenet.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, conv_bias=conv_bias, batch_norm=batch_norm)
                self.densenet.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        if batch_norm:
            self.densenet.add_module('norm{}'.format(len(block_config) + 1), nn.BatchNorm1d(num_features))

        self.densenet.add_module('relu{}'.format(len(block_config) + 1), nn.ReLU())
        self.densenet.add_module('pool{}'.format(len(block_config) + 1), nn.AvgPool1d(kernel_size=7, stride=3))  # stride originally 1

        self.num_features = num_features

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.densenet(x)
        return features.view(features.size(0), -1)


class DenseNetClassifier(nn.Module):

    def __init__(self, growth_rate=32, block_config=(4, 4, 4, 4, 4, 4, 4), in_channels=16, num_init_features=64, bn_size=4, drop_rate=0.2, conv_bias=True, batch_norm=False, drop_fc=0.5, num_classes=6):

        super(DenseNetClassifier, self).__init__()

        self.features = DenseNetEnconder(growth_rate=growth_rate, block_config=block_config, in_channels=in_channels,
                                         num_init_features=num_init_features, bn_size=bn_size, drop_rate=drop_rate,
                                         conv_bias=conv_bias, batch_norm=batch_norm)

        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_fc),
            nn.Linear(self.features.num_features, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = self.classifier(features)
        return out, features


def preprocess_sparcnet(inputs):

    inputs_difference = np.zeros((inputs.shape[0], 16))

    inputs_difference[:, 0] = inputs[:, 0] - inputs[:, 4]
    inputs_difference[:, 1] = inputs[:, 4] - inputs[:, 5]
    inputs_difference[:, 2] = inputs[:, 5] - inputs[:, 6]
    inputs_difference[:, 3] = inputs[:, 6] - inputs[:, 7]

    inputs_difference[:, 4] = inputs[:, 11] - inputs[:, 15]
    inputs_difference[:, 5] = inputs[:, 15] - inputs[:, 16]
    inputs_difference[:, 6] = inputs[:, 16] - inputs[:, 17]
    inputs_difference[:, 7] = inputs[:, 17] - inputs[:, 18]

    inputs_difference[:, 8] = inputs[:, 0] - inputs[:, 1]
    inputs_difference[:, 9] = inputs[:, 1] - inputs[:, 2]
    inputs_difference[:, 10] = inputs[:, 2] - inputs[:, 3]
    inputs_difference[:, 11] = inputs[:, 3] - inputs[:, 7]

    inputs_difference[:, 12] = inputs[:, 11] - inputs[:, 12]
    inputs_difference[:, 13] = inputs[:, 12] - inputs[:, 13]
    inputs_difference[:, 14] = inputs[:, 13] - inputs[:, 14]
    inputs_difference[:, 15] = inputs[:, 14] - inputs[:, 18]

    inputs_difference = notch_filter(inputs_difference, 200, 60, n_jobs=-1, verbose='ERROR')
    inputs_difference = filter_data(inputs_difference, 200, 0.5, 40, n_jobs=-1, verbose='ERROR')

    mean = np.mean(inputs_difference, axis=0)
    std = np.std(inputs_difference, axis=0)
    inputs_difference = (inputs_difference - mean) / (std + 1e-15)

    inputs_difference = inputs_difference[::5, :]

    return inputs_difference



if __name__ == '__main__':

    model_directory = Path(settings.MODELS / 'sparcnet')
    config = yaml.load(open(model_directory / 'config.yaml'), Loader=yaml.FullLoader)
    settings.logger.info(f'Running {model_directory} model in inference mode')

    eeg_dataset_path = settings.DATA / config['dataset']['eeg_dataset']
    normalize_targets = config['dataset']['normalize_targets']

    df = pd.read_csv(settings.DATA / 'hms-harmful-brain-activity-classification' / 'train.csv')
    settings.logger.info(f'Dataset Shape: {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

    # Read and merge precomputed folds
    df_folds = pd.read_csv(settings.DATA / 'folds.csv')
    df = df.merge(df_folds, on=['eeg_id', 'eeg_sub_id', 'spectrogram_id', 'spectrogram_sub_id'], how='left').dropna().reset_index(drop=True)
    del df_folds

    target_columns = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    df = preprocessing.extract_sample_quality(df=df, target_columns=target_columns)
    if normalize_targets:
        df = preprocessing.normalize_targets(df=df, target_columns=target_columns)

    torch.multiprocessing.set_sharing_strategy('file_system')

    df_scores = []
    prediction_columns = [f'{column}_prediction' for column in target_columns]

    folds = config['test']['folds']
    model_file_names = config['test']['model_file_names']

    for fold, model_file_name in zip(folds, model_file_names):

        validation_idx = df.loc[df[fold] == 1].index

        # Create validation inputs and targets
        validation_eeg_paths, validation_targets, validation_target_classes, validation_sample_qualities = torch_datasets.prepare_data(
            df=df.loc[validation_idx],
            eeg_dataset_path=eeg_dataset_path
        )

        settings.logger.info(
            f'''
            Fold: {fold} ({model_file_name})
            '''
        )

        device = torch.device('cuda')
        amp = True
        task_type = 'multiclass'

        model = torch.load(model_directory / 'model.pt')
        model.to(device)
        model.eval()

        validation_predictions = []

        for eeg_path, targets in tqdm(zip(validation_eeg_paths, validation_targets), total=len(validation_eeg_paths)):

            inputs = np.load(eeg_path)
            inputs = pd.DataFrame(inputs).interpolate(method='linear').ffill().bfill().values
            inputs = preprocess_sparcnet(inputs)
            inputs = torch.unsqueeze(torch.as_tensor(inputs.T, dtype=torch.float32), dim=0)
            inputs = inputs.to(device)

            with torch.no_grad():
                if amp:
                    with torch.autocast(device_type=device.type):
                        outputs = model(inputs.half())[0].float()
                else:
                    outputs = model(inputs)

            outputs = outputs.cpu()

            if config['test']['tta']:

                inputs = inputs.to('cpu')
                tta_flip_dimensions = config['test']['tta_flip_dimensions']

                tta_outputs = []

                for dimensions in tta_flip_dimensions:

                    augmented_inputs = torch.flip(inputs, dims=dimensions).to(device)

                    with torch.no_grad():
                        if amp:
                            with torch.autocast(device_type=device.type):
                                augmented_outputs = model(augmented_inputs.half()).float()
                        else:
                            augmented_outputs = model(augmented_inputs)

                    tta_outputs.append(augmented_outputs.cpu())

                outputs = torch.stack(([outputs] + tta_outputs), dim=-1)
                outputs = torch.mean(outputs, dim=-1)

            validation_predictions += [outputs]

        if task_type == 'binary':
            validation_predictions = torch.sigmoid(torch.concatenate(validation_predictions).float()).numpy()
            validation_predictions = validation_predictions / validation_predictions.sum(axis=1).reshape(-1, 1)
        elif task_type == 'multiclass':
            validation_predictions = torch.softmax(torch.concatenate(validation_predictions).float(), dim=-1).numpy()
        else:
            raise ValueError(f'Invalid task type {task_type}')

        df.loc[validation_idx, prediction_columns] = validation_predictions
        validation_scores = metrics.multiclass_classification_scores(
            df_targets=df.loc[validation_idx, target_columns],
            df_predictions=df.loc[validation_idx, prediction_columns]
        )
        for sample_quality in [0, 1, 2]:
            sample_quality_mask = df['sample_quality'] == sample_quality
            sample_quality_validation_idx = validation_idx.intersection(np.where(sample_quality_mask)[0])
            sample_quality_validation_scores = metrics.multiclass_classification_scores(
                df_targets=df.loc[sample_quality_validation_idx, target_columns],
                df_predictions=df.loc[sample_quality_validation_idx, prediction_columns]
            )
            sample_quality_validation_scores = {f'sq_{sample_quality}_{k}': v for k, v in sample_quality_validation_scores.items()}
            validation_scores.update(sample_quality_validation_scores)
        df_scores.append(validation_scores)
        settings.logger.info(f'{fold} Validation Scores: {json.dumps(validation_scores, indent=2)}')

    df_scores = pd.DataFrame(df_scores)
    settings.logger.info(
        f'''
        Mean Validation Scores
        {json.dumps(df_scores.mean(axis=0).to_dict(), indent=2)}
        and Standard Deviations
        ±{json.dumps(df_scores.std(axis=0).to_dict(), indent=2)}
        '''
    )
    visualization.visualize_scores(
        df_scores=df_scores,
        title=f'Scores of {len(folds)} Model(s)',
        path=model_directory / 'scores.png'
    )
    settings.logger.info(f'scores.png is saved to {model_directory}')

    oof_scores = metrics.multiclass_classification_scores(
        df_targets=df.loc[:, target_columns],
        df_predictions=df.loc[:, prediction_columns]
    )
    for sample_quality in [0, 1, 2]:
        sample_quality_mask = df['sample_quality'] == sample_quality
        sample_quality_oof_scores = metrics.multiclass_classification_scores(
            df_targets=df.loc[sample_quality_mask, target_columns],
            df_predictions=df.loc[sample_quality_mask, prediction_columns]
        )
        sample_quality_oof_scores = {f'sq_{sample_quality}_{k}': v for k, v in sample_quality_oof_scores.items()}
        oof_scores.update(sample_quality_oof_scores)
    settings.logger.info(f'OOF Scores: {json.dumps(oof_scores, indent=2)}')

    df_scores = pd.concat((
        df_scores,
        pd.DataFrame([oof_scores])
    ), axis=0).reset_index(drop=True)
    df_scores['fold'] = ['1', '2', '3', '4', '5', 'OOF']
    df_scores.to_csv(model_directory / 'scores.csv', index=False)
    settings.logger.info(f'scores.csv is saved to {model_directory}')

    for target_column, prediction_column in zip(target_columns, prediction_columns):
        visualization.visualize_predictions(
            y_true=df[target_column],
            y_pred=df[prediction_column],
            title=f'{target_column} and {prediction_column} Histograms (OOF Predictions of {len(folds)} Model(s)',
            path=model_directory / f'{target_column}_histogram.png'
        )
        settings.logger.info(f'{target_column}_histogram.png is saved to {model_directory}')

    visualization.visualize_confusion_matrix(
        y_true=np.argmax(df[target_columns], axis=1),
        y_pred=np.argmax(df[prediction_columns], axis=1),
        title='OOF Confusion Matrix',
        path=model_directory / 'confusion_matrix.png'
    )
    settings.logger.info(f'confusion_matrix.png is saved to {model_directory}')

    for sample_quality in [0, 1, 2]:
        sample_quality_mask = df['sample_quality'] == sample_quality
        visualization.visualize_confusion_matrix(
            y_true=np.argmax(df.loc[sample_quality_mask, target_columns], axis=1),
            y_pred=np.argmax(df.loc[sample_quality_mask, prediction_columns], axis=1),
            title=f'Sample Quality {sample_quality} OOF Confusion Matrix',
            path=model_directory / f'sq_{sample_quality}_confusion_matrix.png'
        )
        settings.logger.info(f'sq_{sample_quality}_confusion_matrix.png is saved to {model_directory}')

    id_columns = ['eeg_id', 'eeg_sub_id', 'spectrogram_id', 'spectrogram_sub_id']
    df[id_columns + target_columns + prediction_columns].to_csv(model_directory / 'oof_predictions.csv', index=False)
    settings.logger.info(f'oof_predictions.csv is saved to {model_directory}')
