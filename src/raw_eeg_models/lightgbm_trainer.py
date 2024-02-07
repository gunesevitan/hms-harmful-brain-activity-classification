import sys
import argparse
from pathlib import Path
import yaml
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

import preprocessing
sys.path.append('..')
import settings
import metrics
import visualization


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory', type=str)
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    model_directory = Path(settings.MODELS / args.model_directory)
    config = yaml.load(open(model_directory / 'config.yaml'), Loader=yaml.FullLoader)
    settings.logger.info(f'Running {model_directory} model in {args.mode} mode')

    target_columns = config['dataset']['target_columns']
    prediction_columns = [f'{column}_prediction' for column in target_columns]
    normalize_targets = config['dataset']['normalize_targets']

    df = pd.read_csv(settings.DATA / 'hms-harmful-brain-activity-classification' / 'train.csv')

    # Read and merge precomputed features
    df_eeg_features = pd.read_parquet(settings.DATA / 'eeg_subsample' / 'eeg_features.parquet')
    df = df.merge(df_eeg_features, on=['eeg_id', 'eeg_sub_id'], how='left')
    del df_eeg_features

    # Read and merge precomputed folds
    df_folds = pd.read_csv(settings.DATA / 'folds.csv')
    df = df.merge(df_folds, on=['eeg_id', 'eeg_sub_id', 'spectrogram_id', 'spectrogram_sub_id'], how='left')
    del df_folds

    settings.logger.info(f'Dataset Shape: {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

    if normalize_targets:
        df = preprocessing.normalize_targets(df=df, target_columns=target_columns)
    df = preprocessing.encode_target(df=df)

    folds = config['training']['folds']
    target = config['training']['target']
    features = config['training']['features']
    categorical_features = config['training']['categorical_features']
    seed = config['training']['seed']

    if args.mode == 'validation':

        settings.logger.info(
            f'''
            Running LightGBM trainer in {args.mode} mode
            Dataset Shape: {df.shape} - Memory Consumption: {df.memory_usage().sum() / 1024 ** 2:.2f} MB
            Folds: {folds}
            Features: {json.dumps(features, indent=2)}
            Categorical Features: {json.dumps(categorical_features, indent=2)}
            Target: {target}
            '''
        )

        df_feature_importance_gain = pd.DataFrame(
            data=np.zeros((len(features), len(folds))),
            index=features,
            columns=folds
        )
        df_feature_importance_split = pd.DataFrame(
            data=np.zeros((len(features), len(folds))),
            index=features,
            columns=folds
        )
        df_scores = []

        for fold in folds:

            training_idx, validation_idx = df.loc[df[fold] != 1].index, df.loc[df[fold] == 1].index
            # Validate on training set if validation is set is not specified
            if len(validation_idx) == 0:
                validation_idx = training_idx

            settings.logger.info(
                f'''
                Seed: {seed} Fold: {fold}
                Training: ({len(training_idx)}) - Target Mean: {df.loc[training_idx, target].mean():.4f}
                Validation: ({len(validation_idx)}) - Target Mean: {df.loc[validation_idx, target].mean():.4f}
                '''
            )

            training_dataset = lgb.Dataset(df.loc[training_idx, features], label=df.loc[training_idx, target], categorical_feature=categorical_features)
            validation_dataset = lgb.Dataset(df.loc[validation_idx, features], label=df.loc[validation_idx, target], categorical_feature=categorical_features)

            config['model']['seed'] = seed
            config['model']['feature_fraction_seed'] = seed
            config['model']['bagging_seed'] = seed
            config['model']['drop_seed'] = seed
            config['model']['data_random_seed'] = seed

            model = lgb.train(
                params=config['model'],
                train_set=training_dataset,
                valid_sets=[training_dataset, validation_dataset],
                num_boost_round=config['fit']['boosting_rounds'],
                callbacks=[
                    lgb.log_evaluation(config['fit']['log_evaluation'])
                ]
            )
            model.save_model(
                model_directory / f'model_fold_{fold}.txt',
                num_iteration=None,
                start_iteration=0,
                importance_type='gain'
            )
            settings.logger.info(f'Saved model_fold_{fold}.txt to {model_directory}')

            df_feature_importance_gain[fold] += model.feature_importance(importance_type='gain')
            df_feature_importance_split[fold] += model.feature_importance(importance_type='split')

            validation_predictions = model.predict(df.loc[validation_idx, features])
            df.loc[validation_idx, prediction_columns] = validation_predictions

            val_scores = metrics.multiclass_classification_scores(
                df_targets=df.loc[validation_idx, target_columns],
                df_predictions=df.loc[validation_idx, prediction_columns]
            )
            df_scores.append(val_scores)
            settings.logger.info(f'Fold: {fold} - Validation Scores: {json.dumps(val_scores, indent=2)}')

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

        id_columns = ['eeg_id', 'eeg_sub_id', 'spectrogram_id', 'spectrogram_sub_id']
        df[id_columns + target_columns + prediction_columns].to_csv(model_directory / 'oof_predictions.csv', index=False)
        settings.logger.info(f'oof_predictions.csv is saved to {model_directory}')

    elif args.mode == 'submission':

        settings.logger.info(
            f'''
            Running LightGBM trainer in {args.mode} mode
            Dataset Shape: {df.shape} - Memory Consumption: {df.memory_usage().sum() / 1024 ** 2:.2f} MB
            Folds: {folds}
            Features: {json.dumps(features, indent=2)}
            Categorical Features: {json.dumps(categorical_features, indent=2)}
            Target: {target}
            '''
        )

        training_dataset = lgb.Dataset(df.loc[:, features], label=df.loc[:, target], categorical_feature=categorical_features)

        config['model']['seed'] = seed
        config['model']['feature_fraction_seed'] = seed
        config['model']['bagging_seed'] = seed
        config['model']['drop_seed'] = seed
        config['model']['data_random_seed'] = seed

        model = lgb.train(
            params=config['model'],
            train_set=training_dataset,
            valid_sets=[training_dataset],
            num_boost_round=config['fit']['boosting_rounds'],
            callbacks=[
                lgb.log_evaluation(config['fit']['log_evaluation']),
            ]
        )
        model.save_model(
            model_directory / 'model.txt',
            num_iteration=None,
            start_iteration=0,
            importance_type='gain'
        )
        settings.logger.info(f'Saved model.txt to {model_directory}')

    else:
        raise ValueError(f'Invalid mode {args.mode}')
