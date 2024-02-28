import sys
import json
import numpy as np
import pandas as pd

import preprocessing
sys.path.append('..')
import settings
import metrics


def normalize_probabilities(df, columns):

    """
    Normalize probabilities to 1 within given columns

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with given columns

    columns: list
        List of column names that have probabilities

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with given columns' sum adjusted to 1
    """

    df_sums = df[columns].sum(axis=1)
    for column in columns:
        df[column] /= df_sums

    return df


if __name__ == '__main__':

    df = pd.read_csv(settings.DATA / 'hms-harmful-brain-activity-classification' / 'train.csv')
    settings.logger.info(f'Dataset Shape: {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

    target_columns = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    df = preprocessing.extract_sample_quality(df=df, target_columns=target_columns)
    df = preprocessing.normalize_targets(df=df, target_columns=target_columns)

    raw_eeg_2d_efficientnetv2medium_directory = 'raw_eeg_2d_efficientnetv2medium_384x512_quality_2'
    raw_eeg_2d_coatlitemedium_directory = 'raw_eeg_2d_coatlitemedium_384x512_quality_2'
    raw_eeg_2d_nextvitlarge_directory = 'raw_eeg_2d_nextvitlarge_384x512_quality_2'
    raw_eeg_2d_maxvittiny_directory = 'raw_eeg_2d_maxvittiny_384x512_quality_2'
    raw_eeg_2d_swinsmall_directory = 'raw_eeg_2d_swinsmall_384x512_quality_2'

    prediction_columns = [f'{column}_prediction' for column in target_columns]
    columns_to_merge = ['eeg_id', 'eeg_sub_id', 'sample_quality']
    df_raw_eeg_2d_efficientnetv2medium = pd.read_csv(settings.MODELS / raw_eeg_2d_efficientnetv2medium_directory / 'oof_predictions.csv')
    df_raw_eeg_2d_efficientnetv2medium = df_raw_eeg_2d_efficientnetv2medium.merge(df.loc[:, columns_to_merge], on=columns_to_merge[:-1], how='left')
    df_raw_eeg_2d_coatlitemedium = pd.read_csv(settings.MODELS / raw_eeg_2d_coatlitemedium_directory / 'oof_predictions.csv')
    df_raw_eeg_2d_coatlitemedium = df_raw_eeg_2d_coatlitemedium.merge(df.loc[:, columns_to_merge], on=columns_to_merge[:-1], how='left')
    df_raw_eeg_2d_nextvitlarge = pd.read_csv(settings.MODELS / raw_eeg_2d_nextvitlarge_directory / 'oof_predictions.csv')
    df_raw_eeg_2d_nextvitlarge = df_raw_eeg_2d_nextvitlarge.merge(df.loc[:, columns_to_merge], on=columns_to_merge[:-1], how='left')
    df_raw_eeg_2d_maxvittiny = pd.read_csv(settings.MODELS / raw_eeg_2d_maxvittiny_directory / 'oof_predictions.csv')
    df_raw_eeg_2d_maxvittiny = df_raw_eeg_2d_maxvittiny.merge(df.loc[:, columns_to_merge], on=columns_to_merge[:-1], how='left')
    df_raw_eeg_2d_swinsmall = pd.read_csv(settings.MODELS / raw_eeg_2d_swinsmall_directory / 'oof_predictions.csv')
    df_raw_eeg_2d_swinsmall = df_raw_eeg_2d_swinsmall.merge(df.loc[:, columns_to_merge], on=columns_to_merge[:-1], how='left')

    prediction_dfs = [
        df_raw_eeg_2d_efficientnetv2medium,
        df_raw_eeg_2d_coatlitemedium,
        df_raw_eeg_2d_nextvitlarge,
        df_raw_eeg_2d_maxvittiny,
        df_raw_eeg_2d_swinsmall
    ]
    model_names = [
        'raw_eeg_2d_efficientnetv2medium_384x512_quality_2',
        'raw_eeg_2d_coatlitemedium_384x512_quality_2',
        'raw_eeg_2d_nextvitlarge_384x512_quality_2',
        'raw_eeg_2d_maxvittiny_384x512_quality_2',
        'raw_eeg_2d_swinsmall_384x512_quality_2'
    ]
    for model_name, df in zip(model_names, prediction_dfs):
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
        settings.logger.info(f'{model_name} OOF Scores: {json.dumps(oof_scores, indent=2)}\n')

    df_blend = df_raw_eeg_2d_efficientnetv2medium.copy(deep=True)
    df_blend[prediction_columns] = (df_raw_eeg_2d_efficientnetv2medium[prediction_columns] * 0.2).values + \
                                   (df_raw_eeg_2d_coatlitemedium[prediction_columns] * 0.2).values + \
                                   (df_raw_eeg_2d_nextvitlarge[prediction_columns] * 0.2).values + \
                                   (df_raw_eeg_2d_maxvittiny[prediction_columns] * 0.2).values + \
                                   (df_raw_eeg_2d_swinsmall[prediction_columns] * 0.2).values

    df_blend = normalize_probabilities(df_blend, prediction_columns)
    oof_scores = metrics.multiclass_classification_scores(
        df_targets=df_blend.loc[:, target_columns],
        df_predictions=df_blend.loc[:, prediction_columns]
    )
    for sample_quality in [0, 1, 2]:
        sample_quality_mask = df_blend['sample_quality'] == sample_quality
        sample_quality_oof_scores = metrics.multiclass_classification_scores(
            df_targets=df_blend.loc[sample_quality_mask, target_columns],
            df_predictions=df_blend.loc[sample_quality_mask, prediction_columns]
        )
        sample_quality_oof_scores = {f'sq_{sample_quality}_{k}': v for k, v in sample_quality_oof_scores.items()}
        oof_scores.update(sample_quality_oof_scores)
    settings.logger.info(f'Blend OOF Scores: {json.dumps(oof_scores, indent=2)}\n')

    df_blend['seizure_vote_prediction'] *= 1.
    df_blend['lpd_vote_prediction'] *= 1.
    df_blend['gpd_vote_prediction'] *= 1.
    df_blend['lrda_vote_prediction'] *= 1.
    df_blend['grda_vote_prediction'] *= 1.
    df_blend['other_vote_prediction'] *= 1.
    df_blend = normalize_probabilities(df=df_blend, columns=prediction_columns)

    oof_scores = metrics.multiclass_classification_scores(
        df_targets=df_blend.loc[:, target_columns],
        df_predictions=df_blend.loc[:, prediction_columns]
    )
    for sample_quality in [0, 1, 2]:
        sample_quality_mask = df_blend['sample_quality'] == sample_quality
        sample_quality_oof_scores = metrics.multiclass_classification_scores(
            df_targets=df_blend.loc[sample_quality_mask, target_columns],
            df_predictions=df_blend.loc[sample_quality_mask, prediction_columns]
        )
        sample_quality_oof_scores = {f'sq_{sample_quality}_{k}': v for k, v in sample_quality_oof_scores.items()}
        oof_scores.update(sample_quality_oof_scores)
    settings.logger.info(f' OOF Scores: {json.dumps(oof_scores, indent=2)}\n')
