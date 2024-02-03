import sys
import json
import numpy as np
import pandas as pd

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

    model_directory = 'efficientnetv2tiny_spectrogram_v1.14'

    df = pd.read_csv(settings.MODELS / model_directory / 'oof_predictions.csv')

    # Read and merge precomputed folds
    df_folds = pd.read_csv(settings.DATA / 'folds.csv')
    df = df.merge(df_folds, on=['eeg_id', 'eeg_sub_id', 'spectrogram_id', 'spectrogram_sub_id'], how='left')
    del df_folds

    settings.logger.info(f'OOF Predictions Shape: {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

    target_columns = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    prediction_columns = [f'{column}_prediction' for column in target_columns]

    oof_scores = metrics.multiclass_classification_scores(
        df_targets=df.loc[:, target_columns],
        df_predictions=df.loc[:, prediction_columns]
    )
    settings.logger.info(f'OOF Scores: {json.dumps(oof_scores, indent=2)}')

    df['seizure_vote_prediction'] *= 1.
    df['lpd_vote_prediction'] *= 1.
    df['gpd_vote_prediction'] *= 1.25
    df['lrda_vote_prediction'] *= 1.75
    df['grda_vote_prediction'] *= 1.25
    df['other_vote_prediction'] *= 1.
    df = normalize_probabilities(df=df, columns=prediction_columns)

    df_scores = []

    for fold in ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']:

        training_idx, validation_idx = df.loc[df[fold] != 1].index, df.loc[df[fold] == 1].index
        validation_scores = metrics.multiclass_classification_scores(
            df_targets=df.loc[validation_idx, target_columns],
            df_predictions=df.loc[validation_idx, prediction_columns]
        )
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

    oof_scores = metrics.multiclass_classification_scores(
        df_targets=df.loc[:, target_columns],
        df_predictions=df.loc[:, prediction_columns]
    )
    settings.logger.info(f'OOF Scores: {json.dumps(oof_scores, indent=2)}')
