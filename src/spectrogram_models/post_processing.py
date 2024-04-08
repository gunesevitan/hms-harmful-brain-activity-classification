import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import preprocessing
sys.path.append('..')
import settings
import metrics


def visualize_correlations(df, columns, title, path=None):
    """
    Visualize correlations of given columns in given dataframe

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with given column

    columns: list
        List of names of columns

    title: str
        Title of the plot

    path: path-like str or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(20, 20), dpi=100)
    ax = sns.heatmap(
        df[columns].corr(),
        annot=True,
        square=True,
        cmap='coolwarm',
        annot_kws={'size': 6},
        fmt='.2f'
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_title(title, size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


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

    raw_eeg_50_second_2d_convnextbase_directory = 'raw_eeg_50_second_2d_convnextbase_384x512_quality_2'
    raw_eeg_50_second_2d_maxvittiny_directory = 'raw_eeg_50_second_2d_maxvittiny_384x512_quality_2'
    raw_eeg_50_10_second_2d_convnextbase_directory = 'raw_eeg_50_10_second_2d_convnextbase_448x512_quality_2'
    raw_eeg_50_10_second_2d_maxvittiny_directory = 'raw_eeg_50_10_second_2d_maxvittiny_448x512_quality_2'
    raw_eeg_50_30_10_second_2d_convnextbase_directory = 'raw_eeg_50_30_10_second_2d_convnextbase_672x512_quality_2'
    raw_eeg_50_30_10_second_2d_maxvittiny_directory = 'raw_eeg_50_30_10_second_2d_maxvittiny_672x512_quality_2'

    spectrogram_50_second_2d_convnextbase_directory = 'spectrogram_50_second_2d_convnextbase_512x512_quality_2'
    spectrogram_50_second_2d_maxvittiny_directory = 'spectrogram_50_second_2d_maxvittiny_512x512_quality_2'
    spectrogram_50_10_second_2d_convnextbase_directory = 'spectrogram_50_10_second_2d_convnextbase_512x512_quality_2'
    spectrogram_50_10_second_2d_maxvittiny_directory = 'spectrogram_50_10_second_2d_maxvittiny_512x512_quality_2'
    spectrogram_30_10_second_2d_convnextbase_directory = 'spectrogram_30_10_second_2d_convnextbase_512x512_quality_2'
    spectrogram_30_10_second_2d_maxvittiny_directory = 'spectrogram_30_10_second_2d_maxvittiny_512x512_quality_2'
    spectrogram_50_30_10_second_2d_convnextbase_directory = 'spectrogram_50_30_10_second_2d_convnextbase_512x512_quality_2'
    spectrogram_50_30_10_second_2d_maxvittiny_directory = 'spectrogram_50_30_10_second_2d_maxvittiny_512x512_quality_2'

    multitaper_spectrogram_50_second_2d_convnextbase_directory = 'multitaper_spectrogram_50_second_2d_convnextbase_512x384_quality_2'
    multitaper_spectrogram_50_second_2d_maxvittiny_directory = 'multitaper_spectrogram_50_second_2d_maxvittiny_512x384_quality_2'
    multitaper_spectrogram_50_10_second_2d_convnextbase_directory = 'multitaper_spectrogram_50_10_second_2d_convnextbase_512x448_quality_2'
    multitaper_spectrogram_50_10_second_2d_maxvittiny_directory = 'multitaper_spectrogram_50_10_second_2d_maxvittiny_512x448_quality_2'

    prediction_columns = [f'{column}_prediction' for column in target_columns]
    columns_to_merge = ['eeg_id', 'eeg_sub_id', 'sample_quality']

    df_raw_eeg_50_second_2d_convnextbase = pd.read_csv(settings.MODELS / raw_eeg_50_second_2d_convnextbase_directory / 'oof_predictions.csv')
    df_raw_eeg_50_second_2d_convnextbase = df_raw_eeg_50_second_2d_convnextbase.merge(df.loc[:, columns_to_merge], on=columns_to_merge[:-1], how='left')
    df_raw_eeg_50_second_2d_maxvittiny = pd.read_csv(settings.MODELS / raw_eeg_50_second_2d_maxvittiny_directory / 'oof_predictions.csv')
    df_raw_eeg_50_second_2d_maxvittiny = df_raw_eeg_50_second_2d_maxvittiny.merge(df.loc[:, columns_to_merge], on=columns_to_merge[:-1], how='left')

    df_raw_eeg_50_10_second_2d_convnextbase = pd.read_csv(settings.MODELS / raw_eeg_50_10_second_2d_convnextbase_directory / 'oof_predictions.csv')
    df_raw_eeg_50_10_second_2d_convnextbase = df_raw_eeg_50_10_second_2d_convnextbase.merge(df.loc[:, columns_to_merge], on=columns_to_merge[:-1], how='left')
    df_raw_eeg_50_10_second_2d_maxvittiny = pd.read_csv(settings.MODELS / raw_eeg_50_10_second_2d_maxvittiny_directory / 'oof_predictions.csv')
    df_raw_eeg_50_10_second_2d_maxvittiny = df_raw_eeg_50_10_second_2d_maxvittiny.merge(df.loc[:, columns_to_merge], on=columns_to_merge[:-1], how='left')

    df_raw_eeg_50_30_10_second_2d_convnextbase = pd.read_csv(settings.MODELS / raw_eeg_50_30_10_second_2d_convnextbase_directory / 'oof_predictions.csv')
    df_raw_eeg_50_30_10_second_2d_convnextbase = df_raw_eeg_50_30_10_second_2d_convnextbase.merge(df.loc[:, columns_to_merge], on=columns_to_merge[:-1], how='left')
    df_raw_eeg_50_30_10_second_2d_maxvittiny = pd.read_csv(settings.MODELS / raw_eeg_50_30_10_second_2d_maxvittiny_directory / 'oof_predictions.csv')
    df_raw_eeg_50_30_10_second_2d_maxvittiny = df_raw_eeg_50_30_10_second_2d_maxvittiny.merge(df.loc[:, columns_to_merge], on=columns_to_merge[:-1], how='left')

    df_spectrogram_50_second_2d_convnextbase = pd.read_csv(settings.MODELS / spectrogram_50_second_2d_convnextbase_directory / 'oof_predictions.csv')
    df_spectrogram_50_second_2d_convnextbase = df_spectrogram_50_second_2d_convnextbase.merge(df.loc[:, columns_to_merge], on=columns_to_merge[:-1], how='left')
    df_spectrogram_50_second_2d_maxvittiny = pd.read_csv(settings.MODELS / spectrogram_50_second_2d_maxvittiny_directory / 'oof_predictions.csv')
    df_spectrogram_50_second_2d_maxvittiny = df_spectrogram_50_second_2d_maxvittiny.merge(df.loc[:, columns_to_merge], on=columns_to_merge[:-1], how='left')

    df_spectrogram_50_10_second_2d_convnextbase = pd.read_csv(settings.MODELS / spectrogram_50_10_second_2d_convnextbase_directory / 'oof_predictions.csv')
    df_spectrogram_50_10_second_2d_convnextbase = df_spectrogram_50_10_second_2d_convnextbase.merge(df.loc[:, columns_to_merge], on=columns_to_merge[:-1], how='left')
    df_spectrogram_50_10_second_2d_maxvittiny = pd.read_csv(settings.MODELS / spectrogram_50_10_second_2d_maxvittiny_directory / 'oof_predictions.csv')
    df_spectrogram_50_10_second_2d_maxvittiny = df_spectrogram_50_10_second_2d_maxvittiny.merge(df.loc[:, columns_to_merge], on=columns_to_merge[:-1], how='left')

    df_spectrogram_30_10_second_2d_convnextbase = pd.read_csv(settings.MODELS / spectrogram_30_10_second_2d_convnextbase_directory / 'oof_predictions.csv')
    df_spectrogram_30_10_second_2d_convnextbase = df_spectrogram_30_10_second_2d_convnextbase.merge(df.loc[:, columns_to_merge], on=columns_to_merge[:-1], how='left')
    df_spectrogram_30_10_second_2d_maxvittiny = pd.read_csv(settings.MODELS / spectrogram_30_10_second_2d_maxvittiny_directory / 'oof_predictions.csv')
    df_spectrogram_30_10_second_2d_maxvittiny = df_spectrogram_30_10_second_2d_maxvittiny.merge(df.loc[:, columns_to_merge], on=columns_to_merge[:-1], how='left')

    df_spectrogram_50_30_10_second_2d_convnextbase = pd.read_csv(settings.MODELS / spectrogram_50_30_10_second_2d_convnextbase_directory / 'oof_predictions.csv')
    df_spectrogram_50_30_10_second_2d_convnextbase = df_spectrogram_50_30_10_second_2d_convnextbase.merge(df.loc[:, columns_to_merge], on=columns_to_merge[:-1], how='left')
    df_spectrogram_50_30_10_second_2d_maxvittiny = pd.read_csv(settings.MODELS / spectrogram_50_30_10_second_2d_maxvittiny_directory / 'oof_predictions.csv')
    df_spectrogram_50_30_10_second_2d_maxvittiny = df_spectrogram_50_30_10_second_2d_maxvittiny.merge(df.loc[:, columns_to_merge], on=columns_to_merge[:-1], how='left')

    df_multitaper_spectrogram_50_second_2d_convnextbase = pd.read_csv(settings.MODELS / multitaper_spectrogram_50_second_2d_convnextbase_directory / 'oof_preds_tta.csv').drop(columns=['sample_quality'])
    df_multitaper_spectrogram_50_second_2d_convnextbase = df_multitaper_spectrogram_50_second_2d_convnextbase.merge(df.loc[:, columns_to_merge], on=columns_to_merge[:-1], how='left')
    df_multitaper_spectrogram_50_second_2d_maxvittiny = pd.read_csv(settings.MODELS / multitaper_spectrogram_50_second_2d_maxvittiny_directory / 'oof_preds_tta.csv').drop(columns=['sample_quality'])
    df_multitaper_spectrogram_50_second_2d_maxvittiny = df_multitaper_spectrogram_50_second_2d_maxvittiny.merge(df.loc[:, columns_to_merge], on=columns_to_merge[:-1], how='left')

    df_multitaper_spectrogram_50_10_second_2d_convnextbase = pd.read_csv(settings.MODELS / multitaper_spectrogram_50_10_second_2d_convnextbase_directory / 'oof_preds_tta.csv').drop(columns=['sample_quality'])
    df_multitaper_spectrogram_50_10_second_2d_convnextbase = df_multitaper_spectrogram_50_10_second_2d_convnextbase.merge(df.loc[:, columns_to_merge], on=columns_to_merge[:-1], how='left')
    df_multitaper_spectrogram_50_10_second_2d_maxvittiny = pd.read_csv(settings.MODELS / multitaper_spectrogram_50_10_second_2d_maxvittiny_directory / 'oof_preds_tta.csv').drop(columns=['sample_quality'])
    df_multitaper_spectrogram_50_10_second_2d_maxvittiny = df_multitaper_spectrogram_50_10_second_2d_maxvittiny.merge(df.loc[:, columns_to_merge], on=columns_to_merge[:-1], how='left')

    prediction_dfs = [
        df_raw_eeg_50_second_2d_convnextbase,
        df_raw_eeg_50_second_2d_maxvittiny,
        df_raw_eeg_50_10_second_2d_convnextbase,
        df_raw_eeg_50_10_second_2d_maxvittiny,
        df_raw_eeg_50_30_10_second_2d_convnextbase,
        df_raw_eeg_50_30_10_second_2d_maxvittiny,

        df_spectrogram_50_second_2d_convnextbase,
        df_spectrogram_50_second_2d_maxvittiny,
        df_spectrogram_50_10_second_2d_convnextbase,
        df_spectrogram_50_10_second_2d_maxvittiny,
        df_spectrogram_30_10_second_2d_convnextbase,
        df_spectrogram_30_10_second_2d_maxvittiny,
        df_spectrogram_50_30_10_second_2d_convnextbase,
        df_spectrogram_50_30_10_second_2d_maxvittiny,

        df_multitaper_spectrogram_50_second_2d_convnextbase,
        df_multitaper_spectrogram_50_second_2d_maxvittiny,
        df_multitaper_spectrogram_50_10_second_2d_convnextbase,
        df_multitaper_spectrogram_50_10_second_2d_maxvittiny,
    ]
    model_names = [
        'raw_eeg_50_second_2d_convnextbase_384x512_quality_2',
        'raw_eeg_50_second_2d_maxvittiny_384x512_quality_2',
        'raw_eeg_50_10_second_2d_convnextbase_448x512_quality_2',
        'raw_eeg_50_10_second_2d_maxvittiny_448x512_quality_2',
        'raw_eeg_50_30_10_second_2d_convnext_672x512_quality_2',
        'raw_eeg_50_30_10_second_2d_maxvittiny_672x512_quality_2',

        'spectrogram_50_second_2d_convnextbase_512x512_quality_2',
        'spectrogram_50_second_2d_maxvittiny_512x512_quality_2',
        'spectrogram_50_10_second_2d_convnextbase_512x512_quality_2',
        'spectrogram_50_10_second_2d_maxvittiny_512x480_quality_2',
        'spectrogram_30_10_second_2d_convnextbase_512x512_quality_2',
        'spectrogram_30_10_second_2d_maxvittiny_512x512_quality_2',
        'spectrogram_50_30_10_second_2d_convnextbase_512x512_quality_2',
        'spectrogram_50_30_10_second_2d_maxvittiny_512x512_quality_2',

        'multitaper_spectrogram_50_second_2d_convnextbase_512x384_quality_2',
        'multitaper_spectrogram_50_second_2d_maxvittiny_512x384_quality_2',
        'multitaper_spectrogram_50_10_second_2d_convnextbase_512x448_quality_2',
        'multitaper_spectrogram_50_10_second_2d_maxvittiny_512x448_quality_2',
    ]

    predictions_flat = np.stack([prediction_df[prediction_columns].values.flatten() for prediction_df in prediction_dfs], axis=-1)
    predictions_flat = pd.DataFrame(predictions_flat, columns=model_names)
    visualize_correlations(
        df=predictions_flat,
        columns=model_names,
        title='Model Correlations'
    )

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

    df_raw_eeg_blend = df_raw_eeg_50_second_2d_convnextbase.copy(deep=True)
    df_raw_eeg_blend[prediction_columns] = (df_raw_eeg_50_second_2d_convnextbase[prediction_columns] * 0.166).values + \
                                           (df_raw_eeg_50_second_2d_maxvittiny[prediction_columns] * 0.166).values + \
                                           (df_raw_eeg_50_10_second_2d_convnextbase[prediction_columns] * 0.170).values + \
                                           (df_raw_eeg_50_10_second_2d_maxvittiny[prediction_columns] * 0.166).values + \
                                           (df_raw_eeg_50_30_10_second_2d_convnextbase[prediction_columns] * 0.166).values + \
                                           (df_raw_eeg_50_30_10_second_2d_maxvittiny[prediction_columns] * 0.166).values

    df_raw_eeg_blend = normalize_probabilities(df_raw_eeg_blend, prediction_columns)
    oof_scores = metrics.multiclass_classification_scores(
        df_targets=df_raw_eeg_blend.loc[:, target_columns],
        df_predictions=df_raw_eeg_blend.loc[:, prediction_columns]
    )
    for sample_quality in [0, 1, 2]:
        sample_quality_mask = df_raw_eeg_blend['sample_quality'] == sample_quality
        sample_quality_oof_scores = metrics.multiclass_classification_scores(
            df_targets=df_raw_eeg_blend.loc[sample_quality_mask, target_columns],
            df_predictions=df_raw_eeg_blend.loc[sample_quality_mask, prediction_columns]
        )
        sample_quality_oof_scores = {f'sq_{sample_quality}_{k}': v for k, v in sample_quality_oof_scores.items()}
        oof_scores.update(sample_quality_oof_scores)
    settings.logger.info(f'Raw EEG Blend OOF Scores: {json.dumps(oof_scores, indent=2)}\n')

    df_spectrogram_blend = df_spectrogram_50_second_2d_convnextbase.copy(deep=True)
    df_spectrogram_blend[prediction_columns] = (df_spectrogram_50_second_2d_convnextbase[prediction_columns] * 0.10).values + \
                                               (df_spectrogram_50_second_2d_maxvittiny[prediction_columns] * 0.10).values + \
                                               (df_spectrogram_50_10_second_2d_convnextbase[prediction_columns] * 0.15).values + \
                                               (df_spectrogram_50_10_second_2d_maxvittiny[prediction_columns] * 0.15).values + \
                                               (df_spectrogram_30_10_second_2d_convnextbase[prediction_columns] * 0.15).values + \
                                               (df_spectrogram_30_10_second_2d_maxvittiny[prediction_columns] * 0.15).values + \
                                               (df_spectrogram_50_30_10_second_2d_convnextbase[prediction_columns] * 0.10).values + \
                                               (df_spectrogram_50_30_10_second_2d_maxvittiny[prediction_columns] * 0.10).values

    df_spectrogram_blend = normalize_probabilities(df_spectrogram_blend, prediction_columns)
    oof_scores = metrics.multiclass_classification_scores(
        df_targets=df_spectrogram_blend.loc[:, target_columns],
        df_predictions=df_spectrogram_blend.loc[:, prediction_columns]
    )
    for sample_quality in [0, 1, 2]:
        sample_quality_mask = df_spectrogram_blend['sample_quality'] == sample_quality
        sample_quality_oof_scores = metrics.multiclass_classification_scores(
            df_targets=df_spectrogram_blend.loc[sample_quality_mask, target_columns],
            df_predictions=df_spectrogram_blend.loc[sample_quality_mask, prediction_columns]
        )
        sample_quality_oof_scores = {f'sq_{sample_quality}_{k}': v for k, v in sample_quality_oof_scores.items()}
        oof_scores.update(sample_quality_oof_scores)
    settings.logger.info(f'Spectrogram Blend OOF Scores: {json.dumps(oof_scores, indent=2)}\n')

    df_multitaper_spectrogram_blend = df_spectrogram_50_second_2d_convnextbase.copy(deep=True)
    df_multitaper_spectrogram_blend[prediction_columns] = (df_multitaper_spectrogram_50_second_2d_convnextbase[prediction_columns] * 0.25).values + \
                                                          (df_multitaper_spectrogram_50_second_2d_maxvittiny[prediction_columns] * 0.25).values + \
                                                          (df_multitaper_spectrogram_50_10_second_2d_convnextbase[prediction_columns] * 0.25).values + \
                                                          (df_multitaper_spectrogram_50_10_second_2d_maxvittiny[prediction_columns] * 0.25).values

    df_multitaper_spectrogram_blend = normalize_probabilities(df_multitaper_spectrogram_blend, prediction_columns)
    oof_scores = metrics.multiclass_classification_scores(
        df_targets=df_multitaper_spectrogram_blend.loc[:, target_columns],
        df_predictions=df_multitaper_spectrogram_blend.loc[:, prediction_columns]
    )
    for sample_quality in [0, 1, 2]:
        sample_quality_mask = df_multitaper_spectrogram_blend['sample_quality'] == sample_quality
        sample_quality_oof_scores = metrics.multiclass_classification_scores(
            df_targets=df_multitaper_spectrogram_blend.loc[sample_quality_mask, target_columns],
            df_predictions=df_multitaper_spectrogram_blend.loc[sample_quality_mask, prediction_columns]
        )
        sample_quality_oof_scores = {f'sq_{sample_quality}_{k}': v for k, v in sample_quality_oof_scores.items()}
        oof_scores.update(sample_quality_oof_scores)
    settings.logger.info(f'Multitaper Spectrogram Blend OOF Scores: {json.dumps(oof_scores, indent=2)}\n')

    df_blend = df_raw_eeg_50_second_2d_convnextbase.copy(deep=True)
    df_blend[prediction_columns] = (df_raw_eeg_blend[prediction_columns] * 0.25).values + \
                                   (df_spectrogram_blend[prediction_columns] * 0.6).values + \
                                   (df_multitaper_spectrogram_blend[prediction_columns] * 0.1).values

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

    #for column in prediction_columns:
    #    df_blend.loc[df_blend[column] > 0.8, column] = 1
    #df_blend = normalize_probabilities(df=df_blend, columns=prediction_columns)

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

    df_blend.to_csv(settings.DATA / 'oof_pseudo_labels.csv', index=False)