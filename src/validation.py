from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import settings


def create_folds(df, stratify_column, group_column, n_splits, shuffle=True, random_state=42, verbose=True):

    """
    Create columns of folds on given dataframe

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with given stratify and group columns

    stratify_column: str
        Name of the column to be stratified

    group_column: str
        Name of the column to be grouped

    n_splits: int
        Number of folds (2 <= n_splits)

    shuffle: bool
        Whether to shuffle before split or not

    random_state: int
        Random seed for reproducible results

    verbose: bool
        Verbosity flag

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with created fold columns
    """

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for fold, (training_idx, validation_idx) in enumerate(sgkf.split(X=df, y=df[stratify_column], groups=df[group_column]), 1):
        df.loc[training_idx, f'fold{fold}'] = 0
        df.loc[validation_idx, f'fold{fold}'] = 1
        df[f'fold{fold}'] = df[f'fold{fold}'].astype(np.uint8)

    if verbose:

        settings.logger.info(f'Dataset split into {n_splits} folds')

        validation_sizes = []
        target_value_counts = []
        sample_quality_value_counts = []

        for fold in range(1, n_splits + 1):
            df_fold = df[df[f'fold{fold}'] == 1]
            target_fold_value_counts = df_fold['expert_consensus'].value_counts(normalize=True).sort_index().to_dict()
            sample_quality_fold_value_counts = df_fold['sample_quality'].value_counts(normalize=True).sort_index().to_dict()
            settings.logger.info(f'Fold {fold} {df_fold.shape} - {target_fold_value_counts} - {sample_quality_fold_value_counts}')
            validation_sizes.append(df_fold.shape[0])
            target_value_counts.append(target_fold_value_counts)
            sample_quality_value_counts.append(sample_quality_fold_value_counts)

        validation_sizes = np.array(validation_sizes)
        target_value_counts = pd.DataFrame(target_value_counts)
        sample_quality_value_counts = pd.DataFrame(sample_quality_value_counts)
        settings.logger.info(f'Seed {random_state} Validation Size Std {np.std(validation_sizes):.4f} Target Value Counts Stds {target_value_counts.std().mean():.4f} Sample Quality Value Counts Stds {sample_quality_value_counts.std().mean():.4f}')

    return df


if __name__ == '__main__':

    df_train = pd.read_csv(settings.DATA / 'hms-harmful-brain-activity-classification' / 'train.csv')
    settings.logger.info(f'Train Dataset Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')

    # Extract stationary periods from vote patterns
    df_train['vote_pattern'] = df_train['seizure_vote'].astype(str) + df_train['lpd_vote'].astype(str) + df_train['gpd_vote'].astype(str) + df_train['lrda_vote'].astype(str) + df_train['grda_vote'].astype(str) + df_train['other_vote'].astype(str)
    previous_eeg_id, previous_vote_patern = df_train.loc[0, 'eeg_id'], df_train.loc[0, 'vote_pattern']
    stationary_period = 0
    for i, row in tqdm(df_train.iterrows(), total=len(df_train)):
        eeg_id, vote_patern = row['eeg_id'], row['vote_pattern']
        if eeg_id != previous_eeg_id or vote_patern != previous_vote_patern:
            stationary_period += 1
            previous_eeg_id = eeg_id
            previous_vote_patern = vote_patern
        df_train.loc[i, 'stationary_period'] = stationary_period

    # Select middle sample from each stationary period
    df_train = df_train.groupby(['patient_id', 'eeg_id', 'stationary_period']).apply(lambda x: x.iloc[(x.shape[0] // 2)], include_groups=False).reset_index()

    target_columns = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    df_train['total_vote'] = df_train[target_columns].sum(axis=1)
    df_train['sample_quality'] = 0
    df_train.loc[(df_train['total_vote'] >= 3) & (df_train['total_vote'] < 10), 'sample_quality'] = 1
    df_train.loc[(df_train['total_vote'] >= 10), 'sample_quality'] = 2

    df_train['stratify_column'] = df_train['sample_quality'].astype(str) + df_train['expert_consensus']

    n_splits = 5
    df_train = create_folds(
        df=df_train,
        stratify_column='sample_quality',
        group_column='patient_id',
        n_splits=n_splits,
        shuffle=True,
        random_state=94,
        verbose=True
    )

    id_columns = ['eeg_id', 'eeg_sub_id', 'spectrogram_id', 'spectrogram_sub_id']
    fold_columns = [f'fold{fold}' for fold in range(1, n_splits + 1)]
    df_train[id_columns + fold_columns].to_csv(settings.DATA / 'folds.csv', index=False)
    settings.logger.info(f'folds.csv is saved to {settings.DATA}')
