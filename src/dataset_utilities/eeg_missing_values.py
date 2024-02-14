import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno

sys.path.append('..')
import settings


def visualize_missingno_matrix(df_eeg, path):

    """
    Visualize missing values in EEG dataframes

    Parameters
    ----------
    df_eeg: pandas.DataFrame of shape (10000, 20)
        Dataframe of raw EEG signals

    path: str, pathlib.Path or None
        Path of the output file or None (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(24, 6), dpi=100)
    msno.matrix(df_eeg)

    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':

    dataset_directory = settings.DATA / 'eeg_subsample'
    dataset_directory.mkdir(parents=True, exist_ok=True)
    (dataset_directory / 'eegs').mkdir(parents=True, exist_ok=True)
    eeg_directory = dataset_directory / 'eegs'

    visualization_directory = settings.EDA / 'eeg_missing_values'
    visualization_directory.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(settings.DATA / 'hms-harmful-brain-activity-classification' / 'train.csv')
    df_train = df_train.merge(
        pd.read_csv(dataset_directory / 'metadata.csv')[['eeg_id', 'eeg_sub_id', 'Fp1_nan_count']],
        on=['eeg_id', 'eeg_sub_id'],
        how='left'
    )
    # Missing values 1, 2 and 3 are scattered around, and they can be interpolated
    # Missing values greater than 3 are huge blocks in the beginning or end but not both of them at the same time
    # There is one exception which is 18 missing values that occurs in the middle, and it can be interpolated as well
    df_train = df_train.loc[df_train['Fp1_nan_count'] > 3]
    df_train = df_train.sort_values(by='Fp1_nan_count', ascending=True).reset_index(drop=True)

    for idx, row in tqdm(df_train.iterrows(), total=df_train.shape[0]):

        eeg_id = row['eeg_id']
        eeg_sub_id = row['eeg_sub_id']

        df_eeg = pd.DataFrame(np.load(eeg_directory / f'{eeg_id}_{eeg_sub_id}.npy'))
        missing_value_count = df_eeg.loc[:, 0].isnull().sum()
        visualize_missingno_matrix(df_eeg=df_eeg, path=visualization_directory / f'{missing_value_count}_{eeg_id}_{eeg_sub_id}.png')
