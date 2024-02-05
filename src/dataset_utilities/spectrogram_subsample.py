import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.append('..')
import settings


def get_spectrogram(df_spectrogram, start_time, fill_na=False, log_scale=False):

    """
    Get spectrogram subsample array from the given dataframe

    Parameters
    ----------
    df_spectrogram: pandas.DataFrame
        Dataframe with time and signal columns

    start_time: int
        Spectrogram offset seconds

    fill_na: bool
        Whether to fill missing values or not

    log_scale: bool
        Whether to do log transform or not

    Returns
    -------
    spectrogram: numpy.ndarray of shape (n_frequencies, n_time_steps, n_channels)
        Array of spectrogram
    """

    if start_time % 2 == 0:
        start_time += 1
    end_time = start_time + 598

    df_spectrogram_subsample = df_spectrogram.loc[(df_spectrogram['time'] >= start_time) & (df_spectrogram['time'] <= end_time)].iloc[:, 1:]

    if fill_na:
        df_spectrogram_subsample = df_spectrogram_subsample.fillna(0)

    spectrogram = df_spectrogram_subsample.values.T
    spectrogram = np.stack((
        spectrogram[0:100, :],
        spectrogram[100:200, :],
        spectrogram[200:300, :],
        spectrogram[300:400, :],
    ), axis=-1)

    if log_scale:
        spectrogram = np.log1p(spectrogram)

    return spectrogram


if __name__ == '__main__':

    dataset_directory = settings.DATA / 'spectrogram_subsample'
    dataset_directory.mkdir(parents=True, exist_ok=True)
    (dataset_directory / 'spectrograms').mkdir(parents=True, exist_ok=True)

    spectrogram_directory = settings.DATA / 'hms-harmful-brain-activity-classification' / 'train_spectrograms'
    spectrogram_file_names = os.listdir(spectrogram_directory)

    df_train = pd.read_csv(settings.DATA / 'hms-harmful-brain-activity-classification' / 'train.csv')
    metadata = []

    for spectrogram_id, df_train_spectrogram in tqdm(df_train.groupby('spectrogram_id'), total=df_train['spectrogram_id'].nunique()):

        df_spectrogram = pd.read_parquet(spectrogram_directory / f'{spectrogram_id}.parquet')

        for _, row in df_train_spectrogram.iterrows():

            spectrogram_sub_id = row['spectrogram_sub_id']
            start_time = int(row['spectrogram_label_offset_seconds'])

            spectrogram = get_spectrogram(df_spectrogram, start_time, fill_na=True, log_scale=False)
            spectrogram_file_path = dataset_directory / 'spectrograms' / f'{spectrogram_id}_{spectrogram_sub_id}.npy'
            np.save(spectrogram_file_path, spectrogram)

    df_metadata = pd.DataFrame(metadata)
    df_metadata.to_csv(dataset_directory / 'metadata.csv', index=False)
    settings.logger.info(f'metadata.csv is saved to {dataset_directory}')
