import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.append('..')
import settings


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
        df_spectrogram = df_spectrogram.interpolate(method='linear', limit_area='inside')
        df_spectrogram = df_spectrogram.fillna(0)

        for _, row in df_train_spectrogram.iterrows():

            spectrogram_sub_id = row['spectrogram_sub_id']
            start_time = int(row['spectrogram_label_offset_seconds'])

            if start_time % 2 == 0:
                start_time += 1
            end_time = start_time + 598

            df_spectrogram_subsample = df_spectrogram.loc[(df_spectrogram['time'] >= start_time) & (df_spectrogram['time'] <= end_time)].iloc[:, 1:]

            nan_counts = df_spectrogram_subsample.isnull().sum().to_dict()
            nan_counts = {f'{k}_nan_count': v for k, v in nan_counts.items()}

            metadata_dict = {
                'spectrogram_id': spectrogram_id,
                'spectrogram_sub_id': spectrogram_sub_id,
                'row_count': df_spectrogram_subsample.shape[0],
                'column_count': df_spectrogram_subsample.shape[1]
            }
            metadata_dict.update(nan_counts)
            metadata.append(metadata_dict)

            spectrogram = df_spectrogram_subsample.values.T
            spectrogram_file_path = dataset_directory / 'spectrograms' / f'{spectrogram_id}_{spectrogram_sub_id}.npy'
            np.save(spectrogram_file_path, spectrogram)

    df_metadata = pd.DataFrame(metadata)
    df_metadata.to_csv(dataset_directory / 'metadata.csv', index=False)
    settings.logger.info(f'metadata.csv is saved to {dataset_directory}')
