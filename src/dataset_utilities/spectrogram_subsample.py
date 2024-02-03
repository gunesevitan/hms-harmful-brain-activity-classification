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

    channels = False

    for spectrogram_id, df_train_spectrogram in tqdm(df_train.groupby('spectrogram_id'), total=df_train['spectrogram_id'].nunique()):

        df_spectrogram = pd.read_parquet(spectrogram_directory / f'{spectrogram_id}.parquet')
        spectrogram_columns = df_spectrogram.columns.tolist()
        left_temporal_chain = [column for column in spectrogram_columns if column.startswith('LL')]
        right_temporal_chain = [column for column in spectrogram_columns if column.startswith('RL')]
        left_parasagittal_chain = [column for column in spectrogram_columns if column.startswith('LP')]
        right_parasagittal_chain = [column for column in spectrogram_columns if column.startswith('RP')]

        for _, row in df_train_spectrogram.iterrows():

            spectrogram_sub_id = row['spectrogram_sub_id']
            start_time = int(row['spectrogram_label_offset_seconds'])

            if start_time % 2 == 0:
                start_time += 1
            end_time = start_time + 598

            df_spectrogram_subsample = df_spectrogram.loc[(df_spectrogram['time'] >= start_time) & (df_spectrogram['time'] <= end_time)].iloc[:, 1:]
            df_spectrogram_subsample = df_spectrogram_subsample.fillna(0)
            spectrogram_file_path = dataset_directory / 'spectrograms' / f'{spectrogram_id}_{spectrogram_sub_id}.npy'

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

            if channels:
                spectrogram = np.stack([
                    df_spectrogram_subsample[left_temporal_chain].values,
                    df_spectrogram_subsample[right_temporal_chain].values,
                    df_spectrogram_subsample[left_parasagittal_chain].values,
                    df_spectrogram_subsample[right_parasagittal_chain].values
                ], axis=-1).transpose(1, 0, 2)
            else:
                spectrogram = df_spectrogram_subsample.values.T

            np.save(spectrogram_file_path, spectrogram)

    df_metadata = pd.DataFrame(metadata)
    df_metadata.to_csv(dataset_directory / 'metadata.csv', index=False)
    settings.logger.info(f'metadata.csv is saved to {dataset_directory}')
