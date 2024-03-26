import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.append('..')
import settings
from eeg_to_multitaper_spectrogram import multitaper_spectrogram


def get_eeg_differences(eeg):
    eeg_difference = np.zeros((eeg.shape[0], 18))

    # Left outside temporal chain
    eeg_difference[:, 0] = eeg[:, 0] - eeg[:, 4]
    eeg_difference[:, 1] = eeg[:, 4] - eeg[:, 5]
    eeg_difference[:, 2] = eeg[:, 5] - eeg[:, 6]
    eeg_difference[:, 3] = eeg[:, 6] - eeg[:, 7]

    # Right outside temporal chain
    eeg_difference[:, 4] = eeg[:, 11] - eeg[:, 15]
    eeg_difference[:, 5] = eeg[:, 15] - eeg[:, 16]
    eeg_difference[:, 6] = eeg[:, 16] - eeg[:, 17]
    eeg_difference[:, 7] = eeg[:, 17] - eeg[:, 18]

    # Left inside parasagittal chain
    eeg_difference[:, 8] = eeg[:, 0] - eeg[:, 1]
    eeg_difference[:, 9] = eeg[:, 1] - eeg[:, 2]
    eeg_difference[:, 10] = eeg[:, 2] - eeg[:, 3]
    eeg_difference[:, 11] = eeg[:, 3] - eeg[:, 7]

    # Right inside parasagittal chain
    eeg_difference[:, 12] = eeg[:, 11] - eeg[:, 12]
    eeg_difference[:, 13] = eeg[:, 12] - eeg[:, 13]
    eeg_difference[:, 14] = eeg[:, 13] - eeg[:, 14]
    eeg_difference[:, 15] = eeg[:, 14] - eeg[:, 18]

    # Center chain
    eeg_difference[:, 16] = eeg[:, 8] - eeg[:, 9]
    eeg_difference[:, 17] = eeg[:, 9] - eeg[:, 10]

    return eeg_difference


if __name__ == '__main__':

    dataset_directory = settings.DATA / 'eeg_multitaper_spectrograms'
    dataset_directory.mkdir(parents=True, exist_ok=True)
    (dataset_directory / 'spectrograms').mkdir(parents=True, exist_ok=True)
    (dataset_directory / 'spectrograms_center').mkdir(parents=True, exist_ok=True)

    eeg_directory = settings.DATA / 'hms-harmful-brain-activity-classification' / 'train_eegs'
    eeg_file_names = os.listdir(eeg_directory)

    df_train = pd.read_csv(settings.DATA / 'hms-harmful-brain-activity-classification' / 'train.csv')

    for eeg_id, df_train_eeg in tqdm(df_train.groupby('eeg_id'), total=df_train['eeg_id'].nunique()):

        df_eeg = pd.read_parquet(eeg_directory / f'{eeg_id}.parquet')
        # df_eeg = df_eeg.interpolate(method='linear', limit_area='inside').fillna(0)
        df_eeg = df_eeg.fillna(0)

        eeg_label_offset_seconds = df_train_eeg['eeg_label_offset_seconds'].tolist()
        eeg_label_offset_seconds.sort()

        start_idx = int(eeg_label_offset_seconds[0] * 200)
        end_idx = int((eeg_label_offset_seconds[-1] + 50) * 200)

        eeg = df_eeg.iloc[start_idx:end_idx].reset_index(drop=True).values
        eeg = get_eeg_differences(eeg)

        spectrograms, spectrograms_center = [], []
        for signal_idx in range(eeg.shape[1]):
            spect, stimes, sfreqs = multitaper_spectrogram(
                eeg[:, signal_idx],
                fs=200,
                frequency_range=[0.5, 20],
                window_params=[4, 0.5],
                multiprocess=True,
                plot_on=False,
                verbose=False,
            )
            spectrograms.append(spect)

            spect_center, stimes_center, sfreqs_center = multitaper_spectrogram(
                eeg[4000:6000, signal_idx],
                fs=200,
                frequency_range=[0.5, 20],
                window_params=[4, 0.5],
                multiprocess=True,
                plot_on=False,
                verbose=False,
            )
            spectrograms_center.append(spect_center)

        spectrograms = np.concatenate(spectrograms, axis=0)
        spectrograms_center = np.concatenate(spectrograms_center, axis=0)

        for i, row in df_train_eeg.iterrows():
            eeg_sub_id = row['eeg_sub_id']
            spectrogram_file_path = dataset_directory / 'spectrograms' / f'{eeg_id}_{eeg_sub_id}.npy'
            spectrogram_center_file_path = dataset_directory / 'spectrograms_center' / f'{eeg_id}_{eeg_sub_id}.npy'

            sp_idx = int((eeg_label_offset_seconds[i] - eeg_label_offset_seconds[0]) * 2)
            np.save(spectrogram_file_path, spectrograms[:, :, sp_idx + sp_idx + 93])

            sp_center_idx = i * 20
            np.save(spectrogram_center_file_path, spectrograms_center[:, :, sp_center_idx + sp_center_idx + 13])

            # spectrograms = np.log1p(spectrograms)
            # mean = spectrograms.mean()
            # std = spectrograms.std()
            # min = spectrograms.min()
            # max = spectrograms.max()
            # spec_id = str(spectrogram_file_path).split('/')[-1]
            # visualization.visualize_spectrogram(np.log1p(spectrograms), f'Spectrogram {spec_id} - Mean: {mean:.2f} Std: {std:.2f} Min: {min:.2f} Max: {max:.2f}')
            # exit()