import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import cusignal

sys.path.append('..')
import settings
import visualization


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


def get_spectrogram(signal, fs, nperseg, noverlap, frequency_range):

    frequencies, _, spectrogram = cusignal.spectrogram(
        signal,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=None
    )
    frequency_mask = (frequencies >= frequency_range[0]) & (frequencies <= frequency_range[1])
    spectrogram = spectrogram[frequency_mask, :]
    spectrogram = spectrogram.get().astype(np.float32)

    return spectrogram


if __name__ == '__main__':

    dataset_directory = settings.DATA / 'eeg_spectrograms-50-30-10-second'
    dataset_directory.mkdir(parents=True, exist_ok=True)
    (dataset_directory / 'spectrograms').mkdir(parents=True, exist_ok=True)

    eeg_directory = settings.DATA / 'hms-harmful-brain-activity-classification' / 'train_eegs'
    eeg_file_names = os.listdir(eeg_directory)

    df_train = pd.read_csv(settings.DATA / 'hms-harmful-brain-activity-classification' / 'train.csv')

    time_stack = '50/30/10'
    visualize = False

    for eeg_id, df_train_eeg in tqdm(df_train.groupby('eeg_id'), total=df_train['eeg_id'].nunique()):

        df_eeg = pd.read_parquet(eeg_directory / f'{eeg_id}.parquet')
        df_eeg = df_eeg.interpolate(method='linear', limit_area='inside').fillna(0)

        for _, row in df_train_eeg.iterrows():

            eeg_sub_id = row['eeg_sub_id']
            start_idx = int(row['eeg_label_offset_seconds'] * 200)
            end_idx = int((row['eeg_label_offset_seconds'] + 50) * 200)
            df_eeg_subsample = df_eeg.iloc[start_idx:end_idx].reset_index(drop=True)
            spectrogram_file_path = dataset_directory / 'spectrograms' / f'{eeg_id}_{eeg_sub_id}.npy'

            eeg = df_eeg_subsample.values
            eeg = get_eeg_differences(eeg)

            spectrograms = []

            for signal_idx in range(eeg.shape[1]):

                if time_stack == '50/30/10':

                    # 50 second spectrogram is created and trimmed on the x-axis for 20 pixels
                    spectrogram_50 = get_spectrogram(
                        signal=eeg[:, signal_idx],
                        fs=200,
                        nperseg=99,
                        noverlap=79,
                        frequency_range=(0.5, 20),
                    )[:, 10:-10]
                    spectrograms.append(spectrogram_50)

                    # Center 30 second spectrogram is created and trimmed on the x-axis for 16 pixels
                    spectrogram_30 = get_spectrogram(
                        signal=eeg[2000:8000, signal_idx],
                        fs=200,
                        nperseg=99,
                        noverlap=87,
                        frequency_range=(0.5, 20),
                    )[:, 8:-8]
                    spectrograms.append(spectrogram_30)

                    # Center 10 second spectrogram with the highest time resolution is created
                    spectrogram_10 = get_spectrogram(
                        signal=eeg[4000:6000, signal_idx],
                        fs=200,
                        nperseg=100,
                        noverlap=96,
                        frequency_range=(0.5, 20),
                    )
                    spectrograms.append(spectrogram_10)

                elif time_stack == '50/10':
                    # 50 second spectrogram is created and trimmed on the x-axis for 6 pixels
                    spectrogram_50 = get_spectrogram(
                        signal=eeg[:, signal_idx],
                        fs=200,
                        nperseg=145,
                        noverlap=124,
                        frequency_range=(0.5, 20),
                    )[:, 3:-3]
                    spectrograms.append(spectrogram_50)

                    # Center 10 second spectrogram with the highest time resolution is created
                    spectrogram_10 = get_spectrogram(
                        signal=eeg[4000:6000, signal_idx],
                        fs=200,
                        nperseg=145,
                        noverlap=141,
                        frequency_range=(0.5, 20),
                    )
                    spectrograms.append(spectrogram_10)
                elif time_stack == '50':
                    # 50 second spectrogram is created
                    spectrogram_50 = get_spectrogram(
                        signal=eeg[:, signal_idx],
                        fs=200,
                        nperseg=280,
                        noverlap=260,
                        frequency_range=(0.5, 20),

                    )
                    spectrograms.append(spectrogram_50)

            spectrograms = np.concatenate(spectrograms, axis=0)

            np.save(spectrogram_file_path, spectrograms)

            if visualize:
                spectrograms = np.log1p(spectrograms)
                mean = spectrograms.mean()
                std = spectrograms.std()
                min = spectrograms.min()
                max = spectrograms.max()
                spectrogram_id = str(spectrogram_file_path).split('/')[-1]
                visualization.visualize_spectrogram(
                    spectrograms,
                    f'Spectrogram {spectrogram_id} - Mean: {mean:.2f} Std: {std:.2f} Min: {min:.2f} Max: {max:.2f}'
                )
