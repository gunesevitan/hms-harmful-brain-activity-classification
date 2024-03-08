import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import cusignal


sys.path.append('..')
import settings


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

    dataset_directory = settings.DATA / 'eeg_spectrogramsv2'
    dataset_directory.mkdir(parents=True, exist_ok=True)
    (dataset_directory / 'spectrograms').mkdir(parents=True, exist_ok=True)

    eeg_directory = settings.DATA / 'hms-harmful-brain-activity-classification' / 'train_eegs'
    eeg_file_names = os.listdir(eeg_directory)

    df_train = pd.read_csv(settings.DATA / 'hms-harmful-brain-activity-classification' / 'train.csv')

    groups = [
        ['Fp1', 'F7', 'T3', 'T5', 'O1'],
        ['Fp1', 'F3', 'C3', 'P3', 'O1'],
        ['Fp2', 'F8', 'T4', 'T6', 'O2'],
        ['Fp2', 'F4', 'C4', 'P4', 'O2'],
        ['Fz', 'Cz', 'Pz']
    ]

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
                frequencies, times, spectrogram = cusignal.spectrogram(
                    eeg[:, signal_idx],
                    fs=200,
                    nperseg=280,
                    noverlap=260,
                    nfft=None,
                    detrend='constant',
                    scaling='spectrum'
                )
                frequency_mask = (frequencies >= 0.5) & (frequencies <= 20)
                spectrogram = spectrogram[frequency_mask, :]
                spectrograms.append(spectrogram.get().astype(np.float32))

            spectrograms = np.concatenate(spectrograms, axis=0)
            np.save(spectrogram_file_path, spectrograms)
            #import matplotlib.pyplot as plt
            #plt.imshow(np.log(spectrograms))
            #plt.show()
            #exit()
