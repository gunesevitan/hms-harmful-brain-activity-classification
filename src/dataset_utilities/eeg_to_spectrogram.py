import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import librosa
import pywt

sys.path.append('..')
import settings


def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x, wavelet='haar', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    ret = pywt.waverec(coeff, wavelet, mode='per')

    return ret


if __name__ == '__main__':

    dataset_directory = settings.DATA / 'eeg_spectrograms'
    dataset_directory.mkdir(parents=True, exist_ok=True)
    (dataset_directory / 'spectrograms').mkdir(parents=True, exist_ok=True)

    eeg_directory = settings.DATA / 'hms-harmful-brain-activity-classification' / 'train_eegs'
    eeg_file_names = os.listdir(eeg_directory)

    df_train = pd.read_csv(settings.DATA / 'hms-harmful-brain-activity-classification' / 'train.csv')

    USE_WAVELET = None

    spectrogram_features = ['LL', 'LP', 'RP', 'RR']
    eeg_feature_groups = [
        ['Fp1', 'F7', 'T3', 'T5', 'O1'],
        ['Fp1', 'F3', 'C3', 'P3', 'O1'],
        ['Fp2', 'F8', 'T4', 'T6', 'O2'],
        ['Fp2', 'F4', 'C4', 'P4', 'O2']
    ]

    i = 0

    for eeg_id, df_train_eeg in tqdm(df_train.groupby('eeg_id'), total=df_train['eeg_id'].nunique()):

        i += 1

        if i < 5570:
            continue

        df_eeg = pd.read_parquet(eeg_directory / f'{eeg_id}.parquet')

        for _, row in df_train_eeg.iterrows():

            eeg_sub_id = row['eeg_sub_id']
            start_idx = int(row['eeg_label_offset_seconds'] * 200)
            end_idx = int((row['eeg_label_offset_seconds'] + 50) * 200)
            df_eeg_subsample = df_eeg.iloc[start_idx:end_idx].reset_index(drop=True).fillna(0)

            spectrogram = np.zeros((128, 256, 4), dtype=np.float32)
            signals = []

            for group_idx in range(4):

                eeg_features = eeg_feature_groups[group_idx]

                for feature_idx in range(4):

                    x = df_eeg_subsample[eeg_features[feature_idx]].values - df_eeg_subsample[eeg_features[feature_idx + 1]].values

                    if USE_WAVELET:
                        x = denoise(x, wavelet=USE_WAVELET)
                    signals.append(x)

                    mel_spec = librosa.feature.melspectrogram(
                        y=x,
                        sr=200,
                        hop_length=len(x) // 256,
                        n_fft=1024,
                        n_mels=128,
                        fmin=0,
                        fmax=20,
                        win_length=128
                    )
                    width = (mel_spec.shape[1] // 32) * 32
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:, :width]

                    # STANDARDIZE TO -1 TO 1
                    mel_spec_db = (mel_spec_db + 40) / 40
                    spectrogram[:, :, group_idx] += mel_spec_db

                # AVERAGE THE 4 MONTAGE DIFFERENCES
                spectrogram[:, :, group_idx] /= 4.0

            spectrogram_file_path = dataset_directory / 'spectrograms' / f'{eeg_id}_{eeg_sub_id}.npy'
            np.save(spectrogram_file_path, spectrogram)


