import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.append('..')
import settings


if __name__ == '__main__':

    dataset_directory = settings.DATA / 'eeg_subsample'
    dataset_directory.mkdir(parents=True, exist_ok=True)

    eeg_directory = settings.DATA / 'hms-harmful-brain-activity-classification' / 'train_eegs'
    eeg_file_names = os.listdir(eeg_directory)

    df_train = pd.read_csv(settings.DATA / 'hms-harmful-brain-activity-classification' / 'train.csv')

    for eeg_id, df_train_eeg in tqdm(df_train.groupby('eeg_id'), total=df_train['eeg_id'].nunique()):

        df_eeg = pd.read_parquet(eeg_directory / f'{eeg_id}.parquet')

        for _, row in df_train_eeg.iterrows():

            eeg_sub_id = row['eeg_sub_id']
            start_idx = int(row['eeg_label_offset_seconds'] * 200)
            end_idx = int((row['eeg_label_offset_seconds'] + 50) * 200)
            df_eeg_subsample = df_eeg.iloc[start_idx:end_idx].reset_index(drop=True)

            eeg_file_path = dataset_directory / f'{eeg_id}_{eeg_sub_id}.npy'
            np.save(eeg_file_path, df_eeg_subsample.values)
