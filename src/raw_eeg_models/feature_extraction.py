import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import tsfel

sys.path.append('..')
import settings


if __name__ == '__main__':

    eeg_directory = settings.DATA / 'eeg_subsample' / 'eegs'
    eeg_file_names = os.listdir(eeg_directory)

    df_train = pd.read_csv(settings.DATA / 'hms-harmful-brain-activity-classification' / 'train.csv')
    statistical_features = tsfel.get_features_by_domain('statistical')
    temporal_features = tsfel.get_features_by_domain('temporal')
    config = statistical_features | temporal_features

    eeg_columns = [
        'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3',
        'T5', 'O1', 'Fz', 'Cz', 'Pz', 'Fp2',
        'F4', 'C4', 'P4', 'F8', 'T4', 'T6',
        'O2', 'EKG'
    ]

    df_features = []

    for eeg_file_name in tqdm(eeg_file_names):

        eeg_id = int(eeg_file_name.split('_')[0])
        eeg_sub_id = int(eeg_file_name.split('_')[1].split('.')[0])

        eeg_path = eeg_directory / eeg_file_name
        df_eeg = pd.DataFrame(np.load(eeg_path), columns=eeg_columns).iloc[4000:6000]

        features = tsfel.time_series_features_extractor(config, df_eeg, fs=200, verbose=0).astype(np.float32)
        features['eeg_id'] = eeg_id
        features['eeg_sub_id'] = eeg_sub_id
        df_features.append(features)

    df_features = pd.concat(df_features, axis=0).reset_index(drop=True)
    df_features = df_features.sort_values(by=['eeg_id', 'eeg_sub_id'], ascending=True).reset_index(drop=True)
    feature_columns = df_features.columns.tolist()
    df_features = df_features[feature_columns[-2:] + feature_columns[:-2]]
    df_features.to_parquet(settings.DATA / 'eeg_subsample' / 'eeg_features.parquet')
    settings.logger.info(f'eeg_features.parquet is saved to {eeg_directory}')
