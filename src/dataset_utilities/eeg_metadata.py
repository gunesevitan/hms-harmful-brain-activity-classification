import os
import sys
from tqdm import tqdm
import pandas as pd

sys.path.append('..')
import settings


if __name__ == '__main__':

    dataset_directory = settings.DATA / 'metadata'
    dataset_directory.mkdir(parents=True, exist_ok=True)

    eeg_directory = settings.DATA / 'hms-harmful-brain-activity-classification' / 'train_eegs'
    eeg_file_names = os.listdir(eeg_directory)

    metadata = []

    for file_name in tqdm(eeg_file_names):

        df_eeg = pd.read_parquet(eeg_directory / file_name)

        metadata.append({
            'eeg_id': int(file_name.split('.')[0]),
            'row_count': df_eeg.shape[0],
            'column_count': df_eeg.shape[1]
        })

    df_metadata = pd.DataFrame(metadata)
    df_metadata.to_csv(dataset_directory / 'eeg_metadata.csv', index=False)
    settings.logger.info(f'eeg_metadata.csv is saved to {dataset_directory}')
