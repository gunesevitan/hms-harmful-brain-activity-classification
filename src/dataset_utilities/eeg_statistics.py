import os
import sys
import json
from tqdm import tqdm
import numpy as np

sys.path.append('..')
import settings


if __name__ == '__main__':

    dataset_directory = settings.DATA / 'eeg_subsample' / 'eegs'
    dataset_directory.mkdir(parents=True, exist_ok=True)
    eeg_file_names = os.listdir(dataset_directory)

    sample_count = 0
    signal_sum = np.zeros(20)
    signal_squared_sum = np.zeros(20)

    for file_name in tqdm(eeg_file_names):

        eeg_file_path = dataset_directory / file_name
        eeg = np.load(eeg_file_path)

        if np.isnan(eeg).any():
            continue

        sample_count += eeg.shape[0]
        signal_sum += np.sum(eeg, axis=0)
        signal_squared_sum += np.sum(eeg ** 2, axis=0)

    mean = signal_sum / sample_count
    var = (signal_squared_sum / sample_count) - (mean ** 2)
    std = np.sqrt(var)

    dataset_statistics = {
        'mean': mean.tolist(),
        'std': std.tolist()
    }
    with open(settings.DATA / 'eeg_subsample' / 'statistics.json', mode='w') as f:
        json.dump(dataset_statistics, f, indent=2)

    settings.logger.info(f'Dataset statistics are calculated with {len(eeg_file_names)} eegs')
