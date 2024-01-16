import os
import sys
import json
from tqdm import tqdm
import numpy as np

sys.path.append('..')
import settings


if __name__ == '__main__':

    dataset_directory = settings.DATA / 'spectrogram_subsample' / 'spectrograms'
    dataset_directory.mkdir(parents=True, exist_ok=True)
    spectrogram_file_names = os.listdir(dataset_directory)

    sample_count = 0
    signal_sum = np.zeros(400)
    signal_squared_sum = np.zeros(400)

    for file_name in tqdm(spectrogram_file_names):

        spectrogram_file_path = dataset_directory / file_name
        spectrogram = np.load(spectrogram_file_path)

        if np.isnan(spectrogram).any():
            continue

        sample_count += spectrogram.shape[0]
        signal_sum += np.sum(spectrogram, axis=0)
        signal_squared_sum += np.sum(spectrogram ** 2, axis=0)

    mean = signal_sum / sample_count
    var = (signal_squared_sum / sample_count) - (mean ** 2)
    std = np.sqrt(var)

    dataset_statistics = {
        'mean': mean.tolist(),
        'std': std.tolist()
    }
    with open(settings.DATA / 'spectrogram_subsample' / 'statistics.json', mode='w') as f:
        json.dump(dataset_statistics, f, indent=2)

    settings.logger.info(f'Dataset statistics are calculated with {len(spectrogram_file_names)} spectrograms')
