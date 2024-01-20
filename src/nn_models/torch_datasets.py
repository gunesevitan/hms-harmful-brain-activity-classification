import numpy as np
import torch
from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):

    def __init__(self, data_paths, targets, transforms=None):

        self.data_paths = data_paths
        self.targets = targets
        self.transforms = transforms

    def __len__(self):

        """
        Get the length the dataset

        Returns
        -------
        length: int
            Length of the dataset
        """

        return len(self.data_paths)

    def __getitem__(self, idx):

        """
        Get the idxth element in the dataset

        Parameters
        ----------
        idx: int
            Index of the sample (0 <= idx < length of the dataset)

        Returns
        -------
        spectrogram: torch.Tensor of shape (frequency, time) or (channel, frequency, time)
            Array of spectrogram

        target: torch.Tensor of shape (6)
            Array of targets
        """

        spectrogram = np.load(self.data_paths[idx]).T
        spectrogram = np.log1p(spectrogram)

        if self.transforms is not None:
            spectrogram = self.transforms(image=spectrogram)['image'].float()
        else:
            spectrogram = torch.as_tensor(spectrogram, dtype=torch.float)

        if self.targets is not None:
            target = self.targets[idx]
            target = torch.as_tensor(target, dtype=torch.float)
        else:
            target = None

        return spectrogram, target


def prepare_classification_data(df, eeg_dataset_path, spectrogram_dataset_path):

    """
    Prepare data for classification dataset

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with eeg_id, eeg_sub_id, spectrogram_id, spectrogram_sub_id and target columns

    eeg_dataset_path: str
        EEG dataset root directory path

    spectrogram_dataset_path: str
        Spectrogram dataset root directory path

    Returns
    -------
    eeg_paths: numpy.ndarray of shape (n_samples)
        Array of EEG paths

    spectrogram_paths: numpy.ndarray of shape (n_samples)
        Array of spectrogram paths

    targets: numpy.ndarray of shape (n_samples, 6)
        Array of targets

    target_classes: numpy.ndarray of shape (n_samples)
        Array of target classes
    """

    df['eeg_file_name'] = df['eeg_id'].astype(str) + '_' + df['eeg_sub_id'].astype(str) + '.npy'
    df['eeg_path'] = df['eeg_file_name'].apply(lambda x: str(eeg_dataset_path) + '/eegs/' + x)
    eeg_paths = df['eeg_path'].values

    df['spectrogram_file_name'] = df['spectrogram_id'].astype(str) + '_' + df['spectrogram_sub_id'].astype(str) + '.npy'
    df['spectrogram_path'] = df['spectrogram_file_name'].apply(lambda x: str(spectrogram_dataset_path) + '/spectrograms/' + x)
    spectrogram_paths = df['spectrogram_path'].values

    target_columns = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    targets = df[target_columns].values
    target_classes = df['expert_consensus'].map({
        'Seizure': 0,
        'LPD': 1,
        'GPD': 2,
        'LRDA': 3,
        'GRDA': 4,
        'Other': 5
    }).values

    return eeg_paths, spectrogram_paths, targets, target_classes
