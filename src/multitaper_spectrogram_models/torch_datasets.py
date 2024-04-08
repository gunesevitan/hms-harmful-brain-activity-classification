import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MultitaperSpectrogramDataset(Dataset):

    def __init__(
            self,
            spectrogram_paths, targets, target_classes, sample_qualities, eeg_offset_seconds,
            log_transform, center_idx, transforms=None,
            stationary_period_random_subsample_probability=0.,
            mixup_alpha=2, mixup_probability=0., mixup_center_probability=0.
    ):

        self.spectrogram_paths = spectrogram_paths
        self.targets = targets
        self.target_classes = target_classes
        self.sample_qualities = sample_qualities
        self.eeg_offset_seconds = eeg_offset_seconds

        self.log_transform = log_transform
        self.center_idx = center_idx
        self.transforms = transforms
        self.stationary_period_random_subsample_probability = stationary_period_random_subsample_probability
        self.mixup_alpha = mixup_alpha
        self.mixup_probability = mixup_probability
        self.mixup_center_probability = mixup_center_probability

    def __len__(self):

        """
        Get the length the dataset

        Returns
        -------
        length: int
            Length of the dataset
        """

        return len(self.spectrogram_paths)

    def __getitem__(self, idx):

        """
        Get the idxth element in the dataset

        Parameters
        ----------
        idx: int
            Index of the sample (0 <= idx < length of the dataset)

        Returns
        -------
        spectrogram: torch.Tensor of shape (channel, frequency, time) or (frequency, time)
            Tensor of spectrogram

        targets: torch.Tensor of shape (6)
            Tensor of targets

        target_class: torch.Tensor of shape (1)
            Tensor of encoded target
        """

        spectrogram_path = self.spectrogram_paths[idx]
        eeg_offset_seconds = self.eeg_offset_seconds[idx]
        num_sps = len(eeg_offset_seconds)
        if np.random.rand() < self.stationary_period_random_subsample_probability:
            sp_ind = random.randint(0, num_sps - 1)
        else:
            sp_ind = num_sps // 2

        eeg_sp = int((eeg_offset_seconds[sp_ind] - eeg_offset_seconds[0]) * 2)
        spectrogram = read_spectrogram(
            spectrogram_path=spectrogram_path,
            log_transform=self.log_transform
        )
        spectrogram = spectrogram[:, :, eeg_sp:eeg_sp + 93]

        if self.targets is not None:
            targets = self.targets[idx]
            targets = torch.as_tensor(targets, dtype=torch.float)
        else:
            targets = None

        if self.target_classes is not None:
            target_class = self.target_classes[idx]
            target_class = torch.as_tensor(target_class, dtype=torch.float)
        else:
            target_class = None

        if self.sample_qualities is not None:
            sample_quality = self.sample_qualities[idx]
            sample_quality = torch.as_tensor(sample_quality, dtype=torch.float)
        else:
            sample_quality = None

        if np.random.rand() < self.mixup_probability:

            # Randomly select from the subsample paths
            mixup_idx = np.random.randint(0, len(self.spectrogram_paths))
            mixup_spectrogram = read_spectrogram(
                spectrogram_path=self.spectrogram_paths[mixup_idx],
                log_transform=self.log_transform
            )

            # Sample MixUp lambda from beta distribution
            mixup_lambda = np.clip(np.random.beta(self.mixup_alpha, self.mixup_alpha), a_min=0.4, a_max=0.6)
            if np.random.rand() < self.mixup_center_probability:
                # Apply MixUp to center 10 seconds
                spectrogram[:, self.center_idx[0]:self.center_idx[1]] = spectrogram[:, self.center_idx[0]:self.center_idx[1]] * mixup_lambda + (1 - mixup_lambda) * mixup_spectrogram[:, self.center_idx[0]:self.center_idx[1]]
            else:
                # Apply MixUp to entire spectrogram
                spectrogram = spectrogram * mixup_lambda + (1 - mixup_lambda) * mixup_spectrogram

            if self.targets is not None:
                mixup_targets = self.targets[mixup_idx]
                targets = targets * mixup_lambda + (1 - mixup_lambda) * mixup_targets
                targets /= targets.sum(dim=-1)

            if self.sample_qualities is not None:
                mixup_sample_quality = self.sample_qualities[mixup_idx]
                sample_quality = torch.ceil((sample_quality + mixup_sample_quality) / 2)

        if self.transforms is not None:
            spectrogram = self.transforms(image=spectrogram)['image'].float()
        else:
            spectrogram = torch.as_tensor(spectrogram, dtype=torch.float)

        return spectrogram, targets, target_class, sample_quality


def read_spectrogram(spectrogram_path, log_transform=False):

    """
    Read spectrogram and preprocess it

    Parameters
    ----------
    spectrogram_path: str or pathlib.Path
        Path of the spectrogram file

    log_transform: bool
        Whether to apply log transform or not

    Returns
    -------
    spectrogram: numpy.ndarray of shape (frequency, time)
        Array of spectrogram
    """

    spectrogram = np.load(spectrogram_path)['arr']

    if log_transform:
        spectrogram = np.log1p(spectrogram)

    return spectrogram


def prepare_data(df, spectrogram_dataset_path):

    """
    Prepare data for spectrogram dataset

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with spectrogram_id, spectrogram_sub_id and target columns

    spectrogram_dataset_path: str
        Spectrogram dataset root directory path

    Returns
    -------
    spectrogram_paths: numpy.ndarray of shape (n_samples)
        Array of spectrogram paths

    targets: numpy.ndarray of shape (n_samples, 6)
        Array of targets

    target_classes: numpy.ndarray of shape (n_samples)
        Array of target classes

    sample_qualities: numpy.ndarray of shape (n_samples)
        Array of sample qualities

    eeg_offset_seconds: numpy.ndarray of shape (n_samples)
        Array of sps
    """

    df['spectrogram_file_name'] = df['eeg_id'].index.astype('str') + '_multitaper_spec.npz'
    df['spectrogram_path'] = df['spectrogram_file_name'].apply(lambda x: str(spectrogram_dataset_path) + '/spectrograms/' + x)
    spectrogram_paths = df['spectrogram_path'].values

    target_columns = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    targets = df[target_columns].values
    target_classes = pd.get_dummies(df['expert_consensus'].map({
        'Seizure': 0,
        'LPD': 1,
        'GPD': 2,
        'LRDA': 3,
        'GRDA': 4,
        'Other': 5
    })).astype(np.uint8).values
    sample_qualities = df['sample_quality'].values
    eeg_offset_seconds = df['eeg_label_offset_seconds'].values

    return spectrogram_paths, targets, target_classes, sample_qualities, eeg_offset_seconds
