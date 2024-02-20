from glob import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):

    def __init__(
            self,
            eeg_paths, targets, target_classes, sample_qualities,
            transforms=None,
            stationary_period_random_subsample_probability=0.,
            mixup_alpha=2, mixup_probability=0., mixup_center_probability=0.
    ):

        self.eeg_paths = eeg_paths
        self.targets = targets
        self.target_classes = target_classes
        self.sample_qualities = sample_qualities
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

        return len(self.eeg_paths)

    def __getitem__(self, idx):

        """
        Get the idxth element in the dataset

        Parameters
        ----------
        idx: int
            Index of the sample (0 <= idx < length of the dataset)

        Returns
        -------
        eeg: torch.Tensor of shape (channel, time) or (channel, time)
            Tensor of EEG

        targets: torch.Tensor of shape (6)
            Tensor of targets

        target_class: torch.Tensor of shape (1)
            Tensor of encoded target
        """

        if np.random.rand() < self.stationary_period_random_subsample_probability:

            current_eeg_path = self.eeg_paths[idx]

            # Extract same stationary period subsample paths from the EEG id
            eeg_id_path = '_'.join(current_eeg_path.split('_')[:2])
            stationary_period_eeg_paths = glob(f'{eeg_id_path}*')
            stationary_period_eeg_paths = [path for path in stationary_period_eeg_paths if path != current_eeg_path]

            if len(stationary_period_eeg_paths) > 1:
                # Randomly select an EEG from the subsample paths
                eeg_path = np.random.choice(stationary_period_eeg_paths)
            else:
                eeg_path = current_eeg_path
        else:
            eeg_path = self.eeg_paths[idx]

        eeg = np.load(eeg_path)
        eeg = pd.DataFrame(eeg).interpolate(method='linear', limit_area='inside')
        eeg = eeg.fillna(0).values

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
            mixup_idx = np.random.randint(0, len(self.eeg_paths))

            mixup_eeg = np.load(self.eeg_paths[mixup_idx])
            mixup_eeg = pd.DataFrame(mixup_eeg).interpolate(method='linear', limit_area='inside')
            mixup_eeg = mixup_eeg.fillna(0).values

            # Sample MixUp lambda from beta distribution
            mixup_lambda = np.clip(np.random.beta(self.mixup_alpha, self.mixup_alpha), a_min=0.4, a_max=0.6)
            if np.random.rand() < self.mixup_center_probability:
                # Apply MixUp to center 10 seconds
                eeg[4000:6000, :] = eeg[4000:6000, :] * mixup_lambda + (1 - mixup_lambda) * mixup_eeg[4000:6000, :]
            else:
                # Apply MixUp to entire EEG
                eeg = eeg * mixup_lambda + (1 - mixup_lambda) * mixup_eeg

            if self.targets is not None:
                mixup_targets = self.targets[mixup_idx]
                targets = targets * mixup_lambda + (1 - mixup_lambda) * mixup_targets
                targets /= targets.sum(dim=-1)

            if self.sample_qualities is not None:
                mixup_sample_quality = self.sample_qualities[mixup_idx]
                sample_quality = torch.ceil((sample_quality + mixup_sample_quality) / 2)

        if self.transforms is not None:
            eeg = self.transforms(image=eeg)['image'].float()
        else:
            eeg = torch.as_tensor(eeg, dtype=torch.float)

        return eeg, targets, target_class, sample_quality


def prepare_data(df, eeg_dataset_path):

    """
    Prepare data for EEG dataset

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with eeg_id, eeg_sub_id and target columns

    eeg_dataset_path: str
        EEG dataset root directory path

    Returns
    -------
    eeg_paths: numpy.ndarray of shape (n_samples)
        Array of EEG paths

    targets: numpy.ndarray of shape (n_samples, 6)
        Array of targets

    target_classes: numpy.ndarray of shape (n_samples)
        Array of target classes

    sample_qualities: numpy.ndarray of shape (n_samples)
        Array of sample qualities
    """

    df['eeg_file_name'] = df['eeg_id'].astype(str) + '_' + df['eeg_sub_id'].astype(str) + '.npy'
    df['eeg_path'] = df['eeg_file_name'].apply(lambda x: str(eeg_dataset_path) + '/eegs/' + x)
    eeg_paths = df['eeg_path'].values

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

    return eeg_paths, targets, target_classes, sample_qualities
