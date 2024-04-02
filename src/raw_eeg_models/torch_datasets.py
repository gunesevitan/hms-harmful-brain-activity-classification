from glob import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):

    def __init__(
            self,
            eeg_paths, targets, target_classes, sample_qualities,
            interpolate_center=True, fill_edge_value=0, clip_bounds=(-10000, 10000),
            transforms=None,
            stationary_period_random_subsample_probability=0.,
            mixup_alpha=2, mixup_probability=0., mixup_center_probability=0.,
            cutmix_probability=0., cutmix_center_probability=0.,
    ):

        self.eeg_paths = eeg_paths
        self.targets = targets
        self.target_classes = target_classes
        self.sample_qualities = sample_qualities

        self.interpolate_center = interpolate_center
        self.fill_edge_value = fill_edge_value
        self.clip_bounds = clip_bounds

        self.transforms = transforms
        self.stationary_period_random_subsample_probability = stationary_period_random_subsample_probability
        self.mixup_alpha = mixup_alpha
        self.mixup_probability = mixup_probability
        self.mixup_center_probability = mixup_center_probability
        self.cutmix_probability = cutmix_probability
        self.cutmix_center_probability = cutmix_center_probability

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
        eeg: torch.Tensor of shape (channel, time) or (time, channel)
            Tensor of EEG

        targets: torch.Tensor of shape (6)
            Tensor of targets

        target_class: torch.Tensor of shape (1)
            Tensor of encoded target
        """

        if np.random.rand() < self.stationary_period_random_subsample_probability:

            current_eeg_path = self.eeg_paths[idx]

            # Extract same stationary period subsample paths from the EEG id
            eeg_id_path = '_'.join(current_eeg_path.split('_')[:3])
            stationary_period_eeg_paths = glob(f'{eeg_id_path}*')
            stationary_period_eeg_paths = [path for path in stationary_period_eeg_paths if path != current_eeg_path]

            if len(stationary_period_eeg_paths) > 1:
                # Randomly select an EEG from the subsample paths
                eeg_path = np.random.choice(stationary_period_eeg_paths)
            else:
                eeg_path = current_eeg_path
        else:
            eeg_path = self.eeg_paths[idx]

        eeg = read_eeg(
            eeg_path=eeg_path,
            interpolate_center=self.interpolate_center,
            fill_edge_value=self.fill_edge_value,
            clip_bounds=self.clip_bounds
        )

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

            mixup_eeg = read_eeg(
                eeg_path=self.eeg_paths[mixup_idx],
                interpolate_center=self.interpolate_center,
                fill_edge_value=self.fill_edge_value,
                clip_bounds=self.clip_bounds
            )

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

        if np.random.rand() < self.cutmix_probability:

            # Randomly select from the subsample paths
            cutmix_idx = np.random.randint(0, len(self.eeg_paths))

            cutmix_eeg = read_eeg(
                eeg_path=self.eeg_paths[cutmix_idx],
                interpolate_center=self.interpolate_center,
                fill_edge_value=self.fill_edge_value,
                clip_bounds=self.clip_bounds
            )

            if np.random.rand() < self.cutmix_center_probability:
                # Apply CutMix to center 10 seconds
                eeg[4000:6000, :] = cutmix_eeg[4000:6000, :]

                # Replace other values as well since the center 10 seconds is altered
                if self.targets is not None:
                    targets = self.targets[cutmix_idx]
                    targets = torch.as_tensor(targets, dtype=torch.float)

                if self.sample_qualities is not None:
                    sample_quality = self.sample_qualities[cutmix_idx]
                    sample_quality = torch.as_tensor(sample_quality, dtype=torch.float)

            else:
                # Apply CutMix to randomly selected non-center 10 seconds
                cutmix_seconds = np.random.choice([0, 1, 3, 4])
                cutmix_start_idx = cutmix_seconds * 2000
                cutmix_end_idx = (cutmix_seconds + 1) * 2000
                eeg[cutmix_start_idx:cutmix_end_idx, :] = cutmix_eeg[cutmix_start_idx:cutmix_end_idx, :]

        if self.transforms is not None:
            eeg = self.transforms(image=eeg)['image'].float()
        else:
            eeg = torch.as_tensor(eeg, dtype=torch.float)

        return eeg, targets, target_class, sample_quality


def read_eeg(eeg_path, interpolate_center=True, fill_edge_value=0, clip_bounds=(-10000, 10000)):

    """
    Read EEG and preprocess it

    Parameters
    ----------
    eeg_path: str or pathlib.Path
        Path of the EEG file

    interpolate_center: bool
        Whether to interpolate missing values located at center or not

    fill_edge_value: int, float or None
        Value to fill missing values located at the edges

    clip_bounds: tuple or None
        Lower and upper bound for clipping values

    Returns
    -------
    eeg: numpy.ndarray of shape (channel, time)
        Array of EEG
    """

    eeg = pd.DataFrame(np.load(eeg_path))

    if interpolate_center:
        eeg = eeg.interpolate(method='linear', limit_area='inside')

    if fill_edge_value is not None:
        eeg = eeg.fillna(0)

    if clip_bounds is not None:
        eeg = eeg.clip(clip_bounds[0], clip_bounds[1])

    eeg = eeg.values

    return eeg


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

    df['eeg_file_name'] = df['eeg_id'].astype(str) + '_' + df['stationary_period'].astype(str) + '_' + df['eeg_sub_id'].astype(str) + '.npy'
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
