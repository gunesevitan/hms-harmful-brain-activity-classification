import numpy as np
import scipy
import cv2
import torch
import albumentations as A
from albumentations import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2


class ChannelDifference1D(ImageOnlyTransform):

    def __init__(self, ekg, always_apply=True, p=1.0):

        super(ChannelDifference1D, self).__init__(always_apply=always_apply, p=p)

        self.ekg = ekg

    def apply(self, inputs, **kwargs):

        """
        Create bipolar channel difference features

        Parameters
        ----------
        inputs: numpy.ndarray of shape (time, 20)
             Inputs array

        Returns
        -------
        inputs: numpy.ndarray of shape (time, 18 or 19)
            Inputs array with difference features
        """

        n_channels = 19 if self.ekg else 18
        inputs_difference = np.zeros((inputs.shape[0], n_channels))

        # Left outside temporal chain
        inputs_difference[:, 0] = inputs[:, 0] - inputs[:, 4]
        inputs_difference[:, 1] = inputs[:, 4] - inputs[:, 5]
        inputs_difference[:, 2] = inputs[:, 5] - inputs[:, 6]
        inputs_difference[:, 3] = inputs[:, 6] - inputs[:, 7]

        # Right outside temporal chain
        inputs_difference[:, 4] = inputs[:, 11] - inputs[:, 15]
        inputs_difference[:, 5] = inputs[:, 15] - inputs[:, 16]
        inputs_difference[:, 6] = inputs[:, 16] - inputs[:, 17]
        inputs_difference[:, 7] = inputs[:, 17] - inputs[:, 18]

        # Left inside parasagittal chain
        inputs_difference[:, 8] = inputs[:, 0] - inputs[:, 1]
        inputs_difference[:, 9] = inputs[:, 1] - inputs[:, 2]
        inputs_difference[:, 10] = inputs[:, 2] - inputs[:, 3]
        inputs_difference[:, 11] = inputs[:, 3] - inputs[:, 7]

        # Right inside parasagittal chain
        inputs_difference[:, 12] = inputs[:, 11] - inputs[:, 12]
        inputs_difference[:, 13] = inputs[:, 12] - inputs[:, 13]
        inputs_difference[:, 14] = inputs[:, 13] - inputs[:, 14]
        inputs_difference[:, 15] = inputs[:, 14] - inputs[:, 18]

        # Center chain
        inputs_difference[:, 16] = inputs[:, 8] - inputs[:, 9]
        inputs_difference[:, 17] = inputs[:, 9] - inputs[:, 10]

        if self.ekg:
            inputs_difference[:, 18] = inputs[:, 19]

        return inputs_difference


class ChannelGroupPermute1D(ImageOnlyTransform):

    def __init__(self, ekg, always_apply=False, p=0.5):

        super(ChannelGroupPermute1D, self).__init__(always_apply=always_apply, p=p)

        self.ekg = ekg

    def apply(self, inputs, **kwargs):

        """
        Permute 4 main channel groups

        Parameters
        ----------
        inputs: numpy.ndarray of shape (time, 18 or 19)
            Inputs array with difference features

        Returns
        -------
        inputs_permuted: numpy.ndarray of shape (time, 18 or 19)
            Inputs array with difference features permuted
        """

        groups = np.array([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15]
        ])
        inputs_permuted = np.zeros_like(inputs)
        for group_idx, permuted_group_idx in enumerate(np.random.permutation(np.arange(len(groups)))):
            # Permute channel groups of LL, RL, LP and RP chains
            group = groups[group_idx]
            permuted_group = groups[permuted_group_idx]
            inputs_permuted[group] = inputs[permuted_group]

        # Center chain and EKG are not permuted
        inputs_permuted[:, [16, 17]] = inputs[:, [16, 17]]
        if self.ekg:
            inputs_permuted[:, 18] = inputs[:, 18]

        return inputs_permuted


class InstanceNormalization1D(ImageOnlyTransform):

    def __init__(self, per_channel, epsilon=1e-15, always_apply=True, p=1.0):

        super(InstanceNormalization1D, self).__init__(always_apply=always_apply, p=p)

        self.per_channel = per_channel
        self.epsilon = epsilon

    def apply(self, inputs, **kwargs):

        """
        Normalize inputs by mean and standard deviation of itself

        Parameters
        ----------
        inputs: numpy.ndarray of shape (time, channel)
             Inputs array

        Returns
        -------
        inputs: numpy.ndarray of shape (time, channel)
            Normalized inputs array
        """

        if self.per_channel:
            axis = 0
        else:
            axis = (0, 1)

        mean = np.mean(inputs, axis=axis)
        std = np.std(inputs, axis=axis)
        inputs = (inputs - mean) / (std + self.epsilon)

        return inputs


class InstanceNormalization2D(ImageOnlyTransform):

    def __init__(self, per_channel, epsilon=1e-15, always_apply=True, p=1.0):

        super(InstanceNormalization2D, self).__init__(always_apply=always_apply, p=p)

        self.per_channel = per_channel
        self.epsilon = epsilon

    def apply(self, inputs, **kwargs):

        """
        Normalize inputs by mean and standard deviation of itself

        Parameters
        ----------
        inputs: numpy.ndarray of shape (height, width, channel)
             Inputs array

        Returns
        -------
        inputs: numpy.ndarray of shape (height, width, channel)
            Normalized inputs array
        """

        if self.per_channel:
            axis = (0, 1)
        else:
            axis = (0, 1, 2)

        mean = np.mean(inputs, axis=axis)
        std = np.std(inputs, axis=axis)
        inputs = (inputs - mean) / (std + self.epsilon)

        return inputs


class InstanceNormalization3D(ImageOnlyTransform):

    def __init__(self, per_channel, epsilon=1e-15, always_apply=True, p=1.0):

        super(InstanceNormalization3D, self).__init__(always_apply=always_apply, p=p)

        self.per_channel = per_channel
        self.epsilon = epsilon

    def apply(self, inputs, **kwargs):

        """
        Normalize inputs by mean and standard deviation of itself

        Parameters
        ----------
        inputs: numpy.ndarray of shape (depth, height, width, channel)
             Inputs array

        Returns
        -------
        inputs: numpy.ndarray of shape (depth, height, width, channel)
            Normalized inputs array
        """

        if self.per_channel:
            axis = (0, 1, 2)
        else:
            axis = (0, 1, 2, 3)

        mean = np.mean(inputs, axis=axis)
        std = np.std(inputs, axis=axis)
        inputs = (inputs - mean) / (std + self.epsilon)

        return inputs


class SignalFilter(ImageOnlyTransform):

    def __init__(self, always_apply=True, p=1.0):

        super(SignalFilter, self).__init__(always_apply=always_apply, p=p)

        self.filter = scipy.signal.butter(3, [0.001, 20], btype='bandpass', fs=200, output='sos')

    def apply(self, inputs, **kwargs):

        """
        Filter inputs signals

        Parameters
        ----------
        inputs: numpy.ndarray of shape (time, channel)
             Inputs array

        Returns
        -------
        inputs: numpy.ndarray of shape (time, channel)
            Filtered inputs array
        """

        inputs = scipy.signal.sosfilt(self.filter, inputs, axis=0).astype(np.float32)
        return inputs


class CenterTemporalDropout1D(ImageOnlyTransform):

    def __init__(self, n_time_steps, drop_value=0, always_apply=False, p=0.5):

        super(CenterTemporalDropout1D, self).__init__(always_apply=always_apply, p=p)

        self.n_time_steps = n_time_steps
        self.drop_value = drop_value

    def apply(self, inputs, **kwargs):

        """
        Drop time steps from center 10 seconds randomly

        Parameters
        ----------
        inputs: numpy.ndarray of shape (time, channel)
             Inputs array

        Returns
        -------
        inputs: numpy.ndarray of shape (time, channel)
            Inputs array with dropped time steps from center 10 seconds
        """

        # Uniformly sample N amount of indices to drop on time axis at center 10 seconds
        center_start = 4000
        center_end = 6000
        drop_idx = np.random.choice(np.arange(center_start, center_end + 1), self.n_time_steps, replace=False)
        inputs[drop_idx, :] = self.drop_value

        return inputs


class NonCenterTemporalDropout1D(ImageOnlyTransform):

    def __init__(self, n_time_steps, drop_value=0, always_apply=False, p=0.5):

        super(NonCenterTemporalDropout1D, self).__init__(always_apply=always_apply, p=p)

        self.n_time_steps = n_time_steps
        self.drop_value = drop_value

    def apply(self, inputs, **kwargs):

        """
        Drop time steps from other than center 10 seconds randomly

        Parameters
        ----------
        inputs: numpy.ndarray of shape (time, channel)
             Inputs array

        Returns
        -------
        inputs: numpy.ndarray of shape (time, channel)
            Inputs array with dropped time steps except from center 10 seconds
        """

        # Uniformly sample N amount of indices to drop on time axis except from center 10 seconds
        center_10_seconds_start = 4000
        center_10_seconds_end = 6000
        drop_idx = np.random.choice(
            np.arange(0, center_10_seconds_start).tolist() + np.arange(center_10_seconds_end + 1, inputs.shape[1]).tolist(),
            self.n_time_steps,
            replace=False
        )
        inputs[drop_idx, :] = self.drop_value

        return inputs


class EEGTo2D(ImageOnlyTransform):

    def __init__(self, always_apply=True, p=1.0):

        super(EEGTo2D, self).__init__(always_apply=always_apply, p=p)

    def apply(self, inputs, **kwargs):

        """
        Convert 1D EEG to vertically stacked 2D

        Parameters
        ----------
        inputs: numpy.ndarray of shape (time, channel)
             1D inputs array

        Returns
        -------
        image: numpy.ndarray of shape (height, width)
            2D inputs array
        """

        inputs = inputs.T

        image = []

        for i in range(18):
            for j in range(20):
                image.append(inputs[i, j::20])

        image = np.stack(image, axis=0)

        return image


class EEGTo3D(ImageOnlyTransform):

    def __init__(self, always_apply=True, p=1.0):

        super(EEGTo3D, self).__init__(always_apply=always_apply, p=p)

    def apply(self, inputs, **kwargs):

        """
        Convert 1D EEG to 3D vertically and depth stacked

        Parameters
        ----------
        inputs: numpy.ndarray of shape (time, channel)
             1D inputs array

        Returns
        -------
        image: numpy.ndarray of shape (depth, height, width)
            3D inputs array
        """

        # Horizontally duplicate center chain for symmetrical slices
        inputs = np.hstack((inputs, inputs[:, [16, 17]]))
        inputs = inputs.T

        image = []

        for i in range(20):
            for j in range(20):
                image.append(inputs[i, j::20])

        image = np.stack(image, axis=0)
        image = np.stack([
            image[0:80],
            image[80:160],
            image[160:240],
            image[240:320],
            image[320:400],
        ], axis=0)

        return image


class Flip3D(ImageOnlyTransform):

    def __init__(self, flip_dimension, always_apply=False, p=0.5):

        super(Flip3D, self).__init__(always_apply=always_apply, p=p)

        self.flip_dimension = flip_dimension

    def apply(self, inputs, **kwargs):

        """
        Flip 3D EEG on given axis

        Parameters
        ----------
        inputs: numpy.ndarray of shape (depth, height, width)
            3D inputs array

        Returns
        -------
        inputs: numpy.ndarray of shape (depth, height, width)
            Flipped 3D inputs array
        """

        inputs = np.flip(inputs, axis=self.flip_dimension)

        return  inputs


class ToTensor1D(ImageOnlyTransform):

    def __init__(self, always_apply=True, p=1.0):

        super(ToTensor1D, self).__init__(always_apply=always_apply, p=p)

    def apply(self, inputs, **kwargs):

        """
        Convert inputs array to a torch tensor and transpose it

        Parameters
        ----------
        inputs: numpy.ndarray of shape (time, channel)
             Inputs array

        Returns
        -------
        inputs: torch.Tensor of shape (channel, time)
            Transposed inputs tensor
        """

        inputs = torch.as_tensor(inputs.T, dtype=torch.float32)

        return inputs


class ToTensor3D(ImageOnlyTransform):

    def __init__(self, always_apply=True, p=1.0):

        super(ToTensor3D, self).__init__(always_apply=always_apply, p=p)

    def apply(self, inputs, **kwargs):

        """
        Convert inputs array to a torch tensor and add channel dimension

        Parameters
        ----------
        inputs: numpy.ndarray of shape (depth, height, width)
             Inputs array

        Returns
        -------
        inputs: torch.Tensor of shape (channel, depth, height, width)
            Inputs tensor
        """

        inputs = torch.as_tensor(inputs, dtype=torch.float32)

        return inputs


def get_raw_eeg_1d_transforms(**transform_parameters):

    """
    Get raw EEG 1D transforms for dataset

    Parameters
    ----------
    transform_parameters: dict
        Dictionary of transform parameters

    Returns
    -------
    eeg_transforms: dict
        Transforms for training and inference
    """

    training_transforms = A.Compose([
        ChannelDifference1D(ekg=True, always_apply=True),
        ChannelGroupPermute1D(ekg=False, p=transform_parameters['channel_group_permute_probability']),
        CenterTemporalDropout1D(
            n_time_steps=transform_parameters['center_temporal_dropout_time_steps'],
            drop_value=0,
            p=transform_parameters['center_temporal_dropout_probability']
        ),
        NonCenterTemporalDropout1D(
            n_time_steps=transform_parameters['non_center_temporal_dropout_time_steps'],
            drop_value=0,
            p=transform_parameters['non_center_temporal_dropout_probability']
        ),
        A.VerticalFlip(p=transform_parameters['vertical_flip_probability']),
        A.HorizontalFlip(p=transform_parameters['horizontal_flip_probability']),
        InstanceNormalization1D(per_channel=True, always_apply=True),
        ToTensor1D(always_apply=True)
    ])

    inference_transforms = A.Compose([
        ChannelDifference1D(ekg=True, always_apply=True),
        InstanceNormalization1D(per_channel=True, always_apply=True),
        ToTensor1D(always_apply=True)
    ])

    eeg_transforms = {'training': training_transforms, 'inference': inference_transforms}
    return eeg_transforms


def get_raw_eeg_2d_transforms(**transform_parameters):

    """
    Get raw EEG 2D transforms for dataset

    Parameters
    ----------
    transform_parameters: dict
        Dictionary of transform parameters

    Returns
    -------
    eeg_transforms: dict
        Transforms for training and inference
    """

    training_transforms = A.Compose([
        ChannelDifference1D(ekg=False, always_apply=True),
        ChannelGroupPermute1D(ekg=False, p=transform_parameters['channel_group_permute_probability']),
        CenterTemporalDropout1D(
            n_time_steps=transform_parameters['center_temporal_dropout_time_steps'],
            drop_value=0,
            p=transform_parameters['center_temporal_dropout_probability']
        ),
        NonCenterTemporalDropout1D(
            n_time_steps=transform_parameters['non_center_temporal_dropout_time_steps'],
            drop_value=0,
            p=transform_parameters['non_center_temporal_dropout_probability']
        ),
        EEGTo2D(always_apply=True),
        A.CoarseDropout(
            max_holes=transform_parameters['coarse_dropout_max_holes'],
            min_holes=transform_parameters['coarse_dropout_min_holes'],
            max_height=transform_parameters['coarse_dropout_max_height'],
            max_width=transform_parameters['coarse_dropout_max_width'],
            min_height=transform_parameters['coarse_dropout_min_height'],
            min_width=transform_parameters['coarse_dropout_min_width'],
            fill_value=0,
            p=transform_parameters['coarse_dropout_probability']
        ),
        A.VerticalFlip(p=transform_parameters['vertical_flip_probability']),
        A.HorizontalFlip(p=transform_parameters['horizontal_flip_probability']),
        A.PadIfNeeded(
            min_height=transform_parameters['pad_min_height'],
            min_width=transform_parameters['pad_min_width'],
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),
        InstanceNormalization2D(per_channel=True, always_apply=True),
        ToTensorV2(always_apply=True)
    ])

    inference_transforms = A.Compose([
        ChannelDifference1D(ekg=False, always_apply=True),
        EEGTo2D(always_apply=True),
        A.PadIfNeeded(
            min_height=transform_parameters['pad_min_height'],
            min_width=transform_parameters['pad_min_width'],
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),
        InstanceNormalization2D(per_channel=True, always_apply=True),
        ToTensorV2(always_apply=True)
    ])

    eeg_transforms = {'training': training_transforms, 'inference': inference_transforms}
    return eeg_transforms


def get_raw_eeg_3d_transforms(**transform_parameters):

    """
    Get raw EEG 3D transforms for dataset

    Parameters
    ----------
    transform_parameters: dict
        Dictionary of transform parameters

    Returns
    -------
    eeg_transforms: dict
        Transforms for training and inference
    """

    training_transforms = A.Compose([
        ChannelDifference1D(ekg=False, always_apply=True),
        ChannelGroupPermute1D(ekg=False, p=transform_parameters['channel_group_permute_probability']),
        CenterTemporalDropout1D(
            n_time_steps=transform_parameters['center_temporal_dropout_time_steps'],
            drop_value=0,
            p=transform_parameters['center_temporal_dropout_probability']
        ),
        NonCenterTemporalDropout1D(
            n_time_steps=transform_parameters['non_center_temporal_dropout_time_steps'],
            drop_value=0,
            p=transform_parameters['non_center_temporal_dropout_probability']
        ),
        EEGTo3D(always_apply=True),
        Flip3D(flip_dimension=0, p=transform_parameters['horizontal_flip_probability']),
        Flip3D(flip_dimension=1, p=transform_parameters['vertical_flip_probability']),
        Flip3D(flip_dimension=2, p=transform_parameters['depth_flip_probability']),
        InstanceNormalization3D(per_channel=True, always_apply=True),
        ToTensor3D(always_apply=True)
    ])

    inference_transforms = A.Compose([
        ChannelDifference1D(ekg=False, always_apply=True),
        EEGTo3D(always_apply=True),
        InstanceNormalization3D(per_channel=True, always_apply=True),
        ToTensor3D(always_apply=True)
    ])

    eeg_transforms = {'training': training_transforms, 'inference': inference_transforms}
    return eeg_transforms
