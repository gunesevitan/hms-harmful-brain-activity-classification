import numpy as np
import scipy
import cv2
import torch
import albumentations as A
from albumentations import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2


class ChannelDifference(ImageOnlyTransform):

    def __init__(self, always_apply=True, p=1.0):

        super(ChannelDifference, self).__init__(always_apply=always_apply, p=p)

    def apply(self, inputs, **kwargs):

        """
        Create difference features

        Parameters
        ----------
        inputs: numpy.ndarray of shape (time, 20)
             Inputs array

        Returns
        -------
        inputs: numpy.ndarray of shape (time, 19)
            Inputs array with difference features
        """

        inputs_difference = np.zeros((inputs.shape[0], 19))

        inputs_difference[:, 0] = inputs[:, 0] - inputs[:, 4]
        inputs_difference[:, 1] = inputs[:, 4] - inputs[:, 5]
        inputs_difference[:, 2] = inputs[:, 5] - inputs[:, 6]
        inputs_difference[:, 3] = inputs[:, 6] - inputs[:, 7]

        inputs_difference[:, 4] = inputs[:, 11] - inputs[:, 15]
        inputs_difference[:, 5] = inputs[:, 15] - inputs[:, 16]
        inputs_difference[:, 6] = inputs[:, 16] - inputs[:, 17]
        inputs_difference[:, 7] = inputs[:, 17] - inputs[:, 18]

        inputs_difference[:, 8] = inputs[:, 0] - inputs[:, 1]
        inputs_difference[:, 9] = inputs[:, 1] - inputs[:, 2]
        inputs_difference[:, 10] = inputs[:, 2] - inputs[:, 3]
        inputs_difference[:, 11] = inputs[:, 3] - inputs[:, 7]

        inputs_difference[:, 12] = inputs[:, 11] - inputs[:, 12]
        inputs_difference[:, 13] = inputs[:, 12] - inputs[:, 13]
        inputs_difference[:, 14] = inputs[:, 13] - inputs[:, 14]
        inputs_difference[:, 15] = inputs[:, 14] - inputs[:, 18]

        inputs_difference[:, 16] = inputs[:, 8] - inputs[:, 9]
        inputs_difference[:, 17] = inputs[:, 9] - inputs[:, 10]

        inputs_difference[:, 18] = inputs[:, 19]

        return inputs_difference


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
        inputs: numpy.ndarray of shape (channel, height, width)
             Inputs array

        Returns
        -------
        inputs: numpy.ndarray of shape (channel, height, width)
            Normalized inputs array
        """

        if self.per_channel:
            axis = (0, 1)
        else:
            axis = (0, 1)

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
        Filter inputs

        Parameters
        ----------
        inputs: numpy.ndarray of shape (time, channel)
             Inputs array

        Returns
        -------
        inputs: numpy.ndarray of shape (time, channel)
            Filtered inputs array
        """

        inputs = scipy.signal.sosfilt(self.filter, inputs).astype(np.float32)

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

        # Uniformly sample N amount of indices to drop on time axis between center 10 seconds
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


class RandomGaussianNoise1D(ImageOnlyTransform):

    def __init__(self, mean, std, always_apply=False, p=0.5):

        super(RandomGaussianNoise1D, self).__init__(always_apply=always_apply, p=p)

        self.mean = mean
        self.std = std

    def apply(self, inputs, **kwargs):

        """
        Add random noise to channels

        Parameters
        ----------
        inputs: numpy.ndarray of shape (time, channel)
             Inputs array

        Returns
        -------
        inputs: numpy.ndarray of shape (time, channel)
            Inputs array with random noise added
        """

        for channel_idx in range(inputs.shape[1]):
            mean = inputs[:, channel_idx].mean() / self.mean
            std = np.abs(mean / self.std)
            inputs[:, channel_idx] += np.random.normal(loc=mean, scale=std, size=inputs.shape[0])

        return inputs


class EEGToImage(ImageOnlyTransform):

    def __init__(self, always_apply=True, p=1.0):

        super(EEGToImage, self).__init__(always_apply=always_apply, p=p)

    def apply(self, inputs, **kwargs):

        """
        Convert 1D EEG to 2D vertically stacked image

        Parameters
        ----------
        inputs: numpy.ndarray of shape (time, channel)
             1D EEG inputs array

        Returns
        -------
        image: numpy.ndarray of shape (height, width)
            2D image inputs array
        """

        inputs = inputs.T
        image = []

        for i in range(18):
            for j in range(20):
                image.append(inputs[i, j::20])

        image = np.stack(image, axis=0)

        return image


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
        ChannelDifference(always_apply=True),
        SignalFilter(always_apply=True),
        A.VerticalFlip(p=transform_parameters['vertical_flip_probability']),
        A.HorizontalFlip(p=transform_parameters['horizontal_flip_probability']),
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
        InstanceNormalization1D(per_channel=True, always_apply=True),
        ToTensor1D(always_apply=True)
    ])

    inference_transforms = A.Compose([
        ChannelDifference(always_apply=True),
        SignalFilter(always_apply=True),
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
        ChannelDifference(always_apply=True),
        SignalFilter(always_apply=True),
        A.VerticalFlip(p=transform_parameters['vertical_flip_probability']),
        A.HorizontalFlip(p=transform_parameters['horizontal_flip_probability']),
        InstanceNormalization2D(per_channel=True, always_apply=True),
        EEGToImage(always_apply=True),
        A.Resize(height=384, width=512, interpolation=cv2.INTER_NEAREST),
        ToTensorV2(always_apply=True)
    ])

    inference_transforms = A.Compose([
        ChannelDifference(always_apply=True),
        SignalFilter(always_apply=True),
        InstanceNormalization2D(per_channel=True, always_apply=True),
        EEGToImage(always_apply=True),
        A.Resize(height=384, width=512, interpolation=cv2.INTER_NEAREST),
        ToTensorV2(always_apply=True)
    ])

    eeg_transforms = {'training': training_transforms, 'inference': inference_transforms}
    return eeg_transforms
