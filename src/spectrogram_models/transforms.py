import numpy as np
import scipy
import cv2
import torch
import albumentations as A
from albumentations import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2


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


class CenterTemporalDropout2D(ImageOnlyTransform):

    def __init__(self, center_idx, n_time_steps, drop_value=0, always_apply=False, p=0.5):

        super(CenterTemporalDropout2D, self).__init__(always_apply=always_apply, p=p)

        self.center_idx = center_idx
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
        center_start = self.center_idx[0]
        center_end = self.center_idx[1]
        drop_idx = np.random.choice(np.arange(center_start, center_end), self.n_time_steps, replace=False)
        inputs[drop_idx, :] = self.drop_value

        return inputs


class NonCenterTemporalDropout2D(ImageOnlyTransform):

    def __init__(self, center_idx, n_time_steps, drop_value=0, always_apply=False, p=0.5):

        super(NonCenterTemporalDropout2D, self).__init__(always_apply=always_apply, p=p)

        self.center_idx = center_idx
        self.n_time_steps = n_time_steps
        self.drop_value = drop_value

    def apply(self, image, **kwargs):

        """
        Drop time steps except from center 10 seconds randomly

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
        center_10_seconds_start = self.center_idx[0]
        center_10_seconds_end = self.center_idx[1]
        drop_idx = np.random.choice(
            np.arange(0, center_10_seconds_start).tolist() + np.arange(center_10_seconds_end + 1, image.shape[1]).tolist(),
            self.n_time_steps,
            replace=False
        )
        image[:, drop_idx] = self.drop_value

        return image


class FrequencyDropout(ImageOnlyTransform):

    def __init__(self, n_frequencies, n_consecutive=0, drop_value=0, always_apply=False, p=0.5):

        super(FrequencyDropout, self).__init__(always_apply=always_apply, p=p)

        self.n_frequencies = n_frequencies
        self.n_consecutive = n_consecutive
        self.drop_value = drop_value

    def apply(self, inputs, **kwargs):

        """
        Drop frequencies randomly

        Parameters
        ----------
        inputs: numpy.ndarray of shape (time, channel)
             Inputs array

        Returns
        -------
        inputs: numpy.ndarray of shape (time, channel)
             Inputs array with dropped frequencies
        """

        # Uniformly sample N amount of indices to drop on frequency axis
        drop_idx = np.random.randint(0, inputs.shape[0], self.n_frequencies).tolist()

        if self.n_consecutive > 0:

            consecutive_drop_idx = []

            for idx in drop_idx:

                consecutive_drop_idx.append(idx)
                # Uniformly sample N amount of consecutive indices to drop for each sampled drop index
                consecutive_count = np.random.randint(0, self.n_consecutive + 1)

                if consecutive_count > 0:
                    for consecutive_idx in np.arange(1, consecutive_count + 1):
                        # Add each consecutive index to current drop index and append it indices that'll be dropped
                        consecutive_drop_idx.append(min(inputs.shape[0] - 1, (idx + consecutive_idx)))

            inputs[consecutive_drop_idx, :] = self.drop_value
        else:
            inputs[drop_idx, :] = self.drop_value

        return inputs


def get_spectrogram_2d_transforms(**transform_parameters):

    """
    Get spectrogram 2D transforms for dataset

    Parameters
    ----------
    transform_parameters: dict
        Dictionary of transform parameters

    Returns
    -------
    spectrogram_transforms: dict
        Transforms for training and inference
    """

    training_transforms = A.Compose([
        CenterTemporalDropout2D(
            center_idx=transform_parameters['center_idx'],
            n_time_steps=transform_parameters['center_temporal_dropout_time_steps'],
            drop_value=0,
            p=transform_parameters['center_temporal_dropout_probability']
        ),
        NonCenterTemporalDropout2D(
            center_idx=transform_parameters['center_idx'],
            n_time_steps=transform_parameters['non_center_temporal_dropout_time_steps'],
            drop_value=0,
            p=transform_parameters['non_center_temporal_dropout_probability']
        ),
        FrequencyDropout(
           n_frequencies=transform_parameters['n_frequencies'],
           n_consecutive=transform_parameters['n_consecutive'],
           drop_value=0,
           p=transform_parameters['frequency_dropout_probability']
        ),
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
        A.PadIfNeeded(
            min_height=transform_parameters['pad_min_height'],
            min_width=transform_parameters['pad_min_width'],
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),
        InstanceNormalization2D(per_channel=True, always_apply=True),
        ToTensorV2(always_apply=True)
    ])

    spectrogram_transforms = {'training': training_transforms, 'inference': inference_transforms}
    return spectrogram_transforms
