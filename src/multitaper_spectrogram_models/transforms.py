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
    

class FlipAlongAxis(ImageOnlyTransform):

    def __init__(self, axis, always_apply=False, p=0.5):

        super(FlipAlongAxis, self).__init__(always_apply=always_apply, p=p)

        self.axis = axis

    def apply(self, inputs, **kwargs):

        """
        Flip inputs along the given axis

        Parameters
        ----------
        inputs: numpy.ndarray of shape (channel, height, width)
             Inputs array

        Returns
        -------
        inputs: numpy.ndarray of shape (channel, height, width)
            Inputs array flipped along the given axis
        """

        inputs = np.flip(inputs, axis=self.axis).copy()
        return inputs
    

class FlipHemispheres(ImageOnlyTransform):

    def __init__(self, always_apply=False, p=0.5):

        super(FlipHemispheres, self).__init__(always_apply=always_apply, p=p)

    def apply(self, inputs, **kwargs):

        """
        Flip left and right hemispheres of EEG signals

        Parameters
        ----------
        inputs: numpy.ndarray of shape (channel, height, width)
             Inputs array

        Returns
        -------
        inputs: numpy.ndarray of shape (channel, height, width)
            Inputs array with flipped hemispheres
        """

        permute_idx = [4,5,6,7, 0,1,2,3, 12,13,14,15, 8,9,10,11, 16,17]
        return inputs[permute_idx]
    

class EEG3Dto2D(ImageOnlyTransform):

    def __init__(self, always_apply=True, p=1.0):

        super(EEG3Dto2D, self).__init__(always_apply=always_apply, p=p)
        self.eeg_spec_idxs = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 18)]

    def apply(self, inputs, **kwargs):

        """
        Create 2D EEG image from 3D inputs

        Parameters
        ----------
        inputs: numpy.ndarray of shape (channel, height, width)
             Inputs array

        Returns
        -------
        inputs: numpy.ndarray of shape (new_height, new_width)
            2D EEG image
        """

        image = []
        for eeg_spec_idx in self.eeg_spec_idxs:
            inputs_temp = inputs[eeg_spec_idx[0]:eeg_spec_idx[1]]

            if inputs_temp.shape[0] == 2:
                inputs_temp = np.concatenate([inputs_temp, np.zeros_like(inputs_temp)], axis=0)

            image.append(np.concatenate([inputs_temp[0], inputs_temp[1], inputs_temp[2], inputs_temp[3]], axis=1))
        
        return np.concatenate(image, axis=0)


def get_multitaper_spectrogram_2d_transforms(**transform_parameters):

    """
    Get multitaper spectrogram 2D transforms for dataset

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
        FlipAlongAxis(axis=2, p=transform_parameters['horizontal_flip_probability']),
        FlipHemispheres(p=transform_parameters['hemisphere_flip_probability']),
        EEG3Dto2D(always_apply=True),
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
        EEG3Dto2D(always_apply=True),
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
