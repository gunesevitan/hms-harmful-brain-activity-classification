import numpy as np
import albumentations as A
import torch
from albumentations import ImageOnlyTransform


class InstanceNormalization1D(ImageOnlyTransform):

    def __init__(self, log_scale, channel_wise, epsilon=1e-15, always_apply=True, p=1.0):

        super(InstanceNormalization1D, self).__init__(always_apply=always_apply, p=p)

        self.log_scale = log_scale
        self.channel_wise = channel_wise
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

        if self.log_scale:
            inputs = np.log1p(inputs)

        if self.channel_wise:
            axis = 0
        else:
            axis = (0, 1)

        mean = np.mean(inputs, axis=axis)
        std = np.std(inputs, axis=axis)
        image = (inputs - mean) / (std + self.epsilon)

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


def get_eeg_transforms(**transform_parameters):

    """
    Get EEGs transforms for dataset

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
        InstanceNormalization1D(log_scale=False, channel_wise=True, always_apply=True),
        ToTensor1D(always_apply=True)
    ])

    inference_transforms = A.Compose([
        InstanceNormalization1D(log_scale=False, channel_wise=True, always_apply=True),
        ToTensor1D(always_apply=True)
    ])

    eeg_transforms = {'training': training_transforms, 'inference': inference_transforms}
    return eeg_transforms
