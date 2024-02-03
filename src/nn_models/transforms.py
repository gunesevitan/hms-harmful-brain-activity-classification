import numpy as np
import cv2
import albumentations as A
from albumentations import ImageOnlyTransform
from albumentations.pytorch.transforms import ToTensorV2


class AddChannelDimension(ImageOnlyTransform):

    def __init__(self, always_apply=True, p=1.0):

        super(AddChannelDimension, self).__init__(always_apply=always_apply, p=p)

    def apply(self, image, **kwargs):

        """
        Add channel dimension at the end

        Parameters
        ----------
        image: numpy.ndarray of shape (height, width)
             Image array

        Returns
        -------
        image: numpy.ndarray of shape (height, width, channel)
            Channel added image array
        """

        return image[..., None]


class InstanceNormalization(ImageOnlyTransform):

    def __init__(self, channel_wise, epsilon=1e-6, always_apply=True, p=1.0):

        super(InstanceNormalization, self).__init__(always_apply=always_apply, p=p)

        self.channel_wise = channel_wise
        self.epsilon = epsilon

    def apply(self, image, **kwargs):

        """
        Normalize image by mean and standard deviation of itself

        Parameters
        ----------
        image: numpy.ndarray of shape (height, width, channel)
             Image array

        Returns
        -------
        image: numpy.ndarray of shape (height, width, channel)
            Normalized image array
        """

        if self.channel_wise:
            axis = (0, 1)
        else:
            axis = (0, 1, 2)

        mean = np.mean(image, axis=axis)
        std = np.std(image, axis=axis)
        image = (image - mean) / (std + self.epsilon)

        return image


class GroupFlip(ImageOnlyTransform):

    def __init__(self, flip_dimensions, n_groups, always_apply=False, p=0.5):

        super(GroupFlip, self).__init__(always_apply=always_apply, p=p)

        self.flip_dimensions = flip_dimensions
        self.n_groups = n_groups

    def apply(self, image, **kwargs):

        """
        Flip groups of features in specified axes

        Parameters
        ----------
        image: numpy.ndarray of shape (height, width, channel)
             Image array

        Returns
        -------
        image: numpy.ndarray of shape (height, width, channel)
            Flipped image array
        """

        groups = np.random.choice([0, 1, 2, 3], self.n_groups, replace=False)
        group_height = image.shape[0] // 4

        for group in groups:
            group_idx_start = group_height * group
            group_idx_end = group_height * (group + 1)
            image[group_idx_start:group_idx_end] = np.flip(image[group_idx_start:group_idx_end], axis=self.flip_dimensions)

        return image


class GroupPermute(ImageOnlyTransform):

    def __init__(self, always_apply=False, p=0.5):

        super(GroupPermute, self).__init__(always_apply=always_apply, p=p)

    def apply(self, image, **kwargs):

        """
        Permute groups of features

        Parameters
        ----------
        image: numpy.ndarray of shape (height, width, channel)
             Image array

        Returns
        -------
        image_permuted: numpy.ndarray of shape (height, width, channel)
            Permuted image array
        """

        permuted_groups = np.random.permutation([0, 1, 2, 3])
        group_height = image.shape[0] // 4

        image_permuted = np.zeros_like(image)

        for group, permuted_group in enumerate(permuted_groups):
            group_idx_start = group_height * group
            group_idx_end = group_height * (group + 1)
            permuted_group_idx_start = group_height * permuted_group
            permuted_group_idx_end = group_height * (permuted_group + 1)
            image_permuted[group_idx_start:group_idx_end] = image[permuted_group_idx_start:permuted_group_idx_end]

        return image_permuted


class FrequencyDropout(ImageOnlyTransform):

    def __init__(self, n_frequencies, n_consecutive=0, drop_value=0, always_apply=False, p=0.5):

        super(FrequencyDropout, self).__init__(always_apply=always_apply, p=p)

        self.n_frequencies = n_frequencies
        self.n_consecutive = n_consecutive
        self.drop_value = drop_value

    def apply(self, image, **kwargs):

        """
        Drop frequencies randomly

        Parameters
        ----------
        image: numpy.ndarray of shape (height, width, channel)
             Image array

        Returns
        -------
        image: numpy.ndarray of shape (height, width, channel)
            Image array with dropped frequencies
        """

        # Uniformly sample N amount of indices to drop on frequency axis
        drop_idx = np.random.randint(0, image.shape[0], self.n_frequencies).tolist()

        if self.n_consecutive > 0:

            consecutive_drop_idx = []

            for idx in drop_idx:

                consecutive_drop_idx.append(idx)
                # Uniformly sample N amount of consecutive indices to drop for each sampled drop index
                consecutive_count = np.random.randint(0, self.n_consecutive + 1)

                if consecutive_count > 0:
                    for consecutive_idx in np.arange(1, consecutive_count + 1):
                        # Add each consecutive index to current drop index and append it indices that'll be dropped
                        consecutive_drop_idx.append(min(image.shape[0] - 1, (idx + consecutive_idx)))

            image[consecutive_drop_idx, :, :] = self.drop_value
        else:
            image[drop_idx, :] = self.drop_value

        return image


class SafeTimeDropout(ImageOnlyTransform):

    def __init__(self, n_time_steps, drop_value=0, always_apply=False, p=0.5):

        super(SafeTimeDropout, self).__init__(always_apply=always_apply, p=p)

        self.n_time_steps = n_time_steps
        self.drop_value = drop_value

    def apply(self, image, **kwargs):

        """
        Drop time steps from other than center 10 seconds randomly

        Parameters
        ----------
        image: numpy.ndarray of shape (height, width, channel)
             Image array

        Returns
        -------
        image: numpy.ndarray of shape (height, width, channel)
            Image array with dropped time steps except from center 10 seconds
        """

        # Find center, second per time step and number of time steps in center 10 seconds
        center = image.shape[1] // 2
        second_per_time_step = 600 // image.shape[1]
        center_10_seconds_time_steps = 10 // second_per_time_step

        # Uniformly sample N amount of indices to drop on time axis except from center 10 seconds
        center_10_seconds_start = int(center - (center_10_seconds_time_steps // 2))
        center_10_seconds_end = int(center + (center_10_seconds_time_steps // 2))
        drop_idx = np.random.choice(
            np.arange(0, center_10_seconds_start).tolist() + np.arange(center_10_seconds_end + 1, image.shape[1]).tolist(),
            self.n_time_steps,
            replace=False
        )
        image[:, drop_idx, :] = self.drop_value

        return image


class Center10SecondsDropout(ImageOnlyTransform):

    def __init__(self, n_time_steps, drop_value=0, always_apply=False, p=0.5):

        super(Center10SecondsDropout, self).__init__(always_apply=always_apply, p=p)

        self.n_time_steps = n_time_steps
        self.drop_value = drop_value

    def apply(self, image, **kwargs):

        """
        Drop time steps from center 10 seconds randomly

        Parameters
        ----------
        image: numpy.ndarray of shape (height, width, channel)
             Image array

        Returns
        -------
        image: numpy.ndarray of shape (height, width, channel)
            Image array with dropped time steps from center 10 seconds
        """

        # Find center, second per time step and number of time steps in center 10 seconds
        center = image.shape[1] // 2
        second_per_time_step = 600 // image.shape[1]
        center_10_seconds_time_steps = 10 // second_per_time_step

        # Uniformly sample N amount of indices to drop on time axis between center 10 seconds
        center_10_seconds_start = int(center - (center_10_seconds_time_steps // 2))
        center_10_seconds_end = int(center + (center_10_seconds_time_steps // 2))
        drop_idx = np.random.randint(center_10_seconds_start, center_10_seconds_end + 1, self.n_time_steps)
        image[:, drop_idx] = self.drop_value

        return image


def get_spectrogram_transforms(**transform_parameters):

    """
    Get spectrogram transforms for dataset

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
        #A.HorizontalFlip(p=transform_parameters['horizontal_flip_probability']),
        #A.VerticalFlip(p=transform_parameters['vertical_flip_probability']),
        #GroupFlip(flip_dimensions=0, n_groups=1, p=0.2),
        #GroupPermute(p=0.1),
        #A.RandomBrightnessContrast(
        #    brightness_limit=transform_parameters['brightness_limit'],
        #    contrast_limit=transform_parameters['contrast_limit'],
        #    brightness_by_max=True,
        #    p=transform_parameters['random_brightness_contrast_probability']
        #),
        #A.RandomCrop(
        #    height=transform_parameters['crop_height'],
        #    width=transform_parameters['crop_width'],
        #    always_apply=True
        #),
        A.Resize(
            height=transform_parameters['resize_height'],
            width=transform_parameters['resize_width'],
            interpolation=cv2.INTER_LINEAR,
            always_apply=True
        ),
        #FrequencyDropout(
        #    n_frequencies=transform_parameters['n_frequencies'],
        #    n_consecutive=transform_parameters['n_consecutive'],
        #    drop_value=0,
        #    p=transform_parameters['frequency_dropout_probability']
        #),
        #Center10SecondsDropout(
        #    n_time_steps=transform_parameters['center_n_time_steps'],
        #    drop_value=0,
        #    p=transform_parameters['center_10_second_dropout_probability']
        #),
        #SafeTimeDropout(
        #    n_time_steps=transform_parameters['safe_n_time_steps'],
        #    drop_value=0,
        #    p=transform_parameters['safe_time_dropout_probability']
        #),
        InstanceNormalization(channel_wise=True, always_apply=True),
        ToTensorV2(always_apply=True)
    ])

    inference_transforms = A.Compose([
        #A.CenterCrop(
        #    height=transform_parameters['crop_height'],
        #    width=transform_parameters['crop_width'],
        #    always_apply=True
        #),
        A.Resize(
            height=transform_parameters['resize_height'],
            width=transform_parameters['resize_width'],
            interpolation=cv2.INTER_LINEAR,
            always_apply=True
        ),
        InstanceNormalization(channel_wise=True, always_apply=True),
        ToTensorV2(always_apply=True)
    ])

    spectrogram_transforms = {'training': training_transforms, 'inference': inference_transforms}
    return spectrogram_transforms


def get_dataset_transforms(dataset_type, **transform_parameters):

    if dataset_type == 'spectrogram':
        dataset_transforms = get_spectrogram_transforms(**transform_parameters)
    elif dataset_type == 'eeg_spectrogram':
        dataset_transforms = get_spectrogram_transforms(**transform_parameters)

    return dataset_transforms
