import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

sys.path.append('..')
import settings
import preprocessing
from timm_models import load_timm_model
import raw_eeg_transforms
import transforms


def normalize_probabilities(df, columns):

    """
    Normalize probabilities to 1 within given columns

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with given columns

    columns: list
        List of column names that have probabilities

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with given columns' sum adjusted to 1
    """

    df_sums = df[columns].sum(axis=1)
    for column in columns:
        df[column] /= df_sums

    return df


def blend(df_pseudo_labels):

    """
    Blend pseudo labels in a way that has the highest OOF score of 0.2164 (0.23 LB)

    Parameters
    ----------
    df_pseudo_labels: pandas.DataFrame
        Dataframe with pseudo labels

    Returns
    -------
    df: pandas.DataFrame
        Dataframe with blended and stationary period averaged pseudo labels
    """

    target_columns = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']

    for fold in range(1, 6):

        # 50 second raw EEG prediction columns
        raw_eeg_50_second_2d_convnextbase_prediction_columns = [f'raw_eeg_50_second_2d_convnextbase_{column}_prediction_fold{fold}' for column in target_columns]
        raw_eeg_50_second_2d_maxvittiny_prediction_columns = [f'raw_eeg_50_second_2d_maxvittiny_{column}_prediction_fold{fold}' for column in target_columns]

        # 50/10 second raw EEG prediction columns
        raw_eeg_50_10_second_2d_convnextbase_prediction_columns = [f'raw_eeg_50_10_second_2d_convnextbase_{column}_prediction_fold{fold}' for column in target_columns]
        raw_eeg_50_10_second_2d_maxvittiny_prediction_columns = [f'raw_eeg_50_10_second_2d_maxvittiny_{column}_prediction_fold{fold}' for column in target_columns]

        # 50 second spectrogram prediction columns
        spectrogram_50_second_2d_convnextbase_prediction_columns = [f'spectrogram_50_second_2d_convnextbase_{column}_prediction_fold{fold}' for column in target_columns]
        spectrogram_50_second_2d_maxvittiny_prediction_columns = [f'spectrogram_50_second_2d_maxvittiny_{column}_prediction_fold{fold}' for column in target_columns]

        # 50/10 second spectrogram prediction columns
        spectrogram_50_10_second_2d_convnextbase_prediction_columns = [f'spectrogram_50_10_second_2d_convnextbase_{column}_prediction_fold{fold}' for column in target_columns]
        spectrogram_50_10_second_2d_maxvittiny_prediction_columns = [f'spectrogram_50_10_second_2d_maxvittiny_{column}_prediction_fold{fold}' for column in target_columns]

        # 50/30/10 second spectrogram prediction columns
        spectrogram_50_30_10_second_2d_convnextbase_prediction_columns = [f'spectrogram_50_30_10_second_2d_convnextbase_{column}_prediction_fold{fold}' for column in target_columns]
        spectrogram_50_30_10_second_2d_maxvittiny_prediction_columns = [f'spectrogram_50_30_10_second_2d_maxvittiny_{column}_prediction_fold{fold}' for column in target_columns]

        # Raw EEG blend
        raw_eeg_blend_prediction_columns = [f'raw_eeg_blend_{column}_prediction_fold{fold}' for column in target_columns]
        df_pseudo_labels[raw_eeg_blend_prediction_columns] = (df_pseudo_labels[raw_eeg_50_second_2d_convnextbase_prediction_columns] * 0.25).values + \
                                                             (df_pseudo_labels[raw_eeg_50_second_2d_maxvittiny_prediction_columns] * 0.25).values + \
                                                             (df_pseudo_labels[raw_eeg_50_10_second_2d_convnextbase_prediction_columns] * 0.25).values + \
                                                             (df_pseudo_labels[raw_eeg_50_10_second_2d_maxvittiny_prediction_columns] * 0.25).values

        df_pseudo_labels = normalize_probabilities(df_pseudo_labels, raw_eeg_blend_prediction_columns)

        # Spectrogram blend
        spectrogram_blend_prediction_columns = [f'spectrogram_blend_{column}_prediction_fold{fold}' for column in target_columns]
        df_pseudo_labels[spectrogram_blend_prediction_columns] = (df_pseudo_labels[spectrogram_50_second_2d_convnextbase_prediction_columns] * 0.125).values + \
                                                                 (df_pseudo_labels[spectrogram_50_second_2d_maxvittiny_prediction_columns] * 0.125).values + \
                                                                 (df_pseudo_labels[spectrogram_50_10_second_2d_convnextbase_prediction_columns] * 0.25).values + \
                                                                 (df_pseudo_labels[spectrogram_50_10_second_2d_maxvittiny_prediction_columns] * 0.25).values + \
                                                                 (df_pseudo_labels[spectrogram_50_30_10_second_2d_convnextbase_prediction_columns] * 0.125).values + \
                                                                 (df_pseudo_labels[spectrogram_50_30_10_second_2d_maxvittiny_prediction_columns] * 0.125).values

        df_pseudo_labels = normalize_probabilities(df_pseudo_labels, spectrogram_blend_prediction_columns)

        # Final blend
        blend_prediction_columns = [f'blend_{column}_prediction_fold{fold}' for column in target_columns]
        df_pseudo_labels[blend_prediction_columns] = (df_pseudo_labels[raw_eeg_blend_prediction_columns] * 0.3).values + \
                                                     (df_pseudo_labels[spectrogram_blend_prediction_columns] * 0.7).values

        df_pseudo_labels = normalize_probabilities(df_pseudo_labels, blend_prediction_columns)

    blend_prediction_columns = [f'blend_{column}_prediction_fold{fold}' for column in target_columns for fold in range(1, 6)]
    df_pseudo_labels[blend_prediction_columns] = df_pseudo_labels.groupby(['stationary_period'])[blend_prediction_columns].transform('mean')
    df_pseudo_labels = df_pseudo_labels.loc[:, ['eeg_id', 'eeg_sub_id'] + blend_prediction_columns]

    return df_pseudo_labels


if __name__ == '__main__':

    df = pd.read_csv(settings.DATA / 'hms-harmful-brain-activity-classification' / 'train.csv')
    settings.logger.info(f'Dataset Shape: {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

    # Read and merge precomputed folds
    df_folds = pd.read_csv(settings.DATA / 'folds.csv')
    df = df.merge(df_folds, on=['eeg_id', 'eeg_sub_id', 'spectrogram_id', 'spectrogram_sub_id'], how='left').reset_index(drop=True)
    del df_folds

    target_columns = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    prediction_columns = [f'{column}_prediction' for column in target_columns]
    df = preprocessing.extract_sample_quality(df=df, target_columns=target_columns)
    df = preprocessing.normalize_targets(df=df, target_columns=target_columns)

    raw_eeg_50_second_2d_convnext_models, raw_eeg_50_second_2d_convnext_models_config = load_timm_model(
        model_directory=settings.MODELS / 'raw_eeg_50_second_2d_convnextbase_384x512_quality_2',
        model_file_names=[
            'model_fold_1_epoch_4_best_sq_2_kl_divergence_0.2563.pt',
            'model_fold_2_epoch_2_best_sq_2_kl_divergence_0.2660.pt',
            'model_fold_3_epoch_6_best_sq_2_kl_divergence_0.2658.pt',
            'model_fold_4_epoch_4_best_sq_2_kl_divergence_0.2610.pt',
            'model_fold_5_epoch_5_best_sq_2_kl_divergence_0.2653.pt'
        ],
        device=torch.device('cuda')
    )

    raw_eeg_50_second_2d_maxvit_models, raw_eeg_50_second_2d_maxvit_models_config = load_timm_model(
        model_directory=settings.MODELS / 'raw_eeg_50_second_2d_maxvittiny_384x512_quality_2',
        model_file_names=[
            'model_fold_1_epoch_6_best_sq_2_kl_divergence_0.2674.pt',
            'model_fold_2_epoch_6_best_sq_2_kl_divergence_0.2728.pt',
            'model_fold_3_epoch_7_best_sq_2_kl_divergence_0.2541.pt',
            'model_fold_4_epoch_6_best_sq_2_kl_divergence_0.2706.pt',
            'model_fold_5_epoch_5_best_sq_2_kl_divergence_0.2741.pt'
        ],
        device=torch.device('cuda')
    )

    raw_eeg_50_10_second_2d_convnext_models, raw_eeg_50_10_second_2d_convnext_models_config = load_timm_model(
        model_directory=settings.MODELS / 'raw_eeg_50_10_second_2d_convnextbase_448x512_quality_2',
        model_file_names=[
            'model_fold_1_epoch_10_best_sq_2_kl_divergence_0.2634.pt',
            'model_fold_2_epoch_5_best_sq_2_kl_divergence_0.2712.pt',
            'model_fold_3_epoch_8_best_sq_2_kl_divergence_0.2592.pt',
            'model_fold_4_epoch_6_best_sq_2_kl_divergence_0.2568.pt',
            'model_fold_5_epoch_4_best_sq_2_kl_divergence_0.2685.pt'
        ],
        device=torch.device('cuda')
    )

    raw_eeg_50_10_second_2d_maxvit_models, raw_eeg_50_10_second_2d_maxvit_models_config = load_timm_model(
        model_directory=settings.MODELS / 'raw_eeg_50_10_second_2d_maxvittiny_448x512_quality_2',
        model_file_names=[
            'model_fold_1_epoch_7_best_sq_2_kl_divergence_0.2602.pt',
            'model_fold_2_epoch_3_best_sq_2_kl_divergence_0.2677.pt',
            'model_fold_3_epoch_6_best_sq_2_kl_divergence_0.2542.pt',
            'model_fold_4_epoch_6_best_sq_2_kl_divergence_0.2582.pt',
            'model_fold_5_epoch_6_best_sq_2_kl_divergence_0.2667.pt'
        ],
        device=torch.device('cuda')
    )

    spectrogram_50_second_2d_convnext_models, spectrogram_50_second_2d_convnext_models_config = load_timm_model(
        model_directory=settings.MODELS / 'spectrogram_50_second_2d_convnextbase_512x512_quality_2',
        model_file_names=[
            'model_fold_1_epoch_7_best_sq_2_kl_divergence_0.2453.pt',
            'model_fold_2_epoch_5_best_sq_2_kl_divergence_0.2485.pt',
            'model_fold_3_epoch_4_best_sq_2_kl_divergence_0.2404.pt',
            'model_fold_4_epoch_5_best_sq_2_kl_divergence_0.2440.pt',
            'model_fold_5_epoch_6_best_sq_2_kl_divergence_0.2558.pt'
        ],
        device=torch.device('cuda')
    )

    spectrogram_50_second_2d_maxvit_models, spectrogram_50_second_2d_maxvit_models_config = load_timm_model(
        model_directory=settings.MODELS / 'spectrogram_50_second_2d_maxvittiny_512x512_quality_2',
        model_file_names=[
            'model_fold_1_epoch_7_best_sq_2_kl_divergence_0.2534.pt',
            'model_fold_2_epoch_6_best_sq_2_kl_divergence_0.2342.pt',
            'model_fold_3_epoch_7_best_sq_2_kl_divergence_0.2486.pt',
            'model_fold_4_epoch_7_best_sq_2_kl_divergence_0.2647.pt',
            'model_fold_5_epoch_7_best_sq_2_kl_divergence_0.2560.pt'
        ],
        device=torch.device('cuda')
    )

    spectrogram_50_10_second_2d_convnext_models, spectrogram_50_10_second_2d_convnext_models_config = load_timm_model(
        model_directory=settings.MODELS / 'spectrogram_50_10_second_2d_convnextbase_512x512_quality_2',
        model_file_names=[
            'model_fold_1_epoch_6_best_sq_2_kl_divergence_0.2428.pt',
            'model_fold_2_epoch_7_best_sq_2_kl_divergence_0.2330.pt',
            'model_fold_3_epoch_4_best_sq_2_kl_divergence_0.2405.pt',
            'model_fold_4_epoch_7_best_sq_2_kl_divergence_0.2409.pt',
            'model_fold_5_epoch_4_best_sq_2_kl_divergence_0.2509.pt'
        ],
        device=torch.device('cuda')
    )

    spectrogram_50_10_second_2d_maxvit_models, spectrogram_50_10_second_2d_maxvit_models_config = load_timm_model(
        model_directory=settings.MODELS / 'spectrogram_50_10_second_2d_maxvittiny_512x512_quality_2',
        model_file_names=[
            'model_fold_1_epoch_6_best_sq_2_kl_divergence_0.2493.pt',
            'model_fold_2_epoch_6_best_sq_2_kl_divergence_0.2252.pt',
            'model_fold_3_epoch_6_best_sq_2_kl_divergence_0.2351.pt',
            'model_fold_4_epoch_6_best_sq_2_kl_divergence_0.2568.pt',
            'model_fold_5_epoch_7_best_sq_2_kl_divergence_0.2452.pt'
        ],
        device=torch.device('cuda')
    )

    spectrogram_50_30_10_second_2d_convnext_models, spectrogram_50_30_10_second_2d_convnext_models_config = load_timm_model(
        model_directory=settings.MODELS / 'spectrogram_50_30_10_second_2d_convnextbase_512x512_quality_2',
        model_file_names=[
            'model_fold_1_epoch_7_best_sq_2_kl_divergence_0.2464.pt',
            'model_fold_2_epoch_7_best_sq_2_kl_divergence_0.2438.pt',
            'model_fold_3_epoch_7_best_sq_2_kl_divergence_0.2395.pt',
            'model_fold_4_epoch_8_best_sq_2_kl_divergence_0.2527.pt',
            'model_fold_5_epoch_6_best_sq_2_kl_divergence_0.2537.pt'
        ],
        device=torch.device('cuda')
    )

    spectrogram_50_30_10_second_2d_maxvit_models, spectrogram_50_30_10_second_2d_maxvit_models_config = load_timm_model(
        model_directory=settings.MODELS / 'spectrogram_50_30_10_second_2d_maxvittiny_512x512_quality_2',
        model_file_names=[
            'model_fold_1_epoch_7_best_sq_2_kl_divergence_0.2512.pt',
            'model_fold_2_epoch_6_best_sq_2_kl_divergence_0.2304.pt',
            'model_fold_3_epoch_6_best_sq_2_kl_divergence_0.2364.pt',
            'model_fold_4_epoch_6_best_sq_2_kl_divergence_0.2543.pt',
            'model_fold_5_epoch_7_best_sq_2_kl_divergence_0.2496.pt'
        ],
        device=torch.device('cuda')
    )

    model_names = [
        'raw_eeg_50_second_2d_convnextbase',
        'raw_eeg_50_second_2d_maxvittiny',
        'raw_eeg_50_10_second_2d_convnextbase',
        'raw_eeg_50_10_second_2d_maxvittiny',
        'spectrogram_50_second_2d_convnextbase',
        'spectrogram_50_second_2d_maxvittiny',
        'spectrogram_50_10_second_2d_convnextbase',
        'spectrogram_50_10_second_2d_maxvittiny',
        'spectrogram_50_30_10_second_2d_convnextbase',
        'spectrogram_50_30_10_second_2d_maxvittiny',
    ]

    # Global configurations
    device = torch.device('cuda')
    amp = True
    tta = True
    task_type = 'binary'

    # Raw EEG 50 second models, configurations and transforms
    raw_eeg_50_second_2d_model_configs = [
        raw_eeg_50_second_2d_convnext_models_config,
        raw_eeg_50_second_2d_maxvit_models_config
    ]
    raw_eeg_50_second_2d_models = [
        raw_eeg_50_second_2d_convnext_models,
        raw_eeg_50_second_2d_maxvit_models
    ]
    raw_eeg_50_second_2d_transforms = [
        raw_eeg_transforms.get_raw_eeg_2d_transforms(**config['transforms'])['inference']
        for config in raw_eeg_50_second_2d_model_configs
    ]

    # Raw EEG 50/10 second models, configurations and transforms
    raw_eeg_50_10_second_2d_model_configs = [
        raw_eeg_50_10_second_2d_convnext_models_config,
        raw_eeg_50_10_second_2d_maxvit_models_config
    ]
    raw_eeg_50_10_second_2d_models = [
        raw_eeg_50_10_second_2d_convnext_models,
        raw_eeg_50_10_second_2d_maxvit_models
    ]
    raw_eeg_50_10_second_2d_transforms = [
        raw_eeg_transforms.get_raw_eeg_2d_transforms(**config['transforms'])['inference']
        for config in raw_eeg_50_10_second_2d_model_configs
    ]

    # Spectrogram 50 second models, configurations and transforms
    spectrogram_50_second_2d_model_configs = [
        spectrogram_50_second_2d_convnext_models_config,
        spectrogram_50_second_2d_maxvit_models_config
    ]
    spectrogram_50_second_2d_models = [
        spectrogram_50_second_2d_convnext_models,
        spectrogram_50_second_2d_maxvit_models,
    ]
    spectrogram_50_second_2d_transforms = [
        transforms.get_spectrogram_2d_transforms(**config['transforms'])['inference']
        for config in spectrogram_50_second_2d_model_configs
    ]

    # Spectrogram 50/10 second models, configurations and transforms
    spectrogram_50_10_second_2d_model_configs = [
        spectrogram_50_10_second_2d_convnext_models_config,
        spectrogram_50_10_second_2d_maxvit_models_config
    ]
    spectrogram_50_10_second_2d_models = [
        spectrogram_50_10_second_2d_convnext_models,
        spectrogram_50_10_second_2d_maxvit_models
    ]
    spectrogram_50_10_second_2d_transforms = [
        transforms.get_spectrogram_2d_transforms(**config['transforms'])['inference']
        for config in spectrogram_50_10_second_2d_model_configs
    ]

    # Spectrogram 50/30/10 second models, configurations and transforms
    spectrogram_50_30_10_second_2d_model_configs = [
        spectrogram_50_30_10_second_2d_convnext_models_config,
        spectrogram_50_30_10_second_2d_maxvit_models_config
    ]
    spectrogram_50_30_10_second_2d_models = [
        spectrogram_50_30_10_second_2d_convnext_models,
        spectrogram_50_30_10_second_2d_maxvit_models
    ]
    spectrogram_50_30_10_second_2d_transforms = [
        transforms.get_spectrogram_2d_transforms(**config['transforms'])['inference']
        for config in spectrogram_50_30_10_second_2d_model_configs
    ]

    get_eeg_differences = lambda x: raw_eeg_transforms.ChannelDifference1D(ekg=False, always_apply=True)(image=x)['image']

    n_raw_eeg_50_second_2d_models = 2
    n_raw_eeg_50_10_second_2d_models = 2
    n_spectrogram_50_second_2d_models = 2
    n_spectrogram_50_10_second_2d_models = 2
    n_spectrogram_50_30_10_second_2d_models = 2

    is_submission = True

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):

        if row['sample_quality'] == 2:
            continue

        eeg_path = settings.DATA / 'eeg_subsample' / 'eegs' / f'{row["eeg_id"]}_{row["eeg_sub_id"]}.npy'
        eeg = pd.DataFrame(np.load(eeg_path))
        eeg = eeg.interpolate(method='linear', limit_area='inside').fillna(0).values

        # Create raw EEG 50 second predictions tensors for the current sample
        raw_eeg_50_second_2d_predictions = torch.zeros(n_raw_eeg_50_second_2d_models, 5, 6)

        for model_idx, (models, transforms) in enumerate(zip(raw_eeg_50_second_2d_models, raw_eeg_50_second_2d_transforms)):

            raw_eeg_50_second_2d_inputs = transforms(image=eeg)['image'].float()
            raw_eeg_50_second_2d_inputs = raw_eeg_50_second_2d_inputs.to(device)

            if tta:
                raw_eeg_50_second_2d_inputs = torch.stack((
                    raw_eeg_50_second_2d_inputs,
                    torch.flip(raw_eeg_50_second_2d_inputs, dims=(1,)),
                    torch.flip(raw_eeg_50_second_2d_inputs, dims=(2,)),
                    torch.flip(raw_eeg_50_second_2d_inputs, dims=(1, 2))
                ), dim=0)
            else:
                raw_eeg_50_second_2d_inputs = torch.unsqueeze(raw_eeg_50_second_2d_inputs, dim=0)

            for fold, model in enumerate(models.values()):
                with torch.no_grad():
                    if amp:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            raw_eeg_50_second_2d_outputs = model(raw_eeg_50_second_2d_inputs)
                    else:
                        raw_eeg_50_second_2d_outputs = model(raw_eeg_50_second_2d_inputs)

                raw_eeg_50_second_2d_outputs = raw_eeg_50_second_2d_outputs.cpu()
                if tta:
                    raw_eeg_50_second_2d_outputs = torch.mean(raw_eeg_50_second_2d_outputs, dim=0)
                else:
                    raw_eeg_50_second_2d_outputs = torch.squeeze(raw_eeg_50_second_2d_outputs, dim=0)

                raw_eeg_50_second_2d_predictions[model_idx, fold] = raw_eeg_50_second_2d_outputs

                if is_submission is False:
                    print(f'Predicted with Raw EEG 50 Second Model {model_idx + 1} Fold {fold + 1}')

        # Create raw EEG 50/10 second predictions tensors for the current sample
        raw_eeg_50_10_second_2d_predictions = torch.zeros(n_raw_eeg_50_10_second_2d_models, 5, 6)

        for model_idx, (models, transforms) in enumerate(zip(raw_eeg_50_10_second_2d_models, raw_eeg_50_10_second_2d_transforms)):

            raw_eeg_50_10_second_2d_inputs = transforms(image=eeg)['image'].float()
            raw_eeg_50_10_second_2d_inputs = raw_eeg_50_10_second_2d_inputs.to(device)

            if tta:
                raw_eeg_50_10_second_2d_inputs = torch.stack((
                    raw_eeg_50_10_second_2d_inputs,
                    torch.flip(raw_eeg_50_10_second_2d_inputs, dims=(1,)),
                    torch.flip(raw_eeg_50_10_second_2d_inputs, dims=(2,)),
                    torch.flip(raw_eeg_50_10_second_2d_inputs, dims=(1, 2))
                ), dim=0)
            else:
                raw_eeg_50_10_second_2d_inputs = torch.unsqueeze(raw_eeg_50_10_second_2d_inputs, dim=0)

            for fold, model in enumerate(models.values()):
                with torch.no_grad():
                    if amp:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            raw_eeg_50_10_second_2d_outputs = model(raw_eeg_50_10_second_2d_inputs)
                    else:
                        raw_eeg_50_10_second_2d_outputs = model(raw_eeg_50_10_second_2d_inputs)

                raw_eeg_50_10_second_2d_outputs = raw_eeg_50_10_second_2d_outputs.cpu()
                if tta:
                    raw_eeg_50_10_second_2d_outputs = torch.mean(raw_eeg_50_10_second_2d_outputs, dim=0)
                else:
                    raw_eeg_50_10_second_2d_outputs = torch.squeeze(raw_eeg_50_10_second_2d_outputs, dim=0)

                raw_eeg_50_10_second_2d_predictions[model_idx, fold] = raw_eeg_50_10_second_2d_outputs

                if is_submission is False:
                    print(f'Predicted with Raw EEG 50/10 Second Model {model_idx + 1} Fold {fold + 1}')

        spectrogram_50_second_path = settings.DATA / 'eeg_spectrograms-50-second' / 'spectrograms' / f'{row["eeg_id"]}_{row["eeg_sub_id"]}.npy'
        spectrogram_50_second = np.load(spectrogram_50_second_path)
        spectrogram_50_second = np.log1p(spectrogram_50_second)

        # Create 50 second spectrogram predictions tensors for the current sample
        spectrogram_50_second_2d_predictions = torch.zeros(n_spectrogram_50_second_2d_models, 5, 6)

        for model_idx, (models, transforms) in enumerate(zip(spectrogram_50_second_2d_models, spectrogram_50_second_2d_transforms)):

            spectrogram_50_second_2d_inputs = transforms(image=spectrogram_50_second)['image'].float()
            spectrogram_50_second_2d_inputs = spectrogram_50_second_2d_inputs.to(device)

            if tta:
                spectrogram_50_second_2d_inputs = torch.stack((
                    spectrogram_50_second_2d_inputs,
                    torch.flip(spectrogram_50_second_2d_inputs, dims=(1,)),
                    torch.flip(spectrogram_50_second_2d_inputs, dims=(2,)),
                    torch.flip(spectrogram_50_second_2d_inputs, dims=(1, 2))
                ), dim=0)
            else:
                spectrogram_50_second_2d_inputs = torch.unsqueeze(spectrogram_50_second_2d_inputs, dim=0)

            for fold, model in enumerate(models.values()):
                with torch.no_grad():
                    if amp:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            spectrogram_50_second_2d_outputs = model(spectrogram_50_second_2d_inputs)
                    else:
                        spectrogram_50_second_2d_outputs = model(spectrogram_50_second_2d_inputs)

                spectrogram_50_second_2d_outputs = spectrogram_50_second_2d_outputs.cpu()
                if tta:
                    spectrogram_50_second_2d_outputs = torch.mean(spectrogram_50_second_2d_outputs, dim=0)
                else:
                    spectrogram_50_second_2d_outputs = torch.squeeze(spectrogram_50_second_2d_outputs, dim=0)

                spectrogram_50_second_2d_predictions[model_idx, fold] = spectrogram_50_second_2d_outputs

                if is_submission is False:
                    print(f'Predicted with Spectrogram 50 Second 2D Model {model_idx + 1} Fold {fold + 1}')

        spectrogram_50_10_second_path = settings.DATA / 'eeg_spectrograms-50-10-second' / 'spectrograms' / f'{row["eeg_id"]}_{row["eeg_sub_id"]}.npy'
        spectrogram_50_10_second = np.load(spectrogram_50_10_second_path)
        spectrogram_50_10_second = np.log1p(spectrogram_50_10_second)

        # Create 50/10 second spectrogram predictions tensors for the current sample
        spectrogram_50_10_second_2d_predictions = torch.zeros(n_spectrogram_50_10_second_2d_models, 5, 6)

        for model_idx, (models, transforms) in enumerate(zip(spectrogram_50_10_second_2d_models, spectrogram_50_10_second_2d_transforms)):

            spectrogram_50_10_second_2d_inputs = transforms(image=spectrogram_50_10_second)['image'].float()
            spectrogram_50_10_second_2d_inputs = spectrogram_50_10_second_2d_inputs.to(device)

            if tta:
                spectrogram_50_10_second_2d_inputs = torch.stack((
                    spectrogram_50_10_second_2d_inputs,
                    torch.flip(spectrogram_50_10_second_2d_inputs, dims=(1,)),
                    torch.flip(spectrogram_50_10_second_2d_inputs, dims=(2,)),
                    torch.flip(spectrogram_50_10_second_2d_inputs, dims=(1, 2))
                ), dim=0)
            else:
                spectrogram_50_10_second_2d_inputs = torch.unsqueeze(spectrogram_50_10_second_2d_inputs, dim=0)

            for fold, model in enumerate(models.values()):
                with torch.no_grad():
                    if amp:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            spectrogram_50_10_second_2d_outputs = model(spectrogram_50_10_second_2d_inputs)
                    else:
                        spectrogram_50_10_second_2d_outputs = model(spectrogram_50_10_second_2d_inputs)

                spectrogram_50_10_second_2d_outputs = spectrogram_50_10_second_2d_outputs.cpu()
                if tta:
                    spectrogram_50_10_second_2d_outputs = torch.mean(spectrogram_50_10_second_2d_outputs, dim=0)
                else:
                    spectrogram_50_10_second_2d_outputs = torch.squeeze(spectrogram_50_10_second_2d_outputs, dim=0)

                spectrogram_50_10_second_2d_predictions[model_idx, fold] = spectrogram_50_10_second_2d_outputs

                if is_submission is False:
                    print(f'Predicted with Spectrogram 50/10 Second 2D Model {model_idx + 1} Fold {fold + 1}')

        spectrogram_50_30_10_second_path = settings.DATA / 'eeg_spectrograms-50-30-10-second' / 'spectrograms' / f'{row["eeg_id"]}_{row["eeg_sub_id"]}.npy'
        spectrogram_50_30_10_second = np.load(spectrogram_50_30_10_second_path)
        spectrogram_50_30_10_second = np.log1p(spectrogram_50_30_10_second)

        # Create 50/30/10 second spectrogram predictions tensors for the current sample
        spectrogram_50_30_10_second_2d_predictions = torch.zeros(n_spectrogram_50_30_10_second_2d_models, 5, 6)

        for model_idx, (models, transforms) in enumerate(zip(spectrogram_50_30_10_second_2d_models, spectrogram_50_30_10_second_2d_transforms)):

            spectrogram_50_30_10_second_2d_inputs = transforms(image=spectrogram_50_30_10_second)['image'].float()
            spectrogram_50_30_10_second_2d_inputs = spectrogram_50_30_10_second_2d_inputs.to(device)

            if tta:
                spectrogram_50_30_10_second_2d_inputs = torch.stack((
                    spectrogram_50_30_10_second_2d_inputs,
                    torch.flip(spectrogram_50_30_10_second_2d_inputs, dims=(1,)),
                    torch.flip(spectrogram_50_30_10_second_2d_inputs, dims=(2,)),
                    torch.flip(spectrogram_50_30_10_second_2d_inputs, dims=(1, 2))
                ), dim=0)
            else:
                spectrogram_50_30_10_second_2d_inputs = torch.unsqueeze(spectrogram_50_30_10_second_2d_inputs, dim=0)

            for fold, model in enumerate(models.values()):
                with torch.no_grad():
                    if amp:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            spectrogram_50_30_10_second_2d_outputs = model(spectrogram_50_30_10_second_2d_inputs)
                    else:
                        spectrogram_50_30_10_second_2d_outputs = model(spectrogram_50_30_10_second_2d_inputs)

                spectrogram_50_30_10_second_2d_outputs = spectrogram_50_30_10_second_2d_outputs.cpu()
                if tta:
                    spectrogram_50_30_10_second_2d_outputs = torch.mean(spectrogram_50_30_10_second_2d_outputs, dim=0)
                else:
                    spectrogram_50_30_10_second_2d_outputs = torch.squeeze(spectrogram_50_30_10_second_2d_outputs, dim=0)

                spectrogram_50_30_10_second_2d_predictions[model_idx, fold] = spectrogram_50_30_10_second_2d_outputs

                if is_submission is False:
                    print(f'Predicted with Spectrogram 50/30/10 Second 2D Model {model_idx + 1} Fold {fold + 1}')

        raw_eeg_50_second_2d_convnext_predictions = raw_eeg_50_second_2d_predictions[0, :, :]
        raw_eeg_50_second_2d_maxvit_predictions = raw_eeg_50_second_2d_predictions[1, :, :]

        raw_eeg_50_10_second_2d_convnext_predictions = raw_eeg_50_10_second_2d_predictions[0, :, :]
        raw_eeg_50_10_second_2d_maxvit_predictions = raw_eeg_50_10_second_2d_predictions[1, :, :]

        spectrogram_50_second_2d_convnext_predictions = spectrogram_50_second_2d_predictions[0, :, :]
        spectrogram_50_second_2d_maxvit_predictions = spectrogram_50_second_2d_predictions[1, :, :]

        spectrogram_50_10_second_2d_convnext_predictions = spectrogram_50_10_second_2d_predictions[0, :, :]
        spectrogram_50_10_second_2d_maxvit_predictions = spectrogram_50_10_second_2d_predictions[1, :, :]

        spectrogram_50_30_10_second_2d_convnext_predictions = spectrogram_50_30_10_second_2d_predictions[0, :, :]
        spectrogram_50_30_10_second_2d_maxvit_predictions = spectrogram_50_30_10_second_2d_predictions[1, :, :]

        raw_eeg_50_second_2d_convnext_predictions = torch.sigmoid(raw_eeg_50_second_2d_convnext_predictions).numpy()
        raw_eeg_50_second_2d_convnext_predictions = raw_eeg_50_second_2d_convnext_predictions / raw_eeg_50_second_2d_convnext_predictions.sum(axis=-1).reshape(-1, 1)
        raw_eeg_50_second_2d_maxvit_predictions = torch.sigmoid(raw_eeg_50_second_2d_maxvit_predictions).numpy()
        raw_eeg_50_second_2d_maxvit_predictions = raw_eeg_50_second_2d_maxvit_predictions / raw_eeg_50_second_2d_maxvit_predictions.sum(axis=-1).reshape(-1, 1)

        raw_eeg_50_10_second_2d_convnext_predictions = torch.sigmoid(raw_eeg_50_10_second_2d_convnext_predictions).numpy()
        raw_eeg_50_10_second_2d_convnext_predictions = raw_eeg_50_10_second_2d_convnext_predictions / raw_eeg_50_10_second_2d_convnext_predictions.sum(axis=-1).reshape(-1, 1)
        raw_eeg_50_10_second_2d_maxvit_predictions = torch.sigmoid(raw_eeg_50_10_second_2d_maxvit_predictions).numpy()
        raw_eeg_50_10_second_2d_maxvit_predictions = raw_eeg_50_10_second_2d_maxvit_predictions / raw_eeg_50_10_second_2d_maxvit_predictions.sum(axis=-1).reshape(-1, 1)

        spectrogram_50_second_2d_convnext_predictions = torch.sigmoid(spectrogram_50_second_2d_convnext_predictions).numpy()
        spectrogram_50_second_2d_convnext_predictions = spectrogram_50_second_2d_convnext_predictions / spectrogram_50_second_2d_convnext_predictions.sum(axis=-1).reshape(-1, 1)
        spectrogram_50_second_2d_maxvit_predictions = torch.sigmoid(spectrogram_50_second_2d_maxvit_predictions).numpy()
        spectrogram_50_second_2d_maxvit_predictions = spectrogram_50_second_2d_maxvit_predictions / spectrogram_50_second_2d_maxvit_predictions.sum(axis=-1).reshape(-1, 1)

        spectrogram_50_10_second_2d_convnext_predictions = torch.sigmoid(spectrogram_50_10_second_2d_convnext_predictions).numpy()
        spectrogram_50_10_second_2d_convnext_predictions = spectrogram_50_10_second_2d_convnext_predictions / spectrogram_50_10_second_2d_convnext_predictions.sum(axis=-1).reshape(-1, 1)
        spectrogram_50_10_second_2d_maxvit_predictions = torch.sigmoid(spectrogram_50_10_second_2d_maxvit_predictions).numpy()
        spectrogram_50_10_second_2d_maxvit_predictions = spectrogram_50_10_second_2d_maxvit_predictions / spectrogram_50_10_second_2d_maxvit_predictions.sum(axis=-1).reshape(-1, 1)

        spectrogram_50_30_10_second_2d_convnext_predictions = torch.sigmoid(spectrogram_50_30_10_second_2d_convnext_predictions).numpy()
        spectrogram_50_30_10_second_2d_convnext_predictions = spectrogram_50_30_10_second_2d_convnext_predictions / spectrogram_50_30_10_second_2d_convnext_predictions.sum(axis=-1).reshape(-1, 1)
        spectrogram_50_30_10_second_2d_maxvit_predictions = torch.sigmoid(spectrogram_50_30_10_second_2d_maxvit_predictions).numpy()
        spectrogram_50_30_10_second_2d_maxvit_predictions = spectrogram_50_30_10_second_2d_maxvit_predictions / spectrogram_50_30_10_second_2d_maxvit_predictions.sum(axis=-1).reshape(-1, 1)

        raw_eeg_blend_predictions = (raw_eeg_50_second_2d_convnext_predictions * 0.25) + \
                                    (raw_eeg_50_second_2d_maxvit_predictions * 0.25) + \
                                    (raw_eeg_50_10_second_2d_convnext_predictions * 0.25) + \
                                    (raw_eeg_50_10_second_2d_maxvit_predictions * 0.25)

        spectrogram_blend_predictions = (spectrogram_50_second_2d_convnext_predictions * 0.125) + \
                                        (spectrogram_50_second_2d_maxvit_predictions * 0.125) + \
                                        (spectrogram_50_10_second_2d_convnext_predictions * 0.25) + \
                                        (spectrogram_50_10_second_2d_maxvit_predictions * 0.25) + \
                                        (spectrogram_50_30_10_second_2d_convnext_predictions * 0.125) + \
                                        (spectrogram_50_30_10_second_2d_maxvit_predictions * 0.125)

        blend_predictions = (raw_eeg_blend_predictions * 0.3) + \
                            (spectrogram_blend_predictions * 0.7)

        model_predictions = [
            raw_eeg_50_second_2d_convnext_predictions,
            raw_eeg_50_second_2d_maxvit_predictions,
            raw_eeg_50_10_second_2d_convnext_predictions,
            raw_eeg_50_10_second_2d_maxvit_predictions,
            spectrogram_50_second_2d_convnext_predictions,
            spectrogram_50_second_2d_maxvit_predictions,
            spectrogram_50_10_second_2d_convnext_predictions,
            spectrogram_50_10_second_2d_maxvit_predictions,
            spectrogram_50_30_10_second_2d_convnext_predictions,
            spectrogram_50_30_10_second_2d_maxvit_predictions
        ]

        for model_prediction, model_name in zip(model_predictions, model_names):
            for fold in range(5):
                df.loc[idx, [f'{model_name}_{column}_fold{fold + 1}' for column in prediction_columns]] = model_prediction[fold]

    df = df.loc[df['sample_quality'] < 2, :].reset_index(drop=True)
    prediction_columns = [column for column in df.columns.tolist() if 'prediction' in column]
    df.loc[:, ['eeg_id', 'eeg_sub_id'] + prediction_columns].to_csv(settings.DATA / 'pseudo_labels.csv', index=False)
