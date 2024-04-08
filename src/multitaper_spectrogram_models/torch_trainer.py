import os
import sys
import argparse
import yaml
import json
from glob import glob
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim

import preprocessing
import torch_datasets
import torch_modules
import torch_utilities
import transforms
sys.path.append('..')
import settings
import metrics
import visualization


def train(training_loader, model, criterion, optimizer, device, task_type, scheduler=None, amp=False):

    """
    Train given model on given data loader

    Parameters
    ----------
    training_loader: torch.utils.data.DataLoader
        Training set data loader

    model: torch.nn.Module
        Model to train

    criterion: torch.nn.Module
        Loss function

    optimizer: torch.optim.Optimizer
        Optimizer for updating model weights

    device: torch.device
        Location of the model and inputs

    task_type: str
        Task type ('binary' or 'multiclass')

    scheduler: torch.optim.LRScheduler or None
        Learning rate scheduler

    amp: bool
        Whether to use auto mixed precision or not

    Returns
    -------
    training_results: dict
        Dictionary of training losses and scores after model is fully trained on training set data loader
    """

    model.train()
    progress_bar = tqdm(training_loader)

    running_loss = 0.0
    training_targets = []
    training_predictions = []
    training_sample_qualities = []

    if amp:
        grad_scaler = torch.cuda.amp.GradScaler()
    else:
        grad_scaler = None

    for step, (inputs, targets, target_classes, sample_qualities) in enumerate(progress_bar):

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        if amp:
            with torch.autocast(device_type=device.type):
                outputs = model(inputs.half())
        else:
            outputs = model(inputs)

        loss = criterion(outputs, targets)

        if amp:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.detach().item() * len(inputs)
        training_sample_qualities.append(sample_qualities)
        training_targets.append(targets.cpu())
        training_predictions.append(outputs.detach().cpu())

        lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
        progress_bar.set_description(f'lr: {lr:.8f} - training loss: {running_loss / len(training_loader.sampler):.4f}')

    training_sample_qualities = torch.concatenate(training_sample_qualities).numpy()
    training_targets = torch.concatenate(training_targets).numpy()

    if task_type == 'binary':
        training_predictions = torch.sigmoid(torch.concatenate(training_predictions).float()).numpy()
        training_predictions = training_predictions / training_predictions.sum(axis=1).reshape(-1, 1)
    elif task_type == 'multiclass':
        training_predictions = torch.softmax(torch.concatenate(training_predictions).float(), dim=-1).numpy()
    else:
        raise ValueError(f'Invalid task type {task_type}')

    target_columns = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    prediction_columns = [f'{column}_prediction' for column in target_columns]
    df_training_predictions = pd.DataFrame(
        data=np.hstack((
            training_targets,
            training_predictions,
            training_sample_qualities.reshape(-1, 1)
        )),
        columns=target_columns + prediction_columns + ['sample_quality']
    )

    training_loss = running_loss / len(training_loader.sampler)
    training_results = {'loss': training_loss}

    for sample_quality in [0, 1, 2]:

        sample_quality_mask = df_training_predictions['sample_quality'] == sample_quality

        if sample_quality_mask.sum() == 0:
            continue

        sample_quality_training_scores = metrics.multiclass_classification_scores(
            df_targets=df_training_predictions.loc[sample_quality_mask, target_columns],
            df_predictions=df_training_predictions.loc[sample_quality_mask, prediction_columns]
        )
        sample_quality_training_scores = {f'sq_{sample_quality}_{k}': v for k, v in sample_quality_training_scores.items()}
        training_results.update(sample_quality_training_scores)

    training_scores = metrics.multiclass_classification_scores(
        df_targets=df_training_predictions[target_columns],
        df_predictions=df_training_predictions[prediction_columns]
    )
    training_results.update(training_scores)

    return training_results


def validate(validation_loader, model, criterion, device, task_type, amp=False):

    """
    Validate given model on given data loader

    Parameters
    ----------
    validation_loader: torch.utils.data.DataLoader
        Validation set data loader

    model: torch.nn.Module
        Model to validate

    criterion: torch.nn.Module
        Loss function

    task_type: str
        Task type ('binary' or 'multiclass')

    device: torch.device
        Location of the model and inputs

    amp: bool
        Whether to use auto mixed precision or not

    Returns
    -------
    validation_results: dict
        Dictionary of validation losses and scores after model is fully validated on validation set data loader
    """

    model.eval()
    progress_bar = tqdm(validation_loader)

    running_loss = 0.0
    validation_targets = []
    validation_predictions = []
    validation_sample_qualities = []

    for step, (inputs, targets, target_classes, sample_qualities) in enumerate(progress_bar):

        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            if amp:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs.half())
            else:
                outputs = model(inputs)

        loss = criterion(outputs, targets)
        running_loss += loss.detach().item() * len(inputs)
        validation_sample_qualities.append(sample_qualities)
        validation_targets.append(targets.cpu())
        validation_predictions.append(outputs.detach().cpu())

        progress_bar.set_description(f'validation loss: {running_loss / len(validation_loader.sampler):.4f}')

    validation_sample_qualities = torch.concatenate(validation_sample_qualities).numpy()
    validation_targets = torch.concatenate(validation_targets).numpy()

    if task_type == 'binary':
        validation_predictions = torch.sigmoid(torch.concatenate(validation_predictions).float()).numpy()
        validation_predictions = validation_predictions / validation_predictions.sum(axis=1).reshape(-1, 1)
    elif task_type == 'multiclass':
        validation_predictions = torch.softmax(torch.concatenate(validation_predictions).float(), dim=-1).numpy()
    else:
        raise ValueError(f'Invalid task type {task_type}')

    target_columns = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    prediction_columns = [f'{column}_prediction' for column in target_columns]
    df_validation_predictions = pd.DataFrame(
        data=np.hstack((
            validation_targets,
            validation_predictions,
            validation_sample_qualities.reshape(-1, 1)
        )),
        columns=target_columns + prediction_columns + ['sample_quality']
    )

    validation_loss = running_loss / len(validation_loader.sampler)
    validation_results = {'loss': validation_loss}

    for sample_quality in [0, 1, 2]:

        sample_quality_mask = df_validation_predictions['sample_quality'] == sample_quality

        if sample_quality_mask.sum() == 0:
            continue

        sample_quality_validation_scores = metrics.multiclass_classification_scores(
            df_targets=df_validation_predictions.loc[sample_quality_mask, target_columns],
            df_predictions=df_validation_predictions.loc[sample_quality_mask, prediction_columns]
        )
        sample_quality_validation_scores = {f'sq_{sample_quality}_{k}': v for k, v in sample_quality_validation_scores.items()}
        validation_results.update(sample_quality_validation_scores)

    validation_scores = metrics.multiclass_classification_scores(
        df_targets=df_validation_predictions[target_columns],
        df_predictions=df_validation_predictions[prediction_columns]
    )
    validation_results.update(validation_scores)

    return validation_results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory', type=str)
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    model_directory = Path(settings.MODELS / args.model_directory)
    config = yaml.load(open(model_directory / 'config.yaml'), Loader=yaml.FullLoader)
    settings.logger.info(f'Running {model_directory} model in {args.mode} mode')

    spectrogram_dataset_path = settings.DATA / config['dataset']['spectrogram_dataset']
    normalize_targets = config['dataset']['normalize_targets']

    df = pd.read_csv(settings.DATA / 'hms-harmful-brain-activity-classification' / 'hms_train_metadata_expanded.csv')
    settings.logger.info(f'Dataset Shape: {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

    # Read and merge precomputed folds
    df_folds = pd.read_csv(settings.DATA / 'folds.csv')

    df = df.merge(df_folds, on=['eeg_id', 'eeg_sub_id', 'spectrogram_id', 'spectrogram_sub_id'], how='left').reset_index(drop=True)
    df = df.groupby('sp').agg({
        'eeg_id': 'first',
        'eeg_sub_id': 'first',
        'eeg_label_offset_seconds': list,
        'spectrogram_id': 'first',
        'spectrogram_sub_id': 'first',
        'spectrogram_label_offset_seconds': list,
        'seizure_vote': 'first',
        'lpd_vote': 'first',
        'gpd_vote': 'first',
        'lrda_vote': 'first',
        'grda_vote': 'first',
        'other_vote': 'first',
        'label_type': 'first',
        'expert_consensus': 'first',
        'fold1': 'first', 'fold2': 'first', 'fold3': 'first', 'fold4': 'first', 'fold5': 'first'
    }).reset_index()
    del df_folds

    target_columns = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    df = preprocessing.extract_sample_quality(df=df, target_columns=target_columns)
    if normalize_targets:
        df = preprocessing.normalize_targets(df=df, target_columns=target_columns)

    torch.multiprocessing.set_sharing_strategy('file_system')

    input_dimensions = config['training']['input_dimensions']
    if input_dimensions == 2:
        dataset_transforms = transforms.get_multitaper_spectrogram_2d_transforms(**config['transforms'])
    else:
        raise ValueError(f'Invalid input dimensions {input_dimensions}')

    if args.mode == 'training':

        training_metadata = defaultdict(dict)

        for fold in config['training']['folds']:

            training_idx, validation_idx = df.loc[df[fold] != 1].index, df.loc[df[fold] == 1].index
            # Validate on training set if validation is set is not specified
            if len(validation_idx) == 0:
                validation_idx = training_idx

            training_sample_qualities = config['training']['training_sample_qualities']
            sample_quality_idx = np.where(df['sample_quality'].isin(training_sample_qualities))[0]
            training_idx = training_idx.intersection(sample_quality_idx)

            # Create training and validation inputs and targets
            training_spectrogram_paths, training_targets, training_target_classes, training_sample_qualities, training_sps = torch_datasets.prepare_data(
                df.loc[training_idx],
                spectrogram_dataset_path=spectrogram_dataset_path,
            )
            validation_spectrogram_paths, validation_targets, validation_target_classes, validation_sample_qualities, validation_sps = torch_datasets.prepare_data(
                df=df.loc[validation_idx],
                spectrogram_dataset_path=spectrogram_dataset_path
            )

            settings.logger.info(
                f'''
                Fold: {fold}
                Training: {len(training_targets)} ({len(training_targets) // config["training"]["training_batch_size"] + 1} steps)
                Validation {len(validation_targets)} ({len(validation_targets) // config["training"]["test_batch_size"] + 1} steps)
                '''
            )

            # Create training and validation datasets and dataloaders
            training_dataset = torch_datasets.MultitaperSpectrogramDataset(
                spectrogram_paths=training_spectrogram_paths,
                targets=training_targets,
                target_classes=training_target_classes,
                sample_qualities=training_sample_qualities,
                eeg_offset_seconds=training_sps,
                log_transform=config['transforms']['log_transform'],
                center_idx=config['transforms']['center_idx'],
                transforms=dataset_transforms['training'],
                stationary_period_random_subsample_probability=config['transforms']['stationary_period_random_subsample_probability'],
                mixup_alpha=config['transforms']['mixup_alpha'],
                mixup_probability=config['transforms']['mixup_probability'],
                mixup_center_probability=config['transforms']['mixup_center_probability']
            )
            training_loader = DataLoader(
                training_dataset,
                batch_size=config['training']['training_batch_size'],
                sampler=RandomSampler(training_dataset, replacement=False),
                pin_memory=False,
                drop_last=False,
                num_workers=config['training']['num_workers']
            )
            validation_dataset = torch_datasets.MultitaperSpectrogramDataset(
                spectrogram_paths=validation_spectrogram_paths,
                targets=validation_targets,
                target_classes=validation_target_classes,
                sample_qualities=validation_sample_qualities,
                eeg_offset_seconds=validation_sps,
                log_transform=config['transforms']['log_transform'],
                center_idx=config['transforms']['center_idx'],
                transforms=dataset_transforms['inference'],
                stationary_period_random_subsample_probability=0.,
                mixup_alpha=None,
                mixup_probability=0.,
                mixup_center_probability=0.,
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=config['training']['test_batch_size'],
                sampler=SequentialSampler(validation_dataset),
                pin_memory=False,
                drop_last=False,
                num_workers=config['training']['num_workers']
            )

            # Set model, device and seed for reproducible results
            torch_utilities.set_seed(config['training']['random_state'], deterministic_cudnn=config['training']['deterministic_cudnn'])
            device = torch.device(config['training']['device'])
            task_type = config['training']['task_type']
            criterion = getattr(torch_modules, config['training']['loss_function'])(**config['training']['loss_function_args'])

            model = getattr(torch_modules, config['model']['model_class'])(**config['model']['model_args'])
            model_checkpoint_path = config['model']['model_checkpoint_paths'][int(fold[-1]) - 1]
            if model_checkpoint_path is not None:
                model_checkpoint_path = settings.MODELS / model_checkpoint_path
                model.load_state_dict(torch.load(model_checkpoint_path), strict=False)
            model.to(device)

            # Set optimizer, learning rate scheduler and stochastic weight averaging
            optimizer = getattr(torch.optim, config['training']['optimizer'])(model.parameters(), **config['training']['optimizer_args'])
            scheduler = getattr(optim.lr_scheduler, config['training']['lr_scheduler'])(optimizer, **config['training']['lr_scheduler_args'])
            amp = config['training']['amp']

            best_epoch = 1
            early_stopping = False
            early_stopping_patience = config['training']['early_stopping_patience']
            early_stopping_metric = config['training']['early_stopping_metric']
            training_history = {f'{dataset}_{metric}': [] for metric in config['persistence']['save_best_metrics'] for dataset in ['training', 'validation']}

            for epoch in range(1, config['training']['epochs'] + 1):

                if early_stopping:
                    break

                training_results = train(
                    training_loader=training_loader,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    task_type=task_type,
                    scheduler=scheduler,
                    amp=amp
                )

                validation_results = validate(
                    validation_loader=validation_loader,
                    model=model,
                    criterion=criterion,
                    device=device,
                    task_type=task_type,
                    amp=amp
                )

                settings.logger.info(
                    f'''
                    Epoch {epoch}
                    Training Results: {json.dumps(training_results, indent=2)}
                    Validation Results: {json.dumps(validation_results, indent=2)}
                    '''
                )

                if epoch in config['persistence']['save_epochs']:
                    # Save model if epoch is specified to be saved
                    model_name = f'model_fold_{fold[-1]}_epoch_{epoch}.pt'
                    torch.save(model.state_dict(), model_directory / model_name)
                    settings.logger.info(f'Saved {model_name} to {model_directory}')

                for metric in config['persistence']['save_best_metrics']:
                    best_validation_metric = np.min(training_history[f'validation_{metric}']) if len(training_history[f'validation_{metric}']) > 0 else np.inf
                    last_validation_metric = validation_results[metric]
                    if last_validation_metric < best_validation_metric:

                        previous_model = glob(str(model_directory / f'model_fold_{fold[-1]}_epoch_*_best_{metric}*'))
                        if len(previous_model) > 0:
                            os.remove(previous_model[0])
                            settings.logger.info(f'Deleted {previous_model[0].split("/")[-1]} from {model_directory}')

                        # Save model if specified validation metric improves
                        model_name = f'model_fold_{fold[-1]}_epoch_{epoch}_best_{metric}_{last_validation_metric:.4f}.pt'
                        torch.save(model.state_dict(), model_directory / model_name)
                        settings.logger.info(f'Saved {model_name} to {model_directory} (validation {metric} decreased from {best_validation_metric:.6f} to {last_validation_metric:.6f})\n')

                    training_history[f'training_{metric}'].append(training_results[metric])
                    training_history[f'validation_{metric}'].append(validation_results[metric])

                best_epoch = np.argmin(training_history[f'validation_{early_stopping_metric}'])
                if early_stopping_patience > 0:
                    # Trigger early stopping if early stopping patience is greater than 0
                    if len(training_history[f'validation_{early_stopping_metric}']) - best_epoch > early_stopping_patience:
                        settings.logger.info(
                            f'''
                            Early Stopping (validation {early_stopping_metric} didn\'t improve for {early_stopping_patience} epochs)
                            Best Epoch ({best_epoch + 1}) Validation {early_stopping_metric}: {training_history[early_stopping_metric][best_epoch]:.4f}
                            '''
                        )
                        early_stopping = True

            training_metadata[fold] = {}
            training_metadata[fold][f'training_history'] = training_history
            for metric in config['persistence']['save_best_metrics']:
                best_epoch = int(np.argmin(training_history[f'validation_{metric}']))
                training_metadata[fold][f'best_epoch_{metric}'] = best_epoch + 1
                training_metadata[fold][f'training_{metric}'] = float(training_history[f'training_{metric}'][best_epoch])
                training_metadata[fold][f'validation_{metric}'] = float(training_history[f'validation_{metric}'][best_epoch])
                visualization.visualize_learning_curve(
                    training_scores=training_metadata[fold]['training_history'][f'training_{metric}'],
                    validation_scores=training_metadata[fold]['training_history'][f'validation_{metric}'],
                    best_epoch=training_metadata[fold][f'best_epoch_{metric}'] - 1,
                    metric=metric,
                    path=model_directory / f'learning_curve_fold_{fold[-1]}_{metric}.png'
                )
                settings.logger.info(f'Saved learning_curve_fold_{fold[-1]}_{metric}.png to {model_directory}')

        with open(model_directory / 'training_metadata.json', mode='w') as f:
            json.dump(training_metadata, f, indent=2, ensure_ascii=False)

    elif args.mode == 'test':

        df_scores = []
        prediction_columns = [f'{column}_prediction' for column in target_columns]

        folds = config['test']['folds']
        model_file_names = config['test']['model_file_names']
        has_10s = config['test']['has_10s']

        for fold, model_file_name in zip(folds, model_file_names):

            validation_idx = df.loc[df[fold] == 1].index

            # Create validation inputs and targets
            validation_spectrogram_paths, validation_targets, validation_target_classes, validation_sample_qualities, validation_sps = torch_datasets.prepare_data(
                df=df.loc[validation_idx],
                spectrogram_dataset_path=spectrogram_dataset_path
            )

            settings.logger.info(
                f'''
                Fold: {fold} ({model_file_name})
                Validation {len(validation_targets)} ({len(validation_targets) // config["training"]["test_batch_size"] + 1} steps)
                '''
            )

            # Create validation datasets and dataloaders
            validation_dataset = torch_datasets.MultitaperSpectrogramDataset(
                spectrogram_paths=validation_spectrogram_paths,
                targets=validation_targets,
                target_classes=validation_target_classes,
                sample_qualities=validation_sample_qualities,
                eeg_offset_seconds=validation_sps,
                log_transform=config['transforms']['log_transform'],
                center_idx=config['transforms']['center_idx'],
                transforms=dataset_transforms['inference'],
                stationary_period_random_subsample_probability=0.,
                mixup_alpha=None,
                mixup_probability=0.,
                mixup_center_probability=0.,
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=config['training']['test_batch_size'],
                sampler=SequentialSampler(validation_dataset),
                pin_memory=False,
                drop_last=False,
                num_workers=config['training']['num_workers']
            )

            # Set model, device and seed for reproducible results
            torch_utilities.set_seed(config['training']['random_state'], deterministic_cudnn=config['training']['deterministic_cudnn'])
            device = torch.device(config['training']['device'])
            task_type = config['training']['task_type']
            amp = config['training']['amp']

            model = getattr(torch_modules, config['model']['model_class'])(**config['model']['model_args'])
            model.load_state_dict(torch.load(model_directory / model_file_name))
            model.to(device)
            model.eval()

            validation_predictions = []

            for inputs, _, _, _ in tqdm(validation_loader):

                inputs = inputs.to(device)

                with torch.no_grad():
                    if amp:
                        with torch.autocast(device_type=device.type):
                            outputs = model(inputs.half()).float()
                    else:
                        outputs = model(inputs)

                outputs = outputs.cpu()

                if config['test']['tta']:

                    inputs = inputs.to('cpu')
                    tta_flip_dimensions = config['test']['tta_flip_dimensions']

                    tta_outputs = []
                    for dimensions in tta_flip_dimensions:
                        augmented_inputs = inputs

                        if 2 in dimensions:
                            # v-filp, flip left-right hemispheres
                            new_augmented_inputs = [inputs[:, :, :6]]
                            for idx in [1, 0, 3, 2, 4]:
                                new_augmented_inputs.append(inputs[:, :, 6+idx*100:6+(idx+1)*100])
            
                            new_augmented_inputs.append(inputs[:, :, 506:])
                            augmented_inputs = torch.concatenate(new_augmented_inputs, dim=2)

                        if 3 in dimensions:
                            # h-flip, flip individual signals
                            if has_10s:
                                new_augmented_inputs = [augmented_inputs[:, :, :, :12]]
                                for idx in range(4):
                                    new_augmented_inputs.append(
                                        torch.flip(augmented_inputs[:, :, :, 12+idx*106:-1+(idx+1)*106], dims=[3])
                                    )
                                    new_augmented_inputs.append(
                                        torch.flip(augmented_inputs[:, :, :, -1+(idx+1)*106:12+(idx+1)*106], dims=[3])
                                    )
                                new_augmented_inputs.append(inputs[:, :, :, 436:])
                            else:
                                new_augmented_inputs = [augmented_inputs[:, :, :, :6]]
                                for idx in range(4):
                                    new_augmented_inputs.append(
                                        torch.flip(augmented_inputs[:, :, :, 6+idx*93:6+(idx+1)*93], dims=[3])
                                    )
                                new_augmented_inputs.append(inputs[:, :, :, 378:])

                            augmented_inputs = torch.concatenate(new_augmented_inputs, dim=3)

                        augmented_inputs = augmented_inputs.to(device)

                        with torch.no_grad():
                            if amp:
                                with torch.autocast(device_type=device.type):
                                    augmented_outputs = model(augmented_inputs.half()).float()
                            else:
                                augmented_outputs = model(augmented_inputs)

                        tta_outputs.append(augmented_outputs.cpu())

                    outputs = torch.stack(([outputs] + tta_outputs), dim=-1)
                    outputs = torch.mean(outputs, dim=-1)

                validation_predictions += [outputs]

            if task_type == 'binary':
                validation_predictions = torch.sigmoid(torch.concatenate(validation_predictions).float()).numpy()
                validation_predictions = validation_predictions / validation_predictions.sum(axis=1).reshape(-1, 1)
            elif task_type == 'multiclass':
                validation_predictions = torch.softmax(torch.concatenate(validation_predictions).float(), dim=-1).numpy()
            else:
                raise ValueError(f'Invalid task type {task_type}')

            df.loc[validation_idx, prediction_columns] = validation_predictions
            validation_scores = metrics.multiclass_classification_scores(
                df_targets=df.loc[validation_idx, target_columns],
                df_predictions=df.loc[validation_idx, prediction_columns]
            )
            for sample_quality in [0, 1, 2]:
                sample_quality_mask = df['sample_quality'] == sample_quality
                sample_quality_validation_idx = validation_idx.intersection(np.where(sample_quality_mask)[0])
                sample_quality_validation_scores = metrics.multiclass_classification_scores(
                    df_targets=df.loc[sample_quality_validation_idx, target_columns],
                    df_predictions=df.loc[sample_quality_validation_idx, prediction_columns]
                )
                sample_quality_validation_scores = {f'sq_{sample_quality}_{k}': v for k, v in sample_quality_validation_scores.items()}
                validation_scores.update(sample_quality_validation_scores)
            df_scores.append(validation_scores)
            settings.logger.info(f'{fold} Validation Scores: {json.dumps(validation_scores, indent=2)}')

        df_scores = pd.DataFrame(df_scores)
        settings.logger.info(
            f'''
            Mean Validation Scores
            {json.dumps(df_scores.mean(axis=0).to_dict(), indent=2)}
            and Standard Deviations
            ±{json.dumps(df_scores.std(axis=0).to_dict(), indent=2)}
            '''
        )
        visualization.visualize_scores(
            df_scores=df_scores,
            title=f'Scores of {len(folds)} Model(s)',
            path=model_directory / 'scores.png'
        )
        settings.logger.info(f'scores.png is saved to {model_directory}')

        oof_scores = metrics.multiclass_classification_scores(
            df_targets=df.loc[:, target_columns],
            df_predictions=df.loc[:, prediction_columns]
        )
        for sample_quality in [0, 1, 2]:
            sample_quality_mask = df['sample_quality'] == sample_quality
            sample_quality_oof_scores = metrics.multiclass_classification_scores(
                df_targets=df.loc[sample_quality_mask, target_columns],
                df_predictions=df.loc[sample_quality_mask, prediction_columns]
            )
            sample_quality_oof_scores = {f'sq_{sample_quality}_{k}': v for k, v in sample_quality_oof_scores.items()}
            oof_scores.update(sample_quality_oof_scores)
        settings.logger.info(f'OOF Scores: {json.dumps(oof_scores, indent=2)}')

        df_scores = pd.concat((
            df_scores,
            pd.DataFrame([oof_scores])
        ), axis=0).reset_index(drop=True)
        df_scores['fold'] = ['1', '2', '3', '4', '5', 'OOF']
        df_scores.to_csv(model_directory / 'scores.csv', index=False)
        settings.logger.info(f'scores.csv is saved to {model_directory}')

        for target_column, prediction_column in zip(target_columns, prediction_columns):
            visualization.visualize_predictions(
                y_true=df[target_column],
                y_pred=df[prediction_column],
                title=f'{target_column} and {prediction_column} Histograms (OOF Predictions of {len(folds)} Model(s)',
                path=model_directory / f'{target_column}_histogram.png'
            )
            settings.logger.info(f'{target_column}_histogram.png is saved to {model_directory}')

        visualization.visualize_confusion_matrix(
            y_true=np.argmax(df[target_columns], axis=1),
            y_pred=np.argmax(df[prediction_columns], axis=1),
            title='OOF Confusion Matrix',
            path=model_directory / 'confusion_matrix.png'
        )
        settings.logger.info(f'confusion_matrix.png is saved to {model_directory}')

        for sample_quality in [0, 1, 2]:
            sample_quality_mask = df['sample_quality'] == sample_quality
            visualization.visualize_confusion_matrix(
                y_true=np.argmax(df.loc[sample_quality_mask, target_columns], axis=1),
                y_pred=np.argmax(df.loc[sample_quality_mask, prediction_columns], axis=1),
                title=f'Sample Quality {sample_quality} OOF Confusion Matrix',
                path=model_directory / f'sq_{sample_quality}_confusion_matrix.png'
            )
            settings.logger.info(f'sq_{sample_quality}_confusion_matrix.png is saved to {model_directory}')

        id_columns = ['eeg_id', 'eeg_sub_id', 'spectrogram_id', 'spectrogram_sub_id']
        df[id_columns + target_columns + prediction_columns].to_csv(model_directory / 'oof_predictions.csv', index=False)
        settings.logger.info(f'oof_predictions.csv is saved to {model_directory}')
