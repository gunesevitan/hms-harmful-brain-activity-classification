import sys
import argparse
import yaml
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim

sys.path.append('..')
import settings
import metrics
import visualization
import torch_datasets
import torch_modules
import torch_utilities
import transforms


def train(training_loader, model, criterion, optimizer, device, scheduler=None, amp=False):

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

    if amp:
        grad_scaler = torch.cuda.amp.GradScaler()
    else:
        grad_scaler = None

    for step, (inputs, targets) in enumerate(progress_bar):

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        if amp:
            with torch.cuda.amp.autocast():
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
        training_targets.append(targets.cpu())
        training_predictions.append(outputs.detach().cpu())

        lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
        progress_bar.set_description(f'lr: {lr:.8f} - training loss: {running_loss / len(training_loader.sampler):.4f}')

    training_targets = torch.concatenate(training_targets).numpy()
    training_predictions = torch.softmax(torch.concatenate(training_predictions).float(), dim=-1).numpy()

    target_columns = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    prediction_columns = [f'{column}_prediction' for column in target_columns]
    df_training_predictions = pd.DataFrame(
        data=np.hstack((
            training_targets,
            training_predictions,
        )),
        columns=target_columns + prediction_columns
    )

    training_loss = running_loss / len(training_loader.sampler)
    training_results = {'loss': training_loss}

    training_scores = metrics.multiclass_classification_scores(
        df_targets=df_training_predictions[target_columns],
        df_predictions=df_training_predictions[prediction_columns]
    )
    training_results.update(training_scores)

    return training_results


def validate(validation_loader, model, criterion, device, amp=False):

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

    for step, (inputs, targets) in enumerate(progress_bar):

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
        validation_targets.append(targets.cpu())
        validation_predictions.append(outputs.detach().cpu())

        progress_bar.set_description(f'validation loss: {running_loss / len(validation_loader.sampler):.4f}')

    validation_targets = torch.concatenate(validation_targets).numpy()
    validation_predictions = torch.softmax(torch.concatenate(validation_predictions).float(), dim=-1).numpy()

    target_columns = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    prediction_columns = [f'{column}_prediction' for column in target_columns]
    df_validation_predictions = pd.DataFrame(
        data=np.hstack((
            validation_targets,
            validation_predictions
        )),
        columns=target_columns + prediction_columns
    )

    validation_loss = running_loss / len(validation_loader.sampler)
    validation_results = {'loss': validation_loss}

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

    config = yaml.load(open(settings.MODELS / args.model_directory / 'config.yaml', 'r'), Loader=yaml.FullLoader)
    settings.logger.info(f'Running {config["persistence"]["model_directory"]} model in {args.mode} mode')

    # Create directory for models and results
    model_directory = Path(settings.MODELS / args.model_directory)
    model_directory.mkdir(parents=True, exist_ok=True)

    dataset_type = config['dataset']['dataset_type']
    eeg_dataset_path = settings.DATA / config['dataset']['eeg_dataset']
    spectrogram_dataset_path = settings.DATA / config['dataset']['spectrogram_dataset']
    normalize_targets = config['dataset']['normalize_targets']
    top_n_samples = config['dataset']['top_n_samples']

    df = pd.read_csv(settings.DATA / 'hms-harmful-brain-activity-classification' / 'train.csv')
    settings.logger.info(f'Dataset Shape: {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

    target_columns = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    if normalize_targets:
        df['total_vote'] = df[target_columns].sum(axis=1)
        df[target_columns] /= df['total_vote'].values.reshape(-1, 1)
    df['max_vote'] = df[target_columns].max(axis=1)
    df['expert_consensus_encoded'] = df['expert_consensus'].map({
        'Seizure': 1,
        'LPD': 2,
        'GPD': 3,
        'LRDA': 4,
        'GRDA': 5,
        'Other': 6,
    })
    df['eeg_label_offset_seconds_diff'] = df.groupby('eeg_id')['eeg_label_offset_seconds'].diff()
    df['expert_consensus_encoded_diff'] = df.groupby('eeg_id')['expert_consensus_encoded'].diff()

    # Read and merge precomputed folds
    df_folds = pd.read_csv(settings.DATA / 'folds.csv')
    df = df.merge(df_folds, on=['eeg_id', 'eeg_sub_id', 'spectrogram_id', 'spectrogram_sub_id'], how='left')
    del df_folds

    torch.multiprocessing.set_sharing_strategy('file_system')

    if args.mode == 'training':

        dataset_transforms = transforms.get_dataset_transforms(dataset_type=dataset_type, **config['transforms'])
        training_metadata = defaultdict(dict)

        for fold in config['training']['folds']:

            training_idx, validation_idx = df.loc[df[fold] != 1].index, df.loc[df[fold] == 1].index
            # Validate on training set if validation is set is not specified
            if len(validation_idx) == 0:
                validation_idx = training_idx

            condition = ~(df['eeg_label_offset_seconds_diff'] < 10) & (df['expert_consensus_encoded_diff'] != 0)
            training_idx = training_idx.intersection(np.where(condition)[0])

            # Create training and validation inputs and targets
            training_eeg_paths, training_spectrogram_paths, training_targets, _ = torch_datasets.prepare_classification_data(
                df.loc[training_idx],
                eeg_dataset_path=eeg_dataset_path,
                spectrogram_dataset_path=spectrogram_dataset_path
            )
            validation_eeg_paths, validation_spectrogram_paths, validation_targets, _ = torch_datasets.prepare_classification_data(
                df=df.loc[validation_idx],
                eeg_dataset_path=eeg_dataset_path,
                spectrogram_dataset_path=spectrogram_dataset_path
            )

            if dataset_type == 'eeg':
                training_data_paths = training_eeg_paths
                validation_data_paths = validation_eeg_paths
                dataset_class = torch_datasets.SpectrogramDataset
            elif dataset_type == 'spectrogram':
                training_data_paths = training_spectrogram_paths
                validation_data_paths = validation_spectrogram_paths
                dataset_class = torch_datasets.SpectrogramDataset
            else:
                raise ValueError(f'Invalid dataset type {dataset_type}')

            settings.logger.info(
                f'''
                Fold: {fold}
                Training: {len(training_targets)} ({len(training_targets) // config["training"]["training_batch_size"] + 1} steps)
                Validation {len(validation_targets)} ({len(validation_targets) // config["training"]["test_batch_size"] + 1} steps)
                '''
            )

            # Create training and validation datasets and dataloaders
            training_dataset = dataset_class(
                data_paths=training_data_paths,
                targets=training_targets,
                transforms=dataset_transforms['training']
            )
            training_loader = DataLoader(
                training_dataset,
                batch_size=config['training']['training_batch_size'],
                sampler=RandomSampler(training_dataset, replacement=False),
                pin_memory=False,
                drop_last=False,
                num_workers=config['training']['num_workers']
            )
            validation_dataset = dataset_class(
                data_paths=validation_data_paths,
                targets=validation_targets,
                transforms=dataset_transforms['inference']
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
            criterion = getattr(torch_modules, config['training']['loss_function'])(**config['training']['loss_function_args'])

            model = getattr(torch_modules, config['model']['model_class'])(**config['model']['model_args'])
            if config['model']['model_checkpoint_path'] is not None:
                model.load_state_dict(torch.load(config['model']['model_checkpoint_path']), strict=False)
            model.to(device)

            # Set optimizer, learning rate scheduler and stochastic weight averaging
            optimizer = getattr(torch.optim, config['training']['optimizer'])(model.parameters(), **config['training']['optimizer_args'])
            scheduler = getattr(optim.lr_scheduler, config['training']['lr_scheduler'])(optimizer, **config['training']['lr_scheduler_args'])
            amp = config['training']['amp']

            best_epoch = 1
            early_stopping = False
            training_history = {
                'training_loss': [],
                'training_log_loss': [],
                'training_kl_divergence': [],
                'validation_loss': [],
                'validation_log_loss': [],
                'validation_kl_divergence': [],
            }

            for epoch in range(1, config['training']['epochs'] + 1):

                if early_stopping:
                    break

                training_results = train(
                    training_loader=training_loader,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    scheduler=scheduler,
                    amp=amp
                )

                validation_results = validate(
                    validation_loader=validation_loader,
                    model=model,
                    criterion=criterion,
                    device=device,
                    amp=amp
                )

                settings.logger.info(
                    f'''
                    Epoch {epoch}
                    Training Results: {json.dumps(training_results, indent=2)}
                    Validation Results: {json.dumps(validation_results, indent=2)}
                    '''
                )

                if epoch in config['persistence']['save_epoch_model']:
                    # Save model if current epoch is specified to be saved
                    model_name = f'model_{fold}_epoch_{epoch}.pt'
                    torch.save(model.state_dict(), model_directory / model_name)
                    settings.logger.info(f'Saved {model_name} to {model_directory}')

                best_validation_loss = np.min(training_history['validation_loss']) if len(training_history['validation_loss']) > 0 else np.inf
                last_validation_loss = validation_results['loss']
                if last_validation_loss < best_validation_loss:
                    # Save model if validation loss improves
                    model_name = f'model_{fold}_best.pt'
                    torch.save(model.state_dict(), model_directory / model_name)
                    settings.logger.info(f'Saved {model_name} to {model_directory} (validation loss decreased from {best_validation_loss:.6f} to {last_validation_loss:.6f})\n')

                training_history['training_loss'].append(training_results['loss'])
                training_history['training_log_loss'].append(training_results['log_loss'])
                training_history['training_kl_divergence'].append(training_results['kl_divergence'])
                training_history['validation_loss'].append(validation_results['loss'])
                training_history['validation_log_loss'].append(validation_results['log_loss'])
                training_history['validation_kl_divergence'].append(validation_results['kl_divergence'])

                best_epoch = np.argmin(training_history['validation_loss'])
                if config['training']['early_stopping_patience'] > 0:
                    # Trigger early stopping if early stopping patience is greater than 0
                    if len(training_history['validation_loss']) - best_epoch > config['training']['early_stopping_patience']:
                        settings.logger.info(
                            f'''
                            Early Stopping (validation loss didn\'t improve for {config['training']["early_stopping_patience"]} epochs)
                            Best Epoch ({best_epoch + 1}) Validation Loss: {training_history["validation_loss"][best_epoch]:.4f}
                            '''
                        )
                        early_stopping = True

            training_metadata[fold] = {
                'best_epoch': int(best_epoch),
                'training_loss': float(training_history['training_loss'][best_epoch]),
                'validation_loss': float(training_history['validation_loss'][best_epoch]),
                'training_history': training_history
            }

            visualization.visualize_learning_curve(
                training_losses=training_metadata[fold]['training_history']['training_loss'],
                validation_losses=training_metadata[fold]['training_history']['validation_loss'],
                best_epoch=training_metadata[fold]['best_epoch'],
                path=model_directory / f'learning_curve_{fold}.png'
            )

            with open(model_directory / 'training_metadata.json', mode='w') as f:
                json.dump(training_metadata, f, indent=2, ensure_ascii=False)

    elif args.mode == 'test':

        dataset_transforms = transforms.get_dataset_transforms(dataset_type=dataset_type, **config['transforms'])
        df_scores = []

        prediction_columns = [f'{column}_prediction' for column in target_columns]

        folds = config['test']['folds']
        model_file_names = config['test']['model_file_names']

        for fold, model_file_name in zip(folds, model_file_names):

            validation_idx = df.loc[df[fold] == 1].index

            # Create validation inputs and targets
            validation_eeg_paths, validation_spectrogram_paths, validation_targets = torch_datasets.prepare_classification_data(
                df=df.loc[validation_idx],
                eeg_dataset_path=eeg_dataset_path,
                spectrogram_dataset_path=spectrogram_dataset_path
            )

            if dataset_type == 'eeg':
                validation_data_paths = validation_eeg_paths
                dataset_class = torch_datasets.SpectrogramDataset
            elif dataset_type == 'spectrogram':
                validation_data_paths = validation_spectrogram_paths
                dataset_class = torch_datasets.SpectrogramDataset
            else:
                raise ValueError(f'Invalid dataset type {dataset_type}')

            settings.logger.info(
                f'''
                Fold: {fold} ({model_file_name})
                Validation {len(validation_targets)} ({len(validation_targets) // config["training"]["test_batch_size"] + 1} steps)
                '''
            )

            # Create validation datasets and dataloaders
            validation_dataset = dataset_class(
                data_paths=validation_data_paths,
                targets=validation_targets,
                transforms=dataset_transforms['inference']
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
            amp = config['training']['amp']

            model = getattr(torch_modules, config['model']['model_class'])(**config['model']['model_args'])
            model.load_state_dict(torch.load(model_directory / model_file_name))
            model.to(device)
            model.eval()

            validation_predictions = []

            for inputs, _ in tqdm(validation_loader):

                inputs = inputs.to(device)

                with torch.no_grad():
                    if amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(inputs.half()).float()
                    else:
                        outputs = model(inputs)

                outputs = outputs.cpu()

                if config['test']['tta']:

                    inputs = inputs.to('cpu')
                    tta_flip_dimensions = config['test']['tta_flip_dimensions']

                    tta_outputs = []

                    for dimensions in tta_flip_dimensions:

                        augmented_inputs = torch.flip(inputs, dims=dimensions).to(device)

                        with torch.no_grad():
                            if amp:
                                with torch.cuda.amp.autocast():
                                    augmented_outputs = model(augmented_inputs.half()).float()
                            else:
                                augmented_outputs = model(augmented_inputs)

                        tta_outputs.append(augmented_outputs.cpu())

                    outputs = torch.stack(([outputs] + tta_outputs), dim=-1)
                    outputs = torch.mean(outputs, dim=-1)

                validation_predictions += [outputs]

            validation_predictions = torch.softmax(torch.concatenate(validation_predictions).float(), dim=-1).numpy()
            df.loc[validation_idx, prediction_columns] = validation_predictions
            validation_scores = metrics.multiclass_classification_scores(
                df_targets=df.loc[validation_idx, target_columns],
                df_predictions=df.loc[validation_idx, prediction_columns]
            )
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

        id_columns = ['eeg_id', 'eeg_sub_id', 'spectrogram_id', 'spectrogram_sub_id']
        df[id_columns + target_columns + prediction_columns].to_csv(model_directory / 'oof_predictions.csv', index=False)
        settings.logger.info(f'oof_predictions.csv is saved to {model_directory}')
