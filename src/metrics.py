import numpy as np
from sklearn.metrics import (
    log_loss, accuracy_score
)


def kl_divergence(df_targets, df_predictions, epsilon=1e-15, micro_average=True):

    """
    Calculate Kullback–Leibler divergence on given targets and predictions

    Parameters
    ----------
    df_targets: pd.DataFrame of shape (n_samples, 6)
        Dataframe of targets

    df_predictions: pd.DataFrame of shape (n_samples, 6)
        Dataframe of predictions

    epsilon: float
        Small number for clipping predictions

    micro_average: bool
        Whether to do micro or macro average

    Returns
    -------
    kl_divergence: float
        Calculated Kullback–Leibler divergence
    """

    df_targets = df_targets.copy(deep=True)

    for column in df_targets.columns:

        df_predictions.loc[:, f'{column}_prediction'] = np.clip(df_predictions.loc[:, f'{column}_prediction'], epsilon, 1 - epsilon)

        non_zero_idx = df_targets[column] != 0
        df_targets.loc[:, column] = df_targets.loc[:, column].astype(float)
        df_targets.loc[non_zero_idx, column] = df_targets.loc[non_zero_idx, column] * np.log(df_targets.loc[non_zero_idx, column] / df_predictions.loc[non_zero_idx, f'{column}_prediction'])
        df_targets.loc[~non_zero_idx, column] = 0

    if micro_average:
        return np.average(df_targets.sum(axis=1))
    else:
        return np.average(df_targets.mean())


def multiclass_classification_scores(df_targets, df_predictions):

    """
    Calculate multiclass classification metric scores on given ground truth and predictions

    Parameters
    ----------
    df_targets: pd.DataFrame of shape (n_samples, 6)
        Dataframe of targets

    df_predictions: pd.DataFrame of shape (n_samples, 6)
        Dataframe of predictions

    Returns
    -------
    scores: dict
        Dictionary of calculated multiclass classification metric scores
    """

    predictions_class = np.argmax(df_targets, axis=1)
    targets_class = np.argmax(df_predictions, axis=1)

    scores = {
        'kl_divergence': float(kl_divergence(df_targets, df_predictions)),
        'log_loss': float(log_loss(predictions_class, df_predictions)),
        'accuracy': float(accuracy_score(predictions_class, targets_class))
    }

    return scores
