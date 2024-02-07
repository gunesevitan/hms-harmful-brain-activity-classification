import numpy as np


def normalize_targets(df, target_columns):

    """
    Normalize given target columns so their sum will be 1

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with given target columns

    target_columns: list
        List of names of target columns

    Returns
    -------
    df: pd.DataFrame
        Dataframe with normalized target columns
    """

    df['total_vote'] = df[target_columns].sum(axis=1)
    df[target_columns] /= df['total_vote'].values.reshape(-1, 1)

    return df


def encode_target(df):

    """
    Encode target (expert_consensus) column

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with target column

    Returns
    -------
    df: pd.DataFrame
        Dataframe with encoded target column
    """

    df['expert_consensus_encoded'] = df['expert_consensus'].map({
        'Seizure': 0,
        'LPD': 1,
        'GPD': 2,
        'LRDA': 3,
        'GRDA': 4,
        'Other': 5
    }).astype(np.uint8)

    return df
