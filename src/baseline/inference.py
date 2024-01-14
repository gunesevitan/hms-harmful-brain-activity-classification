import sys
import json
import pandas as pd

sys.path.append('..')
import settings
import metrics


if __name__ == '__main__':

    model_directory = settings.MODELS / 'baseline'
    model_directory.mkdir(parents=True, exist_ok=True)

    dataset_directory = settings.DATA / 'hms-harmful-brain-activity-classification'
    df_train = pd.read_csv(dataset_directory / 'train.csv')

    target_columns = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    df_train['total_vote'] = df_train[target_columns].sum(axis=1)
    df_train[target_columns] = df_train[target_columns] / df_train['total_vote'].values.reshape(-1, 1)

    for column in target_columns:
        df_train[f'{column}_prediction'] = df_train[column].mean()

    prediction_columns = [f'{column}_prediction' for column in target_columns]
    scores = metrics.multiclass_classification_scores(df_targets=df_train[target_columns], df_predictions=df_train[prediction_columns],)

    with open(model_directory / 'scores.json', mode='w') as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    settings.logger.info(f'scores.json is saved to {model_directory}')
