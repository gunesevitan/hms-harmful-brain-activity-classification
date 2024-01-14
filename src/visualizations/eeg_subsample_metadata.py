import sys
import pandas as pd

sys.path.append('..')
import settings
import visualization


if __name__ == '__main__':

    metadata_directory = settings.DATA / 'eeg_subsample'
    visualization_directory = settings.EDA / 'eeg_subsample_metadata'
    visualization_directory.mkdir(parents=True, exist_ok=True)

    df_eeg_subsample_metadata = pd.read_csv(metadata_directory / 'metadata.csv')

    for column in df_eeg_subsample_metadata.columns.tolist()[2:]:
        visualization.visualize_continuous_column_distribution(
            df=df_eeg_subsample_metadata,
            column=column,
            title=f'EEG Subsample Metadata {column}',
            path=visualization_directory / f'eeg_subsample_metadata_{column}.png'
        )
        settings.logger.info(f'eeg_subsample_metadata_{column}.png is saved to {visualization_directory}')
