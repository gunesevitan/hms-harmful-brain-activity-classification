import sys
import pandas as pd

sys.path.append('..')
import settings
import visualization


if __name__ == '__main__':

    metadata_directory = settings.DATA / 'metadata'
    visualization_directory = settings.EDA / 'metadata'
    visualization_directory.mkdir(parents=True, exist_ok=True)

    df_eeg_metadata = pd.read_csv(metadata_directory / 'eeg_metadata.csv')

    for column in ['row_count', 'column_count']:
        visualization.visualize_continuous_column_distribution(
            df=df_eeg_metadata,
            column=column,
            title=f'EEG Metadata',
            path=visualization_directory / f'eeg_metadata_{column}.png'
        )
        settings.logger.info(f'eeg_metadata_{column}.pn is saved to {visualization_directory}')
