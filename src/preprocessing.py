"""This script aims to create the sample training data from the raw data."""

import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import tqdm


def delete_NA_from_train_series(train_data, train_events):
    """Delete NAs from the train series.

    Args:
        train_data (pandas.DataFrame): training data
        train_events (pandas.DataFrame): training events

    Returns:
        pandas.DataFrame: training dataframe without NAs
    """
    indexes_NA_train_series = []

    # Iterate through the train_events dataframe
    i = 0
    while i < len(train_events) - 1:
        current_event = train_events.iloc[i]
        next_event = train_events.iloc[i + 1]
        series_id = next_event['series_id']

        # Initialize j to find the next valid timestamp
        j = i + 1
        # If the next event timestamp is NA
        if pd.isna(next_event['timestamp']):
            # Find the next event with a non-NA timestamp, preventing index out-of-bounds
            while j < len(train_events) and pd.isna(train_events.iloc[j]['timestamp']):
                j += 1

            # If we've gone beyond the length of the DataFrame, break
            if j >= len(train_events):
                break

            # Grab the next valid event
            next_step = train_events.iloc[j]

            # Ensure the series_id is the same before and after the NA block
            if series_id == current_event['series_id'] and series_id == next_step['series_id']:
                # Calculate the time window
                first_time = current_event['timestamp']
                next_time = next_step['timestamp']

                # Find indices in train_data between first_time and next_time
                indexes = train_data[
                    (train_data['series_id'] == series_id) &
                    (train_data['timestamp'] > first_time) &
                    (train_data['timestamp'] < next_time)
                ].index
                indexes_NA_train_series.extend(indexes.tolist())

        # Move to the next valid event
        i = j

    # Drop the collected indices from train_data
    return train_data.drop(set(indexes_NA_train_series))


def filter_last_percents(train_series, max_steps):
    """Filter the last 5% of data for each participant.

    Args:
        train_series (pandas.DataFrame): training data
        max_steps (pandas.DataFrame): maximum step for each series id

    Returns:
        pandas.DataFrame: training data without the last 5%
    """
    # Add the min step from which to drop rows
    max_steps["min_step"] = (max_steps["step"] - max_steps["step"] * 0.05).round()

    # Initialize an empty set to store indices to filter out
    indices_to_drop = set()
    # Step through each row in max_steps DataFrame
    for _, row in max_steps.iterrows():
        series_id = row['series_id']
        min_step = row['min_step']
        step = row['step']
        # Find indices in train_series that match the condition
        condition = (train_series['series_id'] == series_id) & (train_series['step'] >= min_step) & (train_series['step'] <= step)
        # Add indices to the set of indices to drop
        indices_to_drop.update(train_series[condition].index)

    # Drop the identified indices from train_series
    filtered_train_series = train_series.drop(index=indices_to_drop)
    return filtered_train_series


def drop_nulls(df):
    """Drop all participants who have too many NAs.

    Args:
        df (Pandas.DataFrame): training data

    Returns:
        Pandas.DataFrame: The training data without the participants in question
    """
    NAN_SERIES_IDS = [
    '0f9e60a8e56d',
    '390b487231ce',
    '2fc653ca75c7',
    'c7b1283bb7eb',
    '89c7daa72eee',
    'e11b9d69f856',
    'c5d08fc3e040',
    'a3e59c2ce3f6',
    ]
    df = df[~df['series_id'].isin(NAN_SERIES_IDS)]
    return df


def process_and_filter_series(parquet_file_path, events_file_path, 
                              output_file_path, train_events, max_steps,
                              drop_percentage=0.99
                              ):
    """Process and filter the training data, then save in csv file.

    Args:
        parquet_file_path (str): File path to the training data
        events_file_path (str): File path to the events data
        output_file_path (str): File path to the processed training data
        train_events (Pandas.DataFrame): events data
        max_steps (Pandas.DataFrame): maximum step for each series id
        drop_percentage (float, optional): Percent of data without event to drop.
    """
    # Load the events dataframe
    events_df = pd.read_csv(events_file_path)

    # Prepare an empty DataFrame to store the final results
    final_df = pd.DataFrame()

    # Process the parquet file in chunks
    parquet_file = pq.ParquetFile(parquet_file_path)
    
    for i, batch in tqdm.tqdm(enumerate(parquet_file.iter_batches(batch_size=500_000))):
        # Convert the batch to a pandas DataFrame
        df_batch = batch.to_pandas()
        
        df_batch = filter_last_percents(df_batch, max_steps)
        df_batch = drop_nulls(df_batch)
        df_batch = delete_NA_from_train_series(df_batch, train_events)

        # Get a list of unique series_ids in the current batch
        series_ids = df_batch['series_id'].unique()

        # Process each series_id individually
        for series_id in series_ids:
            # Filter the data for this particular series_id
            series_data = df_batch[df_batch['series_id'] == series_id]

            # Get the corresponding events for this series_id
            events_for_series = events_df[events_df['series_id'] == series_id]

            # Identify the timestamps to retain (timestamps that are labeled as events)
            labeled_timestamps = events_for_series['timestamp'].tolist()

            # Filter out rows that have labeled events to avoid dropping them
            non_labeled_data = series_data[~series_data['timestamp'].isin(labeled_timestamps)]

            # Randomly sample the data, keeping only (1 - drop_percentage) proportion of non-labeled data
            sample_size = int(len(non_labeled_data) * (1 - drop_percentage))
            sampled_data = non_labeled_data.sample(n=sample_size, random_state=42)

            # Combine the sampled data with the labeled data (which we do not want to drop)
            final_series_data = pd.concat([sampled_data, series_data[series_data['timestamp'].isin(labeled_timestamps)]])

            # Merge with events to add labels (onset, wakeup, NaN for non-events)
            final_series_data = pd.merge(final_series_data, 
                                         events_for_series[['timestamp', 'event']], 
                                         on='timestamp', 
                                         how='left')

            # Add the final series data to the result DataFrame
            final_df = pd.concat([final_df, final_series_data], ignore_index=True)

    # Save the final result to a CSV file
    if not final_df.empty:
        print(len(final_df))
        final_df.to_csv(output_file_path, header=not pd.io.common.file_exists(output_file_path), index=False)


if __name__ == "__main__":
    train_events = pd.read_csv('data/train_events.csv')
    max_steps = pd.read_csv("data/max_steps.csv")
    process_and_filter_series('data/train_series.parquet', 
                            'data/train_events.csv', 
                            'data/filtered_train_series.csv',
                            train_events, max_steps)

