"""
Add time-based, statistical, and interaction features to a dataset.

This script processes a dataset containing 'timestamp', 'anglez', 'enmo', and
'series_id' columns, and adds various features such as time-based features
(extracted from the 'timestamp' column), aggregate statistical features for 
'anglez' and 'enmo', rolling window statistics, interaction features, 
and value counts. 

Usage:
    python script.py --input path_to_input.csv --output path_to_output.csv

Arguments:
    --input, -i   Path to the input CSV file.
    --output, -o  Path to the output CSV file.
"""

__authors__ = "Nadezhda Zhukova, Giulia Di Gennaro"
__contact__ = "nadiajuckova@gmail.com"
__copyright__ = "MIT"
__date__ = "2024-10-14"
__version__ = "1.0.0"

import numpy as np
import pandas as pd
import argparse


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based, statistical, and interaction features to the input dataset.

    Args:
        df (pd.DataFrame): Input DataFrame with 'timestamp', 'anglez', 'enmo',
                           and 'series_id' columns.

    Returns:
        pd.DataFrame: The DataFrame with added features.
    """
    
    # Convert timestamp to datetime and extract time-based features
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['time_of_day_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['time_of_day_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df = df.drop(columns=['timestamp', 'hour_of_day'])

    # Group by 'series_id'
    group_anglez = df.groupby('series_id')['anglez']
    group_enmo = df.groupby('series_id')['enmo']

    # Apply aggregation functions
    anglez_stats = group_anglez.agg(['mean', 'median', 'std', 
                                     'count', 'max', 'min', 'sum'])
    enmo_stats = group_enmo.agg(['mean', 'median', 'std', 
                                 'count', 'max', 'min', 'sum'])

    # Flatten the MultiIndex after aggregation
    anglez_stats.columns = [f'series_anglez_{col}' 
                            for col in anglez_stats.columns]
    enmo_stats.columns = [f'series_enmo_{col}' 
                          for col in enmo_stats.columns]

    # Merge the aggregated stats back into the original DataFrame
    df = df.merge(anglez_stats, on='series_id', how='left')
    df = df.merge(enmo_stats, on='series_id', how='left')

    # Cumulative and rolling window features
    df['cumulative_anglez'] = group_anglez.cumsum()
    df['cumulative_enmo'] = group_enmo.cumsum()
    df['cumulative_max_anglez'] = group_anglez.cummax()
    df['cumulative_max_enmo'] = group_enmo.cummax()

    window = 10
    df['anglez_rolling_mean'] = (
        group_anglez.rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df['anglez_rolling_std'] = (
        group_anglez.rolling(window=window, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
    )
    df['enmo_rolling_mean'] = (
        group_enmo.rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df['enmo_rolling_std'] = (
        group_enmo.rolling(window=window, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
    )

    # Value counts using thresholds
    df['series_anglez_vc_greater_1'] = (
        (df['anglez'] > 1).groupby(df['series_id']).transform('sum')
    )
    df['series_anglez_vc_greater_0'] = (
        (df['anglez'] > 0).groupby(df['series_id']).transform('sum')
    )
    df['series_enmo_vc_greater_1'] = (
        (df['enmo'] > 1).groupby(df['series_id']).transform('sum')
    )
    df['series_enmo_vc_greater_0'] = (
        (df['enmo'] > 0).groupby(df['series_id']).transform('sum')
    )

    # Interaction features between anglez and enmo
    df['anglez_enmo_product'] = df['anglez'] * df['enmo']
    df['anglez_enmo_ratio'] = df['anglez'] / (df['enmo'] + 1e-6)

    # Change point features
    df['anglez_change_point'] = group_anglez.diff().abs()
    df['enmo_change_point'] = group_enmo.diff().abs()

    # Unique value counts
    df['series_anglez_unique_count'] = group_anglez.transform('nunique')
    df['series_enmo_unique_count'] = group_enmo.transform('nunique')

    return df


def parse_args():
    """
    Parse command-line arguments for input and output file paths.

    Returns:
        argparse.Namespace: Parsed arguments with input and output file paths.
    """
    parser = argparse.ArgumentParser(
        description="Add features to the input dataset and save the result."
    )
    parser.add_argument(
        '--input', '-i', type=str, required=True, 
        help="Path to the input CSV file."
    )
    parser.add_argument(
        '--output', '-o', type=str, required=True, 
        help="Path to the output CSV file."
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse the input and output file paths
    args = parse_args()

    # Load the filtered train series data
    df = pd.read_csv(args.input)

    # Add features to the dataset
    df = add_features(df)

    # Drop rows with infinite and missing values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Save the featured dataset
    df.to_csv(args.output, index=False)
