"""
Split a dataset into training and test sets based on series_id and event.

This script processes a dataset that contains 'series_id', 'step', and 'event' 
columns, and splits it into training and test sets. The split is stratified 
based on the 'event' column, ensuring proportional representation of events 
across the sets. The script outputs two CSV files for training and testing.

Usage:
    python split_data.py input_file.csv output_train_file.csv    \
                            output_test_file.csv                 \
                            [--test_size 0.3] [--random_state 1]

Arguments:
    input_file        Path to the input CSV file with the dataset.
    output_train_file Path to save the training set CSV.
    output_test_file  Path to save the test set CSV.
    --test_size       Proportion of the dataset to include in the test split.
    --random_state    Random state for reproducibility. Default is 1.
"""

__authors__ = "Nadezhda Zhukova"
__contact__ = "nadiajuckova@gmail.com"
__copyright__ = "MIT"
__date__ = "2024-10-22"
__version__ = "1.0.0"


import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(input_file, output_train_file, output_test_file, 
                  test_size=0.3, random_state=1):
    """
    Split a dataset into training and test sets.
    
    Parameters:
    - input_file (str): Path to the input CSV file.
    - output_train_file (str): Path where the training set will be saved.
    - output_test_file (str): Path where the test set will be saved.
    - test_size (float): Proportion of the dataset to include in the test.
    - random_state (int): Random seed for reproducibility.
    
    Returns:
    None: Saves the resulting train and test datasets as CSV files.
    """
    # Load dataset
    df = pd.read_csv(input_file)
    df = df.sort_values(by=['series_id', 'step']).reset_index(drop=True)

    # Group by 'series_id' and keep complete series
    grouped = df.groupby('series_id')

    # Prepare a list of unique series and their corresponding events
    series_event = grouped['event'].apply(lambda x: x.iloc[0])

    # Stratified split based on the 'event' column
    train_series, test_series = train_test_split(series_event.index, 
                                                 stratify=series_event, 
                                                 test_size=test_size, 
                                                 random_state=random_state)

    # Create the two DataFrames by selecting series
    df_train = df[df['series_id'].isin(train_series)]
    df_test = df[df['series_id'].isin(test_series)]

    # Save the DataFrames
    df_train.to_csv(output_train_file, index=False)
    df_test.to_csv(output_test_file, index=False)


def main():

    # Create an argument parser
    parser = argparse.ArgumentParser(
        description="Split a dataset into training and test sets."
    )

    # Adding arguments
    parser.add_argument('input_file', 
                        type=str, 
                        help="Path to the input CSV file with the dataset.")
    parser.add_argument('output_train_file', 
                        type=str, 
                        help="Path to save the training set CSV.")
    parser.add_argument('output_test_file', 
                        type=str, 
                        help="Path to save the test set CSV.")
    parser.add_argument('--test_size', 
                        type=float, 
                        default=0.3, 
                        help=("Proportion of the dataset to include in the "
                              "test split. Default is 0.3."))
    parser.add_argument('--random_state', 
                        type=int, 
                        default=1, 
                        help="Random state for reproducibility. Default is 1.")

    args = parser.parse_args()

    # Call the split function with parsed arguments
    split_dataset(args.input_file, 
                  args.output_train_file, 
                  args.output_test_file, 
                  args.test_size, 
                  args.random_state)


if __name__ == '__main__':
    main()
