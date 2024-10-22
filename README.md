# Child Mind Institute - Detect Sleep States - Kaggle Competition

Sleep is vital in regulating mood, emotions, and behavior across all age groups, especially in children. Accurately detecting sleep and wakefulness using wrist-worn accelerometer data enables researchers to better understand sleep patterns and disturbances in children. These predictions could have significant implications, particularly for children and youth experiencing mood and behavioral difficulties.

## Download the data

The data is available on the [Kaggle competition page](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/data).

## Prepare the train and test data

To prepare the data, we need to incorporate the following steps:
1. Load the data
2. Preprocess the data
3. Split the data into train and test sets
4. Add features

To preprocess the data, use the following code:

```python
python src/preprocessing.py
```

This script produces the file `data/filtered_train_series.csv` containing the preprocessed train data with only 1% of the data.

To split the data into train and test sets, use the following code:

```python
python src/split_data.py data/filtered_train_series.csv              \
                         data/filtered_train_series.csv              \
                         data/filtered_test_series.csv               \
                         [--test_size 0.3] [--random_state 1]
```

This script produces the files `data/filtered_train_series.csv` and `data/filtered_test_series.csv` containing the train and test data, respectively.

To add features, use the following code:

```python 
python src/features.py --input data/filtered_train_series.csv --output data/featured_train_series.csv
python src/features.py --input data/filtered_test_series.csv --output data/featured_test_series.csv
```
