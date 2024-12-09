# 🛏️ Predicting Sleep States for Kaggle Competition

![cmi](assets/cmi.png)

## 🔍 Overview
Sleep is a key factor in overall health, particularly for children, where issues like sleep duration, quality, and variability are often linked to broader health concerns. Traditional methods for studying sleep, such as journals, can introduce biases and fail to capture nuances like the difference between going to bed and falling asleep or waking up and getting up. While polysomnography provides detailed insights, it is expensive and impractical for large-scale or routine use.

Wrist-worn accelerometers offer a promising alternative for sleep research. These lightweight, non-invasive devices enable easy and extensive data collection without disrupting natural sleep patterns. To explore their potential, the [Child Mind Institute](https://childmind.org/), a non-profit specializing in children's mental and learning disorders, launched a competition to develop models predicting sleep and wake events based on accelerometer data.

This project focuses on building a highly accurate and efficient predictive model for determining sleep and wake moments in children. By evaluating and optimizing a range of predictive model architectures, we aim to identify the most effective approach to advance sleep research and improve health outcomes.

## 🎯 Our solution

## 🖥️ Run the code

### Download the data

The data is available on the [Kaggle competition page](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/data).

### Prepare the train and test data

To prepare the data, we need to incorporate the following steps:
1. Load the data
2. Preprocess the data
3. Split the data into train and test sets
4. Add features

To preprocess the data, use the following code:

```python
python src/process_train_data.py
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
python src/add_features.py --input data/filtered_train_series.csv --output data/featured_train_series.csv
python src/add_features.py --input data/filtered_test_series.csv --output data/featured_test_series.csv
```
