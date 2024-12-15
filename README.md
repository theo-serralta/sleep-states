# 🛏️ Predicting Sleep States for Kaggle Competition

<p align="center">
  <img src="assets/cmi.png" width="45%" />
</p>

## 🔍 Overview
Sleep is a key factor in overall health, particularly for children, where issues like sleep duration, quality, and variability are often linked to broader health concerns. Traditional methods for studying sleep, such as journals, can introduce biases and fail to capture nuances like the difference between going to bed and falling asleep or waking up and getting up. While polysomnography provides detailed insights, it is expensive and impractical for large-scale or routine use.

Wrist-worn accelerometers offer a promising alternative for sleep research. These lightweight, non-invasive devices enable easy and extensive data collection without disrupting natural sleep patterns. To explore their potential, the [Child Mind Institute](https://childmind.org/), a non-profit specializing in children's mental and learning disorders, launched a competition to develop models predicting sleep and wake events based on accelerometer data.

This project focuses on building an efficient predictive model for determining sleep and wake moments in children. By evaluating and optimizing a range of predictive model architectures, we aim to identify the most effective approach to advance sleep research and improve health outcomes.

## 🎯 Our solution

The target variable is categorical, consisting of three highly imbalanced classes: `onset`, `wakeup`, and `no event`, representing either the type of event or its absence. To address this imbalance, we employed several techniques, including: 
- stratified sampling for training and validation
- weight adjustment
- SMOTE oversampling
- decision threshold tuning

Data preprocessing involved removing accelerometer readings corresponding to times when the watch was removed, as well as excluding individuals with too many missing values. Additionally, we filtered out the last 5% of the data for each series and extracted 1% of the remaining dataset by selecting one out of every 100 rows, maintaining temporal continuity. Feature engineering was then carried out as outlined below. All numeric features were scaled, and the target variable was encoded as integers from 0 to 2 (0: `no event`, 1: `onset`, 2: `wakeup`).

Five models were trained and evaluated using 3-fold cross-validation, with results presented in the results section. Each model was composed of two steps: the first performs binary classification (`event`/`no event`), and the second classifies the events predicted by the first model (`onset`/`wakeup`). This architecture also helps mitigate the effects of data imbalance.

### Feature Engineering 

| Feature                              | Description                                                                 |
|--------------------------------------|-----------------------------------------------------------------------------|
| `time_of_day_sin`                  | Sinusoidal transformation of the hour of the day to capture time-of-day cyclic behavior. |
| `time_of_day_cos`                  | Cosine transformation of the hour of the day, used alongside `time_of_day_sin` to capture cyclic time-of-day patterns. |
| `series_anglez_mean`               | Mean of `anglez` values for each `series_id`.                                |
| `series_anglez_median`             | Median of `anglez` values for each `series_id`.                              |
| `series_anglez_std`                | Standard deviation of `anglez` values for each `series_id`.                  |
| `series_anglez_count`              | Count of `anglez` values for each `series_id`.                               |
| `series_anglez_max`                | Maximum `anglez` value for each `series_id`.                                 |
| `series_anglez_min`                | Minimum `anglez` value for each `series_id`.                                 |
| `series_anglez_sum`                | Sum of `anglez` values for each `series_id`.                                 |
| `series_enmo_mean`                 | Mean of `enmo` values for each `series_id`.                                  |
| `series_enmo_median`               | Median of `enmo` values for each `series_id`.                                |
| `series_enmo_std`                  | Standard deviation of `enmo` values for each `series_id`.                    |
| `series_enmo_count`                | Count of `enmo` values for each `series_id`.                                 |
| `series_enmo_max`                  | Maximum `enmo` value for each `series_id`.                                   |
| `series_enmo_min`                  | Minimum `enmo` value for each `series_id`.                                   |
| `series_enmo_sum`                  | Sum of `enmo` values for each `series_id`.                                   |
| `cumulative_anglez`                | Cumulative sum of `anglez` values for each `series_id`.                      |
| `cumulative_enmo`                  | Cumulative sum of `enmo` values for each `series_id`.                        |
| `cumulative_max_anglez`            | Cumulative maximum of `anglez` values for each `series_id`.                  |
| `cumulative_max_enmo`              | Cumulative maximum of `enmo` values for each `series_id`.                    |
| `anglez_rolling_mean`              | Rolling mean (window size 10) of `anglez` values for each `series_id`.       |
| `anglez_rolling_std`               | Rolling standard deviation (window size 10) of `anglez` values for each `series_id`. |
| `enmo_rolling_mean`                | Rolling mean (window size 10) of `enmo` values for each `series_id`.         |
| `*enmo_rolling_std`                 | Rolling standard deviation (window size 10) of `enmo` values for each `series_id`. |
| `series_anglez_vc_greater_1`       | Count of `anglez` values greater than 1 for each `series_id`.                 |
| `series_anglez_vc_greater_0`       | Count of `anglez` values greater than 0 for each `series_id`.                 |
| `series_enmo_vc_greater_1`         | Count of `enmo` values greater than 1 for each `series_id`.                   |
| `series_enmo_vc_greater_0`         | Count of `enmo` values greater than 0 for each `series_id`.                   |
| `anglez_enmo_product`              | Interaction feature: product of `anglez` and `enmo`.                         |
| `anglez_enmo_ratio`                | Interaction feature: ratio of `anglez` to `enmo` (avoiding division by zero). |
| `anglez_change_point`              | Absolute difference between consecutive `anglez` values for each `series_id`. |
| `enmo_change_point`                | Absolute difference between consecutive `enmo` values for each `series_id`.  |
| `series_anglez_unique_count`       | Number of unique `anglez` values for each `series_id`.                       |
| `series_enmo_unique_count`         | Number of unique `enmo` values for each `series_id`.                         |
| `series_anglez_fft_mean`           | Mean of the FFT magnitudes for `anglez` values for each `series_id`.         |
| `series_anglez_fft_std`            | Standard deviation of the FFT magnitudes for `anglez` values for each `series_id`. |
| `series_anglez_fft_max`            | Maximum of the FFT magnitudes for `anglez` values for each `series_id`.      |
| `series_anglez_fft_min`            | Minimum of the FFT magnitudes for `anglez` values for each `series_id`.      |
| `series_enmo_fft_mean`             | Mean of the FFT magnitudes for `enmo` values for each `series_id`.           |
| `series_enmo_fft_std`              | Standard deviation of the FFT magnitudes for `enmo` values for each `series_id`. |
| `series_enmo_fft_max`              | Maximum of the FFT magnitudes for `enmo` values for each `series_id`.        |
| `series_enmo_fft_min`              | Minimum of the FFT magnitudes for `enmo` values for each `series_id`.        |

## 📊 Results

| Model        | Accuracy (onset / wakeup)      | F1 Score (onset / wakeup)     | Precision (onset / wakeup)      | Recall (onset / wakeup)       | AUC-ROC (onset / wakeup)        |
|--------------|----------------|----------------|----------------|----------------|----------------|
| **Random Forest** |  |  |  |  |  |
| **DNN**  |  |  |  |  |  |
| **CNN** |  |  |  | |  |
| **GRU** |  |  |  |  |  |
| **LSTM** | |  |  |  |  |

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
