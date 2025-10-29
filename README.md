# Future-traffic-prediction-using-LSTM

🚦 Traffic Volume Prediction

This repository contains a Google Colab notebook that demonstrates traffic analysis and prediction using Python, Data Visualization, and Machine Learning techniques.

The notebook explores real-world traffic patterns, applies data preprocessing, builds models, and visualizes results to gain insights into how traffic volume changes over time.

DATASET: Uploaded Metro_Interstate_Traffic_Volume.csv


🛠️ Features

Data preprocessing and cleaning

Exploratory Data Analysis (EDA)

Visualization of trends (line graphs, correlations, etc.)

Machine Learning model for traffic prediction

Interactive and reproducible Google Colab workflow

📊 Example Visualization

Traffic Volume vs Time (Line Graph)

🚀 How to Run

Open the notebook in Google Colab.

Upload the dataset when prompted.

Run all cells in order.

View visualizations and predictions directly in the notebook.

📦 Requirements

All dependencies are included in the Colab environment.
For running locally, install:

pip install pandas numpy matplotlib seaborn scikit-learn

📌 Notes

This repository contains the complete Colab notebook only.

No separate models or files are uploaded.

Designed for learning and experimenting with traffic prediction using ML and visualization.

------------------


## **for complete expalantion of project see below**


## **Project Overview**

This project predicts **hourly traffic volume** on a metro/interstate highway using a **multivariate LSTM (Long Short-Term Memory)** neural network. The model incorporates weather conditions, time-of-day patterns, and day-of-week effects to forecast traffic for the next 24 hours.

The dataset is from [Metro Interstate Traffic Volume Dataset](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume).


## **Key Features**

* Predict traffic volume for the next 24 hours
* Weather-aware and time-aware predictions
* Multivariate LSTM using minimal but important features
* Temporal train/test split to prevent data leakage
* Visualizations of model predictions and training history
* Forecast summary including expected rush hours


## **Dataset**

The dataset contains **hourly traffic and weather data**.

**Columns:**

* `holiday` – Indicates if the day is a holiday (`None` if not)
* `temp` – Temperature in Kelvin (converted to Celsius during preprocessing)
* `rain_1h` – Rain volume in last hour (mm)
* `snow_1h` – Snow volume in last hour (mm)
* `clouds_all` – Cloud coverage (%)
* `weather_main` – Main weather category
* `weather_description` – Detailed weather description
* `date_time` – Timestamp of observation
* `traffic_volume` – Traffic volume (vehicles/hour)

**Sample Data:**

| holiday | temp   | rain_1h | snow_1h | clouds_all | weather_main | weather_description | date_time        | traffic_volume |
| ------- | ------ | ------- | ------- | ---------- | ------------ | ------------------- | ---------------- | -------------- |
| None    | 288.28 | 0.0     | 0.0     | 40         | Clouds       | scattered clouds    | 2012-10-02 09:00 | 5545           |

> **Note:** Some preprocessing is applied before feeding the data to the LSTM.

## **Feature Engineering**

Minimal features are selected to reduce complexity while capturing essential traffic patterns:

| Feature       | Description                              |
| ------------- | ---------------------------------------- |
| `temp_c`      | Temperature in Celsius                   |
| `rain_1h`     | Rain in last hour (mm)                   |
| `hour`        | Hour of the day (0–23)                   |
| `day_of_week` | Day of the week (0=Monday, 6=Sunday)     |
| `is_weekend`  | Weekend indicator (1 if Sat/Sun, else 0) |

Target variable: `traffic_volume`

---

## **Data Preprocessing**

1. Convert `date_time` to datetime and set as index
2. Extract time features: hour, day_of_week, weekend
3. Encode holidays (`is_holiday`) and weather (`weather_main`)
4. Fill missing values (default 0)
5. Scale features and target with `MinMaxScaler`
6. Create **sequences of 24 hours** for LSTM input:

   * X → previous 24 hours of features
   * y → traffic volume of the current hour
7. Temporal train/test split (last 20% for testing)


## **Model Architecture**

A **Sequential LSTM model** is used:

1. LSTM layer with 32 units (`return_sequences=True`)
2. Dropout (0.1)
3. LSTM layer with 16 units (`return_sequences=False`)
4. Dropout (0.1)
5. Dense layer with 8 units (ReLU activation)
6. Dense output layer with 1 unit

**Training Details:**

* Loss: Mean Squared Error (MSE)
* Optimizer: Adam
* Metrics: Mean Absolute Error (MAE)
* Epochs: 30, Batch Size: 32
* Validation: Temporal split
* Shuffle: False (preserve sequence order)


## **Model Evaluation**

Metrics computed on the test set:

| Metric | Value                |
| ------ | -------------------- |
| MSE    | 479,340.55           |
| RMSE   | 692.34 vehicles/hour |
| MAE    | 452.18 vehicles/hour |
| R²     | 0.8762               |

✅ Positive R² indicates good predictive performance.


## **24-Hour Forecast**

* Predicts traffic volume for the next 24 hours using the trained model
* Uses **physically plausible future features**:

  * Temperature: sinusoidal pattern
  * Rain: sampled from historical data for the same hour & day
  * Hour, weekday, weekend: deterministic
* Traffic status legend:

  * 🟢 LOW (< 800)
  * 🟡 MODERATE (800–2500)
  * 🟠 HIGH (2500–4000)
  * 🔴 PEAK (> 4000)

**Example Forecast Output (next 24 hours):**

```
+01h | Mon 01 00:00 | 1356 vehicles | 🟡 MODERATE
+02h | Mon 01 01:00 |  647 vehicles | 🟢 LOW
...
+08h | Mon 01 07:00 | 6333 vehicles | 🔴 PEAK
...
+24h | Mon 01 23:00 |  770 vehicles | 🟢 LOW
```


## **Visualizations**

* Training history: Loss & validation loss
* Actual vs Predicted scatter plot
* Time series plot: One-week comparison of predictions and actual traffic
* 24-hour forecast plot: Hourly traffic trend

Example plot:

```python
plt.figure(figsize=(12, 6))
plt.plot(hours, future_traffic, marker='o', linewidth=2, markersize=6)
plt.title('24-Hour Traffic Forecast')
plt.xlabel('Time (Hours)')
plt.ylabel('Traffic Volume (vehicles/hour)')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## **Future Improvements**

* Include more weather features (wind, cloudiness)
* Test deeper LSTM or GRU networks
* Hyperparameter tuning for better accuracy
* Incorporate real-time traffic and event data

---

