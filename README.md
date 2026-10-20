# AirPassengers Time Series Forecasting with ARIMA

## Project Overview

This project explores time-series analysis and forecasting using the classic AirPassengers dataset, which contains monthly airline passenger numbers from 1949 to 1960.

The goal is to understand temporal patterns in airline passenger traffic and develop an **Autoregressive Integrated Moving Average (ARIMA)** model to forecast future passenger numbers.

Using Python, Pandas, NumPy, Matplotlib, and Statsmodels, the project demonstrates data preprocessing, stationarity analysis, trend removal, seasonal decomposition, ARIMA parameter selection, and walk-forward forecasting.

## Objectives

- Explore historical monthly airline passenger data.
- Analyze trends and seasonal patterns in passenger traffic.
- Test time-series stationarity using the Augmented Dickey-Fuller (ADF) test.
- Apply logarithmic transformation to stabilize variance.
- Explore moving-average and exponentially weighted moving-average smoothing.
- Decompose the time series into trend, seasonal, and residual components.
- Search for suitable ARIMA parameters using a grid search.
- Evaluate forecasting performance using Root Mean Squared Error (RMSE).
- Visualize actual and predicted passenger counts.

## Dataset

The AirPassengers dataset contains 144 monthly observations spanning January 1949 to December 1960.

| Feature | Description |
|---|---|
| Month | Month and year of observation |
| #Passengers | Number of airline passengers, recorded in thousands |

The dataset is a univariate time series, meaning that the forecast is based on historical passenger counts rather than multiple independent predictor variables.

## Technologies Used

- **Python** — Programming language
- **Pandas** — Data manipulation and time-series indexing
- **NumPy** — Numerical computations and logarithmic transformations
- **Matplotlib** — Data visualization
- **Statsmodels** — ADF testing, seasonal decomposition, and ARIMA modelling
- **Scikit-learn** — Forecast evaluation using mean squared error

## Project Workflow

### 1. Data Loading and Preprocessing

The dataset is loaded from a CSV file. The `Month` column is converted to datetime format and set as the time-series index.

### 2. Stationarity Analysis

The Augmented Dickey-Fuller test is used to investigate whether the series is stationary. Rolling means and standard deviations are also plotted to examine changes in the series over time.

### 3. Logarithmic Transformation

A natural logarithm transformation is applied to reduce the effect of increasing variance:

`ts_log = np.log(ts)`

### 4. Smoothing and Trend Removal

Two smoothing techniques are explored:

- **Moving Average:** Uses a rolling window of 12 observations to estimate the trend.
- **Exponentially Weighted Moving Average (EWMA):** Assigns greater weight to recent observations.

The smoothed trend estimates are subtracted from the log-transformed series to examine the resulting fluctuations.

### 5. Seasonal Decomposition

The time series is decomposed into:

- Observed series
- Trend component
- Seasonal component
- Residual component

This helps reveal the long-term growth in passenger numbers and recurring annual seasonal patterns.

### 6. ARIMA Grid Search

Different combinations of ARIMA parameters are evaluated:

- `p = [1, 2]`
- `d = [0, 1]`
- `q = [1, 2]`

This produces eight candidate ARIMA configurations. A chronological training and test split is used, and walk-forward validation evaluates one-step-ahead forecasts.

The model with the lowest RMSE on the evaluation set is selected from the tested configurations.

### 7. Forecast Evaluation

Forecasting performance is evaluated using Root Mean Squared Error (RMSE). Actual and predicted passenger counts are plotted after converting the log-scale forecasts back to the original scale.

## Installation

Clone the repository:

```bash
git clone https://github.com/Vedline2547/Arima.git
cd Arima
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install pandas numpy matplotlib statsmodels scikit-learn
```

Ensure that `AirPassengers.csv` is available in the project directory, then run:

```bash
python main.py
```

Update the CSV file path in `main.py` if the dataset is stored elsewhere.

## Evaluation Metric

**Root Mean Squared Error (RMSE)** measures the typical magnitude of forecast errors, with larger errors receiving greater weight.

A lower RMSE indicates better predictive performance when comparing models evaluated on the same observations and scale.

## Key Learning Outcomes

Through this project, I explored:

- Time-series preprocessing and datetime indexing.
- Stationarity testing and interpretation of ADF results.
- Logarithmic transformations and smoothing techniques.
- Trend and seasonal decomposition.
- ARIMA parameter selection through grid search.
- Walk-forward validation for sequential forecasting.
- Forecast evaluation and visualization.

## Future Improvements

- Compare ARIMA with Seasonal ARIMA (SARIMA) to model annual seasonality explicitly.
- Use seasonal differencing and compare stationarity results.
- Evaluate models using a separate validation set and an untouched final test set.
- Compare ARIMA against baseline forecasting methods.
- Investigate forecast intervals and additional evaluation metrics.
- Build an interactive forecasting application.

## Conclusion

This project provides a practical introduction to statistical time-series forecasting using historical airline passenger data. It combines exploratory analysis, stationarity testing, smoothing, decomposition, and ARIMA modelling into a reproducible forecasting workflow.

The project also provides a foundation for exploring seasonal forecasting methods and more advanced time-series models.

## Author

**Vedline Ochieng**

- GitHub: [Vedline2547](https://github.com/Vedline2547)
- LinkedIn: [Vedline Ochieng](https://www.linkedin.com/in/vedline-ochieng-025146290/)

*Civil Engineering Student | Machine Learning Enthusiast | Python Developer | Aspiring AI Engineer*

---

If you find this project useful, feel free to explore the repository and follow my journey in machine learning and AI.
