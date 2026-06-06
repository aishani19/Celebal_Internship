# Celebal Assignments [DATA SCIENCE]

## Details
- Name: Aishani Billore
- College: SKIT
- Domain: Data Science

**##WEEK 1 ASSIGNMENTS**
## Assignment Overview
This assignment introduces the fundamental concepts required for Machine Learning and Data Science. It combines programming, mathematics, and data analysis concepts through practical exercises.

## Topics Covered

### 1. Python Fundamentals
Practiced conditional statements, loops, functions, exception handling, lambda functions, and basic data structures such as lists, sets, and dictionaries.

### 2. NumPy
Learned how to create and manipulate arrays, perform indexing and slicing, and carry out mathematical and matrix operations efficiently.

### 3. Pandas
Worked with Series and DataFrames, data filtering, grouping, handling missing values, and basic data analysis operations.

### 4. Linear Algebra
Explored vectors, matrices, matrix multiplication, eigenvalues, eigenvectors, and Singular Value Decomposition (SVD), which are important mathematical foundations for Machine Learning.

### 5. Statistics
Calculated descriptive statistics such as mean, median, standard deviation, and IQR. Also performed hypothesis testing and analyzed data distributions using visualizations.

### 6. Probability
Studied basic probability concepts, conditional probability, Bayes' Theorem, and their applications in Machine Learning.

## Libraries Used
- Python
- NumPy
- Pandas
- Matplotlib
- SciPy

## Key Learning Outcomes
- Developed Python programming skills for data analysis.
- Learned to manipulate and analyze datasets using NumPy and Pandas.
- Understood important linear algebra concepts used in Machine Learning.
- Applied statistical techniques to summarize and interpret data.
- Explored probability concepts that form the basis of predictive models.

## Conclusion
This assignment provided hands-on experience with the core concepts required for Machine Learning. It strengthened my understanding of programming, mathematics, statistics, and data analysis, which are essential for building and understanding machine learning models.

**##WEEK2 ASSIGNMENT**
# Tesla Deliveries Analysis: An End-to-End Machine Learning Project

## Project Overview

This project involved a comprehensive analysis of a synthetic Tesla deliveries dataset, covering data loading, exploratory data analysis (EDA), feature engineering, regression modeling, and time series forecasting. The primary goal was to predict `Estimated_Deliveries` using various machine learning techniques and to understand the characteristics of the provided dataset.

## Dataset

The dataset `tesla_deliveries_dataset_2015_2025.csv` contains simulated Tesla delivery data from 2015 to 2025. Key features include:
- `Year`, `Month`, `Region`, `Model`
- `Estimated_Deliveries` (Target Variable)
- `Production_Units` (a strong predictor indicating supply)
- `Avg_Price_USD`, `Battery_Capacity_kWh`, `Range_km`, `Charging_Stations`

**Key Insight:** Through EDA and model performance, the dataset was identified as **synthetic** due to highly consistent monthly totals and strong, often artificial, correlations between certain features.

## Methodology

### 1. Data Loading and Initial Exploration
- Loaded data into a pandas DataFrame (`df_raw`).
- Inspected data types, missing values, and numerical summaries (`df_raw.info()`, `df_raw.describe()`).

### 2. Exploratory Data Analysis (EDA)
- **Distribution Analysis:** Visualized the distribution of `Estimated_Deliveries` using histograms and box plots across different regions.
- **Time Series Visualization:** Plotted total monthly deliveries to observe trends and seasonality, revealing unusually consistent monthly totals (around 195k).
- **Model & Region Analysis:** Explored mean deliveries by `Model` and `Region`.
- **Correlation Analysis:** Generated a correlation heatmap to understand relationships between numerical features and the target variable.

### 3. Data Preprocessing & Feature Engineering
- **Leakage Audit:** Identified and dropped `CO2_Saved_tons` due to data leakage (calculated from the target). `Production_Units` was kept as a 'supply driver' feature.
- **Categorical Encoding:** Used `LabelEncoder` for `Region` and `Model` to create `Region_enc` and `Model_enc` numerical features.
- **Date and Time Features:** Created a `Date` column and extracted `Quarter`. The data was sorted chronologically per `Region` and `Model`.
- **Lag & Rolling Features:** Implemented a custom function to create lagged features (`Lag1`, `Lag2`, `Lag3`) and rolling statistics (`Roll_mean3`, `Roll_std3`, `Roll_mean6`, `Roll_std6`, `MoM_pct_change`) per `Region-Model` group, carefully avoiding data leakage by using `.shift(1)`.
- **Price Efficiency:** Calculated `Price_per_km`.
- **Missing Value Handling:** Dropped rows with NaNs introduced by lag features.

### 4. Train/Test Split (Chronological)
- Performed an 80/20 chronological split on the processed data (`df`), ensuring the test set comprised the latest 20% of records (`train_df`, `test_df`).
- Scaled numerical features using `StandardScaler`, fitting only on training data to prevent leakage (`X_train_sc`, `X_test_sc`).

### 5. Regression Modeling

**Models Used:**
- Linear Regression
- Ridge Regression
- Lasso Regression

**Hyperparameter Tuning:**
- Used `GridSearchCV` with `TimeSeriesSplit` (5 folds) for robust cross-validation, respecting the chronological order of data.
- Tuned `alpha` for Ridge and Lasso models.

**Performance (on Test Set):**
| Model             | R²     | RMSE    | MAE     |
|:------------------|:-------|:--------|:--------|
| Lasso (tuned)     | 0.9865 | 427.6   | 353.9   |
| Linear Regression | 0.9864 | 428.9   | 355.4   |
| Ridge (tuned)     | 0.9864 | 428.9   | 355.4   |

**Findings:** Regression models performed exceptionally well, achieving R² scores close to 0.99. This high performance is largely attributed to the strong predictive power of the `Production_Units` feature and the synthetic nature of the dataset. Careful regularization and `TimeSeriesSplit` ensured robustness against overfitting.

### 6. Time Series Forecasting

**Models Used (on monthly aggregated data):**
- Naive Seasonal Forecast
- Holt-Winters Exponential Smoothing
- SARIMA (Seasonal Autoregressive Integrated Moving Average)

**Analysis Steps:**
- Aggregated `Estimated_Deliveries` to monthly totals (`ts`).
- Performed **Augmented Dickey-Fuller (ADF) Test** to check for stationarity (series found to be stationary).
- Conducted **Time Series Decomposition** to analyze trend, seasonality, and residuals, confirming strong seasonality and a flat trend.
- Examined **Rolling Statistics** (mean and standard deviation) to further understand series stability.
- Split `ts` into `ts_train` and `ts_test` (80/20 chronological).

**Performance (on Test Set):**
| Model           | R²      | RMSE    | MAE     |
|:----------------|:--------|:--------|:--------|
| Naive Seasonal  | -0.407  | 14968   | 11809   |
| Holt-Winters    | -0.423  | 15050   | 11012   |
| SARIMA          | -2.498  | 23597   | 16707   |

**Findings:** All time series models yielded **negative R² values**, indicating they performed worse than simply predicting the mean. This is a crucial finding, reinforcing the dataset's synthetic and highly consistent nature. The models struggled to find patterns that would significantly improve upon a simple average guess, as the data lacked the typical variability found in real-world time series.

**4-Month Future Forecasts (2026):**
| Model          | Jan 2026 | Feb 2026 | Mar 2026 | Apr 2026 |
|:---------------|:---------|:---------|:---------|:---------|
| Holt-Winters   | 186,512  | 189,154  | 196,538  | 185,161  |
| SARIMA         | 220,251  | 184,567  | 216,836  | 219,418  |
| Naive Seasonal | 201,440  | 201,440  | 201,440  | 201,440  |

## Conclusion

The project successfully demonstrated an end-to-end ML pipeline for predicting Tesla deliveries. Regression models achieved excellent performance, largely due to the `Production_Units` feature. However, the time series models highlighted the dataset's synthetic nature by struggling to provide meaningful forecasts beyond a simple average, resulting in negative R² scores. This underscores the importance of understanding data characteristics before applying complex modeling techniques.
