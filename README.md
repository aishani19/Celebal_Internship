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

## Week 3
# Customer Intelligence System — Unsupervised Learning on Country Data

## Project Overview
This project develops a **Customer Intelligence System** using unsupervised learning techniques to segment countries based on various socio-economic and health indicators. The primary goal is to identify countries that are most in need of aid, enabling strategic decision-making for resource allocation.

## Dataset
The analysis is performed on the `Country-data.csv` dataset, which contains 9 socio-economic and health indicators for 167 countries. Key features include: `child_mort` (child mortality), `exports`, `health` (total health spending), `imports`, `income`, `inflation`, `life_expec` (life expectancy), `total_fer` (total fertility rate), and `gdpp` (GDP per capita).

## Methodology and Analysis
The project follows a standard data science workflow:

### 1. Importing Libraries
All necessary Python libraries for data manipulation, visualization, clustering, and classification were imported, including `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, `xgboost`, and `lightgbm`.

### 2. Data Loading
The `Country-data.csv` file was loaded into a pandas DataFrame. Initial inspection confirmed its shape (167 rows, 10 columns) and structure.

### 3. Data Cleaning
-   Whitespace was stripped from column names.
-   Duplicate rows were checked for and removed (none found).
-   All feature columns were converted to numeric types, coercing errors to NaN.
-   Missing values (NaNs) in feature columns were imputed using the median of their respective columns.

### 4. Exploratory Data Analysis (EDA)
-   **Statistical Summary:** Basic descriptive statistics (mean, std, min, max, quartiles) were generated for all numerical features.
-   **Feature Distributions:** Histograms revealed that most features, especially `child_mort`, `income`, and `gdpp`, are right-skewed.
-   **Correlation Heatmap:** Visualized relationships between features. Notable strong correlations include: `child_mort` and `life_expec` (negative, -0.89), `income` and `gdpp` (positive, 0.90).
-   **Boxplots:** Used to identify outliers. `income` and `gdpp` showed extreme outliers, likely representing highly developed nations.

### 5. Feature Scaling
`StandardScaler` was applied to the numerical features to ensure they have a mean of 0 and a standard deviation of 1. This prevents features with larger ranges from dominating distance-based algorithms.

### 6. Clustering: Determining Optimal K
-   **Elbow Method:** Used inertia values for K-Means with K ranging from 2 to 10. An 'elbow' was observed at K=3.
-   **Silhouette Score:** Calculated for K=2 to 10. While not the highest, K=3 showed a reasonable balance with the elbow method.
-   **Conclusion:** Optimal number of clusters chosen as K=3.

### 7. Clustering: K-Means & K-Medoids
-   **K-Means Clustering:** A K-Means model was trained with K=3. The countries were grouped into 3 clusters, with a Silhouette Score of 0.2833.
-   **K-Medoids Clustering:** A K-Medoids model was trained with K=3, yielding a Silhouette Score of 0.1562.
-   **Cluster Profiling:** Mean values of features for each K-Means cluster were calculated to understand their characteristics.
-   **Cluster Labeling:** Clusters were manually labeled based on their profiles:
    -   **Cluster 0: Developed** (high GDP, low child mortality)
    -   **Cluster 1: Underdeveloped** (high child mortality, low life expectancy, low GDP)
    -   **Cluster 2: Developing** (intermediate characteristics)

### 8. Clustering: DBSCAN & Hierarchical Methods
-   **DBSCAN:** Applied to identify density-based clusters and outliers. It found 1 main cluster and 30 countries identified as 'noise' or outliers.
-   **Hierarchical Clustering:** A dendrogram was generated using Ward's method. Agglomerative clustering with 3 clusters was performed, yielding a Silhouette Score of 0.2456.

### 9. Dimensionality Reduction: PCA
-   **PCA Application:** Principal Component Analysis was used to reduce the 9 features to 2 principal components for visualization.
-   **Variance Explained:** The first two principal components captured 63.1% of the total variance.
-   **Visualization:** A scatterplot of the PCA-transformed data, colored by K-Means clusters, showed good separation between the three development levels.
-   **Scree Plot:** Showed that the first 4 PCs explain approximately 87.2% of the total variance.

### 10. Supervised Classification Models
To predict the assigned development levels, several supervised models were trained and evaluated:
-   **Random Forest Classifier:** Achieved 100.0% test accuracy and 95.8% cross-validation accuracy.
-   **Logistic Regression:** Achieved 100.0% test accuracy and 95.8% cross-validation accuracy.
-   **Naive Bayes:** Achieved 100.0% test accuracy and 95.8% cross-validation accuracy.
-   **K-Nearest Neighbors (KNN):** Achieved 91.2% test accuracy and 94.0% cross-validation accuracy.
-   **Support Vector Machine (SVM):** Achieved 97.1% test accuracy and 96.4% cross-validation accuracy.
-   **Decision Tree:** Achieved 94.1% test accuracy and 92.2% cross-validation accuracy.
-   **Gradient Boosting Classifier:** Achieved 97.1% test accuracy and 94.0% cross-validation accuracy.
-   **LightGBM:** Achieved 100.0% test accuracy and 95.2% cross-validation accuracy.
-   **Stacking Classifier:** Achieved 100.0% test accuracy and 96.4% cross-validation accuracy.
-   **Ada Boost Classifier:** Achieved 97.1% test accuracy and 97.0% cross-validation accuracy.

### 11. Model Evaluation & Comparison
A bar chart visually compared the test and cross-validation accuracies of all trained supervised models, demonstrating the high predictive power of most models in classifying countries into their development levels.

### 12. Cluster Profiling & Visualization
Boxplots illustrated the distribution of key indicators (`child_mort`, `life_expec`, `gdpp`, `income`, `health`, `total_fer`) across the `Developed`, `Developing`, and `Underdeveloped` levels, clearly showcasing the distinct characteristics of each cluster.

### 13. Interactive Summary & Top Aid Countries
-   **Summary Table:** Presented sample countries for each development level along with their average key metrics (GDP, life expectancy, child mortality).
-   **Top 10 Countries in Need of Aid:** Identified and visualized the top 10 countries from the 'Underdeveloped' cluster with the highest 'need_score' (a composite score based on child mortality, life expectancy, and GDP per capita).

## 14. Key Observations and Conclusion
-   The project successfully segmented 167 countries into three distinct development levels: **Developed**, **Developing**, and **Underdeveloped**, using K-Means clustering.
-   Critical indicators like `child_mort`, `life_expec`, and `gdpp` were identified as the most influential features in determining a country's development level.
-   Various supervised classification models demonstrated high accuracy (many achieving 100% test accuracy) in predicting these development levels, validating the robustness of the clustering.
-   A clear list of the top 10 countries most in need of aid was generated, providing actionable insights for organizations focusing on international development.
