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

### WEEK4 Assignment
# CIFAR-10 Image Classification: ANN vs. CNN Performance Analysis

## Overview
This notebook explores the performance of Artificial Neural Networks (ANNs) and Convolutional Neural Networks (CNNs) on a synthetic subset of the CIFAR-10 image classification dataset. The primary goal is to compare how different network architectures and training strategies impact model accuracy and generalization capabilities for image-based tasks.

## Project Scope
The assignment covers the following key areas:
-   **ANN vs. CNN Comparison**: Directly contrasting the effectiveness of ANNs (which flatten image data) and CNNs (which process spatial features).
-   **Deeper ANN Layers (Task 1)**: Investigating if increasing the depth of an ANN can overcome its inherent limitations for image data.
-   **More CNN Filters (Task 2)**: Evaluating the impact of increasing the number of filters in convolutional layers on feature extraction and performance.
-   **Early Stopping (Tasks 3 & 4)**: Implementing a callback to prevent overfitting and optimize training duration by monitoring validation loss.
-   **Data Augmentation (Task 5)**: Applying image transformations during training to enhance model robustness and generalization.
-   **Final Comparison**: A comprehensive analysis and visualization of all models' performance.

## Dataset
The project uses a synthetic version of the CIFAR-10 dataset, consisting of:
-   **10,000 training samples**
-   **2,000 test samples**
-   **10 classes** of images (e.g., airplane, automobile, bird, cat, etc.)
-   **Image size**: 32x32 pixels with 3 color channels (RGB).

## Architectures and Strategies Implemented
1.  **Baseline ANN**: A foundational ANN with two hidden dense layers.
2.  **Deep ANN (Task 1)**: An expanded ANN with four hidden dense layers to test the effect of increased depth.
3.  **Baseline CNN**: A standard CNN architecture with convolutional, batch normalization, and max-pooling layers.
4.  **CNN More Filters (Task 2)**: An enhanced CNN based on the baseline, but with an increased number of filters in its convolutional layers.
5.  **CNN EarlyStopping (Tasks 3 & 4)**: The 'CNN More Filters' architecture trained with an `EarlyStopping` callback to halt training when validation loss stops improving.
6.  **CNN Augmentation (Task 5)**: The 'CNN More Filters' architecture integrated with a data augmentation pipeline (random flips, rotations, and zooms) to improve generalization.

## Key Findings & Conclusion
The experiments clearly demonstrate that **Convolutional Neural Networks (CNNs) significantly outperform Artificial Neural Networks (ANNs)** for image classification tasks. This is primarily due to CNNs' ability to automatically learn and preserve spatial hierarchies within image data, which ANNs lose when images are flattened.

Enhancements such as **increasing CNN filter count**, implementing **EarlyStopping**, and especially **Data Augmentation** further improved the CNN models' performance and generalization capabilities, underscoring their importance in robust image classification.

## Week 5 Assingment

# Deep Learning Text Generation Learning Project

## Text Generation using Vanilla RNN, LSTM, and GRU

This project is designed for students and beginners to explore and understand how sequence models learn: grammar, sentence flow, contextual dependencies, and perform next-word prediction for text generation.

**Goal:** To compare the performance of **Simple RNN, LSTM, and GRU** models on the same text corpus and understand why gated architectures are generally more effective for sequence modeling.

## Problem Statement

Design and implement Deep Learning (DL) models capable of learning the underlying structure, grammar, and contextual dependencies of a given text corpus to generate coherent and meaningful text sequences using:

1.  **Vanilla RNN**
2.  **LSTM**
3.  **GRU**

Then compare:
-   training loss
-   generated text quality
-   memory handling
-   long-term dependency learning

## Implementation Details

### 1. Data Preparation
-   A small, custom text corpus is used to allow for quick experimentation and clear observation of model behavior.
-   Text is tokenized, and `n-gram` sequences are created to train the models for next-word prediction.
-   Sequences are padded to ensure uniform input length for the neural networks.

### 2. Model Architectures
Each model consists of:
-   An **Embedding Layer**: Converts input integer tokens into dense vector representations.
-   A **Recurrent Layer**: 
    -   `SimpleRNN` for the Vanilla RNN.
    -   `LSTM` for the LSTM model.
    -   `GRU` for the GRU model.
-   A **Dense Output Layer**: With `softmax` activation to predict the probability distribution of the next word in the vocabulary.

### 3. Training
-   Models are compiled using `sparse_categorical_crossentropy` as the loss function and the `Adam` optimizer.
-   Each model is trained for 200 epochs to observe their learning dynamics.

### 4. Text Generation Function
-   A utility function `generate_text` is implemented to predict subsequent words based on a given seed text, allowing us to qualitatively assess each model's text generation capability.

## Key Findings 

During my work on this project, I observed the following:

-   **Vanilla RNN** was effective for very short patterns but really struggled with remembering information over longer sentences. This made its generated text less coherent, which was a clear limitation.
-   **LSTM** was a significant step up! Its ability to 'remember' context for much longer, thanks to its special gates, meant it achieved much lower training loss and generated more meaningful text. It was exciting to see how much better it performed.
-   **GRU** was also impressive. It performed very similarly to LSTM, but I found it to be a bit more streamlined. It's a great option when you need good performance without all the complexity of LSTM.

This project truly helped me understand, both theoretically and practically, why **gated recurrent networks (like LSTM and GRU) are far more effective than simple RNNs** for sequence modeling tasks, especially in text generation. The architectural differences directly translated into better learning, more robust models, and ultimately, higher quality generated text, which is a crucial insight for future deep learning endeavors.


## Week 6

# Image Denoising with a Convolutional Autoencoder on MNIST

## Project Objective
This project aims to build a deep learning model capable of removing noise from images using a Convolutional Autoencoder (CAE) on the MNIST dataset.

## Project Overview
We implement a denoising autoencoder to reconstruct clean MNIST digit images from their noisy counterparts. The process involves loading and preprocessing the MNIST dataset, artificially introducing Gaussian noise, training a CAE, and evaluating its performance.

## Table of Contents
1.  [Setup and Dependencies](#setup-and-dependencies)
2.  [Data Loading and Preprocessing](#data-loading-and-preprocessing)
3.  [Adding Noise to Images](#adding-noise-to-images)
4.  [Model Architecture: Convolutional Autoencoder](#model-architecture-convolutional-autoencoder)
5.  [Model Training](#model-training)
6.  [Evaluation and Denoising Results](#evaluation-and-denoising-results)
7.  [Conclusion](#conclusion)

### 1. Setup and Dependencies
This project uses TensorFlow and Keras for building and training the deep learning model, along with NumPy, Matplotlib, and PIL for data handling and visualization.

### 2. Data Loading and Preprocessing
- The MNIST dataset, provided as PNG images in `archive.zip`, is extracted.
- Images are loaded from `mnist_png` directories (training and testing splits) and resized to 28x28 pixels.
- Pixel values are normalized to the range \[0, 1\].
- A channel dimension is added, resulting in image shapes of `(28, 28, 1)`.

**Dataset Statistics:**
- Training images: 20,000 (28, 28, 1)
- Testing images: 4,000 (28, 28, 1)

### 3. Adding Noise to Images
Gaussian noise with a `NOISE_FACTOR` of 0.4 is programmatically added to the clean MNIST images to simulate real-world noisy conditions. The autoencoder's task is to learn to map these noisy inputs back to their original clean versions.

### 4. Model Architecture: Convolutional Autoencoder
The CAE comprises an encoder and a decoder:
- **Encoder:** Uses `Conv2D` and `MaxPooling2D` layers to downsample the noisy input image into a compressed latent representation.
- **Decoder:** Uses `Conv2D` and `UpSampling2D` layers to reconstruct the clean image from the latent space.

The model is compiled with the Adam optimizer and `binary_crossentropy` as the loss function.

### 5. Model Training
The autoencoder is trained for 20 epochs with a batch size of 128. Noisy training images (`x_train_noisy`) are used as input, and original clean training images (`x_train`) are used as targets. The model's performance is validated on the noisy test set (`x_test_noisy`) against clean test images (`x_test`).

**Training Insights:**
The training and validation losses rapidly decreased and then flattened, indicating effective learning and good generalization without overfitting.

### 6. Evaluation and Denoising Results
- The model achieved a test loss (binary crossentropy) of approximately **0.0916**.
- The Mean Squared Error (MSE) across the test set is **0.00979**, with a standard deviation of **0.00369**.
- Visualizations demonstrate the autoencoder's ability to significantly restore clarity to noisy digits, showcasing clean original images, noisy inputs, and the denoised outputs for each digit from 0 to 9.

### 7. Conclusion
This project successfully implemented a Convolutional Autoencoder for image denoising on the MNIST dataset. The model effectively learned to remove noise, as evidenced by low test loss and MSE, and visually improved image clarity. This demonstrates the practical application of autoencoders in image processing.

## Week 7
# FetchWise: RAG Chatbot

FetchWise is a sophisticated Retrieval-Augmented Generation (RAG) chatbot designed to provide intelligent answers based on your own documents. It leverages advanced NLP models and vector databases to deliver accurate, context-aware responses.

## 🚀 Features

- **Multi-Document Support**: Upload and index PDFs, Word documents, and HTML files.
- **RAG-Powered Conversations**: Uses Retrieval-Augmented Generation to ground AI responses in your specific data.
- **Conversational Memory**: Maintains chat history within a session for context-aware follow-up responses.
- **Interactive UI**: A clean, modern Streamlit interface for seamless user interaction.
- **High-Performance Vector Search**: Uses ChromaDB for fast, accurate semantic retrieval.
- **Scalable Backend**: Powered by FastAPI for robust and efficient API handling.

### ⚙️ How it Works

1.  **Ingestion & Vectorization**: Uploaded files are chunked into 1000-character segments with overlap. These chunks are embedded using HuggingFace's `all-MiniLM-L6-v2` and stored in **ChromaDB**.
2.  **Conversational Retrieval**: The system is "history-aware." It uses the session's chat log to contextualize user queries, ensuring follow-up questions (e.g., "Why?") are understood correctly.
3.  **High-Speed Generation**: We utilize **Llama 3.3 (70B) on Groq Cloud** for near-instant inference, synthesizing the retrieved context into a clear and accurate final response.

> [!TIP]
> **Vector-First Architecture**: We use **ChromaDB** for specialized vector search, keeping retrieval fast and the stack lightweight — no separate database server required for chat history or metadata.

## 🛠️ Technology Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: [FastAPI](https://fastapi.tiangolo.com/)
- **Orchestration**: [LangChain](https://www.langchain.com/)
- **LLM**: Groq — `llama-3.3-70b-versatile` (via `langchain-groq`)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **Document Parsing**: PyPDF, docx2txt, Unstructured
