# FootballPlayers - Machine Learning for Role Prediction and Player Recommendation

## Overview

**FootballPlayers** is a university project aimed at classifying football players and recommending similar ones based on user preferences using machine learning techniques. The project leverages classification models and a recommendation system to make role predictions and personalized suggestions based on player statistics.

The dataset, sourced from Kaggle, was preprocessed and used to train and evaluate various machine learning models. The system is composed of four main modules: data preprocessing, data analysis, classification, and recommendation.

## Dataset

- Source: [Kaggle – All Football Players Stats in Top 5 Leagues (2023/24)](https://www.kaggle.com/datasets/orkunaktas/all-football-players-stats-in-top-5-leagues-2324)

## Modules

### 1. Preprocessing

- Handled null values and outliers
- Converted categorical variables (e.g., player names, positions) into numerical formats
- Normalized numerical features using `StandardScaler`
- Selected relevant features: `Player`, `Pos`, `Born`, `90s`, `Gls`, `Ast`, `CrdY`, `CrdR`

**Tools**: `Pandas`, `Scikit-Learn`

### 2. Data Analysis (EDA)

- Descriptive statistics and visualizations (heatmaps, histograms)
- Correlation analysis between variables
- Detection and treatment of outliers
- Analysis of the target variable (`Pos_format`)

**Tools**: `Pandas`, `Matplotlib`, `Seaborn`

### 3. Classification

Predicted player roles using:
- **K-Nearest Neighbors**
- **Gaussian Naive Bayes**
- **Random Forest** (Best performing model)

**Best Model**:
- Random Forest with `15-fold cross-validation`
- Accuracy: **96.55%**
- GridSearchCV for hyperparameter optimization

**Evaluation Metrics**:
- Accuracy, Precision, Recall, F1-Score, ROC AUC

**Tools**: `Scikit-Learn`, `NumPy`, `Pandas`

### 4. Recommendation System

- Recommends similar players based on user input
- Uses **Manhattan Distance** (1 / (1 + distance)) for similarity calculation
- Dynamically adapts to available user input
- Returns top 10 most similar players excluding already selected ones

**Tools**: `Scikit-Learn (pairwise_distances)`

## Results

- **Random Forest** achieved the best performance among classification models.
- The **recommendation module** provided accurate and relevant suggestions based on input statistics.
- The system demonstrates solid integration of machine learning techniques in a sports context.
