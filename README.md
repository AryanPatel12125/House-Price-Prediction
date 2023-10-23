## Aryan Patel
##### aryan12patel@gmail.com

# Project Overview:
# House Price Prediction Project

## Overview
This project aims to predict house prices in Ames, Iowa, utilizing a machine learning model trained on various features such as size, quality, and additional amenities. The predictions are made based on hypotheses formulated around the impact of these features on house prices.

## Hypotheses
1. **Size Matters:** Larger houses, as indicated by features like first floor area (`1stFlrSF`), second floor area (`2ndFlrSF`), and above-ground living area (`GrLivArea`), are expected to have higher market prices.
2. **Quality Sells:** The quality of a house, represented by features like overall condition (`OverallCond`) and overall material and finish quality rating (`OverallQual`), significantly influences its price. Higher quality ratings are hypothesized to correlate with higher house prices.
3. **Amenities Add Value:** Additional features such as finished basements (`BsmtFinSF1`), garage area (`GarageArea`), and larger lot sizes (`LotArea`) are expected to contribute positively to house prices.

## Validation Process
- **Hypothesis 1 Validation:** Correlation analysis and scatter plots will be utilized to confirm the positive relationship between size-related features and house prices.
- **Hypothesis 2 Validation:** Box plots and bar charts will be employed to compare price distributions across different quality ratings. Statistical tests will validate the correlation between quality ratings and house prices.
- **Hypothesis 3 Validation:** Visualizations and statistical methods will be applied to validate the impact of amenities on house prices.

## Project Structure

### 1. Data Preprocessing and Feature Engineering (File 1)
- Exploratory data analysis, handling missing values, and outlier removal.
- Encoding categorical variables and preparing the dataset for model training.

### 2. Model Training and Hyperparameter Tuning (File 2)
- Loading the pre-trained XGBoost model and refining the dataset.
- Training the model, selecting relevant features, and optimizing hyperparameters using GridSearchCV.

### 3. Streamlit Web Application (File 3)
- Building a user-friendly web interface using Streamlit for real-time house price predictions.
- Implementing input features and integrating the optimized model for accurate predictions.

## How to Use
1. **Data Preprocessing:** Run `File 1` in a Jupyter notebook to preprocess the data and engineer relevant features.
2. **Model Training:** Execute `File 2` to load the pre-trained model, select features, and optimize hyperparameters.
3. **Web Application:** Run `File 3` (app.py) in your local environment. Input house features in the web interface to get real-time price predictions.

## Requirements
- Python 3.x
- Libraries: Pandas, NumPy, Seaborn, Matplotlib, XGBoost, Streamlit

## Author
- Aryan Patel

- https://linkedin.com/aryanpatel1205


---
