# Predictive_analysis

## Overview

This project demonstrates the use of Predictive Analytics and Machine Learning to forecast future sales using historical store sales data. The dataset contains information such as sales, store details, promotional activities, and holiday indicators. A Linear Regression model is used to learn patterns from historical data and predict future sales values.

---

## Problem Statement

Businesses generate large amounts of historical sales data. Analyzing this data helps identify trends and forecast future sales. Accurate sales prediction supports inventory management, marketing strategies, and business planning.

---

## Objectives

- Load and analyze historical sales data.
- Clean and preprocess the dataset.
- Extract useful features from the date column.
- Train a Linear Regression model.
- Predict future sales values.
- Evaluate model performance using standard metrics.
- Visualize actual and predicted sales.

---

## Dataset Features

| Feature | Description |
|----------|------------|
| date | Date of sale |
| store | Store identifier |
| sales | Sales amount (Target Variable) |
| promo | Promotion status (0 or 1) |
| holiday | Holiday status (0 or 1) |

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn

---

## Methodology

### 1. Data Loading
The historical sales dataset is loaded using Pandas.

### 2. Data Preprocessing
- Check missing values.
- Convert date column into datetime format.
- Extract year, month, day, and weekday features.

### 3. Feature Selection
Input Features:
- Store
- Promotion
- Holiday
- Year
- Month
- Day
- Day of Week

Target Variable:
- Sales

### 4. Model Training
The dataset is divided into:
- 80% Training Data
- 20% Testing Data

A Linear Regression model is trained using the training data.

### 5. Prediction
The trained model predicts sales values for the test dataset.

### 6. Model Evaluation
The following metrics are used:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### 7. Visualization
A graph comparing Actual Sales and Predicted Sales is generated.

---

## Expected Output

- Sales prediction values.
- Model accuracy metrics.
- Actual vs Predicted Sales graph.
- Future sales forecast.

---

## Conclusion

This project successfully applies Machine Learning techniques to historical sales data for forecasting purposes. Predictive Analytics enables better business decision-making by providing accurate sales predictions and trend analysis.

---

## Author

Thejeswini G
