# ==========================================
# PREDICTIVE ANALYTICS USING HISTORICAL DATA
# Store Sales Forecasting using Linear Regression
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------
# STEP 1: LOAD DATASET
# ------------------------------------------

df = pd.read_csv("store_sales.csv")

print("\nFIRST 5 RECORDS")
print(df.head())

# ------------------------------------------
# STEP 2: CHECK DATASET INFORMATION
# ------------------------------------------

print("\nDATASET INFORMATION")
print(df.info())

print("\nMISSING VALUES")
print(df.isnull().sum())

# ------------------------------------------
# STEP 3: PREPROCESS DATA
# ------------------------------------------

# Convert Date Column into Datetime Format
df['date'] = pd.to_datetime(df['date'])

# Create New Features from Date
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek

print("\nPREPROCESSED DATA")
print(df.head())

# ------------------------------------------
# STEP 4: DEFINE FEATURES AND TARGET
# ------------------------------------------

X = df[['store', 'promo', 'holiday',
        'year', 'month', 'day', 'dayofweek']]

y = df['sales']

# ------------------------------------------
# STEP 5: SPLIT DATASET
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Records :", len(X_train))
print("Testing Records  :", len(X_test))

# ------------------------------------------
# STEP 6: TRAIN MODEL
# ------------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

print("\nMODEL TRAINED SUCCESSFULLY")

# ------------------------------------------
# STEP 7: MAKE PREDICTIONS
# ------------------------------------------

y_pred = model.predict(X_test)

# ------------------------------------------
# STEP 8: EVALUATE MODEL
# ------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nMODEL EVALUATION")
print("-----------------------------")
print("Mean Absolute Error (MAE) :", round(mae, 2))
print("Mean Squared Error (MSE)  :", round(mse, 2))
print("Root Mean Squared Error   :", round(rmse, 2))
print("R2 Score                  :", round(r2, 4))

# ------------------------------------------
# STEP 9: COMPARE ACTUAL VS PREDICTED
# ------------------------------------------

results = pd.DataFrame({
    'Actual Sales': y_test.values,
    'Predicted Sales': y_pred
})

print("\nACTUAL VS PREDICTED SALES")
print(results.head(10))

# ------------------------------------------
# STEP 10: VISUALIZATION
# ------------------------------------------

plt.figure(figsize=(12,6))

plt.plot(
    results['Actual Sales'].values[:100],
    label='Actual Sales'
)

plt.plot(
    results['Predicted Sales'].values[:100],
    label='Predicted Sales'
)

plt.title("Actual Sales vs Predicted Sales")

plt.xlabel("Records")

plt.ylabel("Sales")

plt.legend()

plt.grid(True)

plt.savefig("output_graph.png")

plt.show()

print("\nGraph Saved as output_graph.png")

# ------------------------------------------
# STEP 11: PREDICT FUTURE SALES
# ------------------------------------------

future_data = pd.DataFrame({
    'store': [1],
    'promo': [1],
    'holiday': [0],
    'year': [2026],
    'month': [7],
    'day': [1],
    'dayofweek': [2]
})

future_sales = model.predict(future_data)

print("\nFUTURE SALES PREDICTION")
print("-----------------------------")
print("Predicted Sales:", round(future_sales[0], 2))

# ==========================================
# END OF PROJECT
# ==========================================