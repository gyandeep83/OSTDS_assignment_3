import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_excel("/Users/gyandeep/OSTDS_assign_3/Datasets/Real estate valuation data set.xlsx")

# Rename columns
df.columns = [
    'No', 'Transaction_Date', 'House_Age', 'Distance_MRT',
    'Convenience_Stores', 'Latitude', 'Longitude', 'Price_per_Unit_Area'
]

# Drop irrelevant columns
df.drop(columns=['No', 'Transaction_Date'], inplace=True)

# Keep for location-based plotting
df_latlong = df[['Latitude', 'Longitude', 'Price_per_Unit_Area']].copy()

# Features and target
features = ['House_Age', 'Distance_MRT', 'Convenience_Stores']
X = df[features]
y = df['Price_per_Unit_Area']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MSE:", round(mse, 2))
print("R² Score:", round(r2, 2))

# ========== USER PREDICTION INPUT ==========
print("\nEnter house details to predict price.")

try:
    house_age = float(input("Enter House Age (years): "))
    distance_mrt = float(input("Enter Distance to MRT (in meters): "))
    num_stores = int(input("Enter Number of Convenience Stores nearby: "))

    user_input = [[house_age, distance_mrt, num_stores]]
    predicted_price = model.predict(user_input)[0]

    print(f"Predicted Price per Unit Area: {predicted_price:.2f}")

    # Fake coordinates for user marker (center of dataset)
    user_lat = df_latlong['Latitude'].mean()
    user_long = df_latlong['Longitude'].mean()

    # ========== 1. LOCATION PLOT ==========
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(
        df_latlong['Longitude'],
        df_latlong['Latitude'],
        c=df_latlong['Price_per_Unit_Area'],
        cmap='coolwarm',
        s=60,
        edgecolor='k',
        label='Dataset Prices'
    )

    plt.scatter(user_long, user_lat, color='black', s=100, marker='X', label='User Prediction')
    plt.annotate(
        f'Predicted: {predicted_price:.2f}',
        (user_long, user_lat),
        textcoords="offset points",
        xytext=(10, 10),
        ha='left',
        fontsize=10,
        color='black'
    )

    plt.colorbar(scatter, label='Price per Unit Area')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('House Prices by Location (With Your Prediction)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ========== 2. ACTUAL vs PREDICTED ==========
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=y_test, y=y_pred, color='blue', s=60, label='Model Predictions')
    plt.scatter(predicted_price, predicted_price, color='red', s=100, marker='X', label='Your Prediction')

    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='gray', label='Ideal Fit')
    plt.xlabel('Actual Price per Unit Area')
    plt.ylabel('Predicted Price per Unit Area')
    plt.title('Actual vs Predicted House Prices')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

except ValueError:
    print("Invalid input. Please enter numeric values only.")
