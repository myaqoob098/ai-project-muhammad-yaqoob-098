# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Expanded Synthetic Dataset Creation (More Data)
data = {
    'Area (sq ft)': [1000, 1500, 2000, 2500, 3000, 3500, 1200, 1600, 2200, 2700, 3200, 3800, 1300, 1700, 2300],
    'Bedrooms': [2, 3, 3, 4, 4, 5, 2, 3, 3, 4, 4, 5, 2, 3, 3],
    'Age (years)': [10, 5, 20, 15, 10, 8, 12, 7, 18, 14, 9, 6, 11, 8, 16],
    'Location_Score': [8, 7, 6, 9, 10, 5, 7, 8, 6, 9, 10, 5, 8, 7, 6],  # New feature
    'Price (in lakhs)': [50, 70, 85, 110, 150, 180, 55, 75, 90, 115, 160, 185, 58, 76, 92]
}
df = pd.DataFrame(data)

# Save the dataset to a CSV file
df.to_csv('house_prices_dataset.csv', index=False)

# Display Dataset
print("Dataset:")
print(df)

# Data Visualization: Pairplot and Correlation Matrix
sns.pairplot(df)
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Splitting Data into Features (X) and Target (y)
X = df[['Area (sq ft)', 'Bedrooms', 'Age (years)', 'Location_Score']]
y = df['Price (in lakhs)']

# Splitting into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training: Random Forest Regressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# Interactive Prediction
print("\n--- Predict House Price ---")

try:
    area = float(input("Enter the area (sq ft): "))
    bedrooms = int(input("Enter the number of bedrooms: "))
    age = int(input("Enter the age of the house: "))
    location_score = int(input("Enter the location score (1-10): "))

    # Create new data point for prediction as a DataFrame (not numpy array)
    new_data = pd.DataFrame([[area, bedrooms, age, location_score]], columns=['Area (sq ft)', 'Bedrooms', 'Age (years)', 'Location_Score'])

    # Make prediction
    predicted_price = model.predict(new_data)
    print(f"Predicted Price: {predicted_price[0]:.2f} lakhs")
except ValueError as e:
    print(f"Invalid input, please enter numeric values: {e}")
