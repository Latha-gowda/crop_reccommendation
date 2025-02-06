# crop_recommendation.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# ----------------------------
# Step 1. Load the dataset
# ----------------------------
# The CSV is assumed to have columns:
# ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall', 'Crop']
df = pd.read_csv('Crop_recommendation.csv')
print("Dataset preview:")
print(df.head())

# ----------------------------
# Step 2. Preprocess the data
# ----------------------------
# In this simple example we assume the data is mostly clean.
# (You can add additional steps such as handling missing values, scaling, etc.)
X = df[['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']]
y = df['Crop']

# Optionally, fill missing values (if any)
X.fillna(X.mean(), inplace=True)

# ----------------------------
# Step 3. Split the dataset
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ----------------------------
# Step 4. Train a KNN model
# ----------------------------
# Here we use k=3; you can optimize k using cross-validation.
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# ----------------------------
# Step 5. Evaluate the model
# ----------------------------
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ----------------------------
# Step 6. Save the trained model
# ----------------------------
with open('crop_recommendation_model.pkl', 'wb') as file:
    pickle.dump(knn, file)
print("Trained model saved as 'crop_recommendation_model.pkl'.")

# ----------------------------
# Step 7. Prediction function
# ----------------------------
def recommend_crop(N, P, K, Temperature, Humidity, pH, Rainfall):
    """
    Given the input parameters:
      - N: Nitrogen content in soil
      - P: Phosphorous content in soil
      - K: Potassium content in soil
      - Temperature: Temperature (°C)
      - Humidity: Humidity (%)
      - pH: Soil pH value
      - Rainfall: Rainfall (mm)
    this function returns the recommended crop.
    """
    # Create a NumPy array from the inputs and reshape it into a 2D array
    input_data = np.array([N, P, K, Temperature, Humidity, pH, Rainfall]).reshape(1, -1)
    prediction = knn.predict(input_data)
    return prediction[0]

# Example usage:
example_recommendation = recommend_crop(90, 42, 43, 20.87, 82.0, 6.5, 202.9)
print("\nRecommended Crop for the example input:", example_recommendation)
