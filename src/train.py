import pandas as pd
import json
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load dataset
df = pd.read_csv("data/housing.csv")
df = df.dropna()
# Convert categorical column
df['ocean_proximity'] = df['ocean_proximity'].astype('category').cat.codes

# Features and target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
preds = model.predict(X_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

metrics = {
    "dataset_size": len(df),
    "rmse": float(rmse),
    "r2": float(r2)
}

print(metrics)

# Save metrics for GitHub Actions
os.makedirs("output", exist_ok=True)

joblib.dump(model, "output/model.pkl")
with open("output/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
