import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("traffic_signal_data.csv")

print("\nDataset Loaded Successfully!")
print("Total Records:", len(data))


# ==========================================
# 2. CREATE ADVANCED FEATURES
# ==========================================

# Traffic density
data["traffic_density"] = (
    data["vehicles"] / (data["waiting_time"] + 1)
)

# Estimated traffic pressure
data["traffic_pressure"] = (
    data["vehicles"] * 0.7 +
    data["waiting_time"] * 0.3
)

print("\nFeatures Created Successfully!")


# ==========================================
# 3. SELECT FEATURES
# ==========================================

features = [
    "vehicles",
    "waiting_time",
    "traffic_density",
    "traffic_pressure"
]

X = data[features]

# Target
y = data["signal_time"]


# ==========================================
# 4. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 5. TRAIN ADVANCED AI MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

model.fit(X_train, y_train)

print("\nAI Model Training Completed!")


# ==========================================
# 6. PREDICTION
# ==========================================

predictions = model.predict(X_test)


# ==========================================
# 7. MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, predictions)


print("\n========== MODEL PERFORMANCE ==========")

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 2))


# ==========================================
# 8. FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")

print(importance)


# ==========================================
# 9. SAVE MODEL
# ==========================================

with open("smart_traffic_model.pkl", "wb") as file:
    pickle.dump(model, file)


print("\nModel saved as:")
print("smart_traffic_model.pkl")


# ==========================================
# 10. SAVE FEATURE LIST
# ==========================================

with open("features.pkl", "wb") as file:
    pickle.dump(features, file)


print("Feature information saved!")


# ==========================================
# 11. TEST AI WITH SAMPLE DATA
# ==========================================

sample = pd.DataFrame([{
    "vehicles": 80,
    "waiting_time": 40,
    "traffic_density": 80 / 41,
    "traffic_pressure": 80 * 0.7 + 40 * 0.3
}])

result = model.predict(sample)[0]

print("\n========== SAMPLE PREDICTION ==========")

print(
    "Recommended Green Signal:",
    round(result, 2),
    "seconds"
)

print("\n=======================================")
print("Smart Traffic AI is ready!")