import xgboost as xgb
from sklearn.datasets import make_regression

# Generate dummy data
X, y = make_regression(n_samples=100, n_features=10, random_state=42)

# Try initializing an XGBoost model with GPU
try:
    model = xgb.XGBRegressor(tree_method="gpu_hist")
    model.fit(X, y)  # Fit model
    print("XGBoost GPU is working correctly!")
except xgb.core.XGBoostError as e:
    print("XGBoost GPU is not configured properly. Error:")
    print(e)
