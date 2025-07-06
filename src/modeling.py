import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

def train_predictive_model(df, country="US"):
    print(f"Training model for {country}...")

    # Filter data for the chosen country only
    df_country = df[df["Country"] == country].copy()

    # Drop rows where target or lag features are NaN
    df_country = df_country.dropna(subset=["NewCases", "ConfirmedCases_7d_lag"])

    # Features and target
    feature_cols = ["DayOfWeek", "WeekOfYear", "Month", "ConfirmedCases_7d_lag"]
    X = df_country[feature_cols]
    y = df_country["NewCases"]

    # Split into train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )  # No shuffle for time-series

    # Initialize and train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluate with RMSE
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"Test RMSE: {rmse:.2f}")

    # Return model and predictions for further use
    return model, X_test, y_test, y_pred
