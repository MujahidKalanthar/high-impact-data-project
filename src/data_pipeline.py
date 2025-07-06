import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.modeling import train_predictive_model

def main():
    download_data()
    df_raw = load_data()
    df_clean = clean_data(df_raw)
    df_features = feature_engineering(df_clean)

    # Train and evaluate model for US
    model, X_test, y_test, y_pred = train_predictive_model(df_features, country="US")


def basic_eda(df):
    print("Starting EDA...")

    # Summary statistics of cases
    print(df["ConfirmedCases"].describe())

    # Plot total confirmed cases over time globally
    df_global = df.groupby("Date")["ConfirmedCases"].sum().reset_index()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_global, x="Date", y="ConfirmedCases")
    plt.title("Global Confirmed COVID-19 Cases Over Time")
    plt.xlabel("Date")
    plt.ylabel("Confirmed Cases")
    plt.tight_layout()
    plt.savefig("../data/global_cases_over_time.png")  # Save plot as PNG
    plt.show()

    print("EDA complete, plot saved to data/global_cases_over_time.png")


DATA_URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/" \
           "csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATA_PATH = os.path.join(DATA_DIR, "covid_confirmed_global.csv")

def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print("Downloading dataset...")
        response = requests.get(DATA_URL)
        if response.status_code == 200:
            with open(DATA_PATH, "wb") as f:
                f.write(response.content)
            print(f"Data downloaded and saved to {DATA_PATH}")
        else:
            raise Exception(f"Failed to download data, status code: {response.status_code}")
    else:
        print("Data file already exists, skipping download.")


def feature_engineering(df):
    print("Starting feature engineering...")

    # Sort data for difference calculation
    df = df.sort_values(by=["Country", "Province", "Date"])

    # 1. Calculate daily new confirmed cases (difference of cumulative cases)
    df["NewCases"] = df.groupby(["Country", "Province"])["ConfirmedCases"].diff().fillna(0)

    # Replace any negative new cases (data corrections) with 0
    df["NewCases"] = df["NewCases"].apply(lambda x: x if x > 0 else 0)

    # 2. Extract date features for time series context
    df["DayOfWeek"] = df["Date"].dt.dayofweek  # Monday=0, Sunday=6
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week
    df["Month"] = df["Date"].dt.month

    # 3. Create a feature for cumulative cases lagged by 7 days (to capture recent trend)
    df["ConfirmedCases_7d_lag"] = df.groupby(["Country", "Province"])["ConfirmedCases"].shift(7).fillna(0)

    print("Feature engineering completed.")
    return df


def load_data():
    print("Loading data into DataFrame...")
    df = pd.read_csv(DATA_PATH)
    print(f"Data shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    return df

def clean_data(df):
    print("Starting data cleaning...")

    # 1. Rename columns for consistency
    df = df.rename(columns={
        "Province/State": "Province",
        "Country/Region": "Country",
        "Lat": "Latitude",
        "Long": "Longitude"
    })

    # 2. Check for missing values
    missing = df.isnull().sum()
    print("Missing values per column:")
    print(missing[missing > 0])

    # 3. Fill missing Province values with empty string (no province)
    df["Province"] = df["Province"].fillna("")

    # 4. Check for duplicates (unlikely but good practice)
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows count: {duplicates}")

    # 5. Convert date columns from wide to long format for easier analysis
    # Date columns start after 4 fixed columns: Province, Country, Latitude, Longitude
    date_cols = df.columns[4:]
    df_long = df.melt(
        id_vars=["Province", "Country", "Latitude", "Longitude"],
        value_vars=date_cols,
        var_name="Date",
        value_name="ConfirmedCases"
    )

    # 6. Convert 'Date' column to datetime type
    df_long["Date"] = pd.to_datetime(df_long["Date"])

    print("Data cleaning completed.")
    return df_long


if __name__ == "__main__":
    main()
