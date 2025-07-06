import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.data_pipeline import download_data, load_data, clean_data, feature_engineering
from src.modeling import train_predictive_model

# Set Streamlit page config (optional)
st.set_page_config(page_title="COVID Dashboard", layout="wide")

# =========================
# Sidebar FIRST — always visible
# =========================
st.sidebar.title("Settings")

# =========================
# Main App Logic
# =========================
def main():
    st.title("🦠 COVID-19 Data Analysis & Prediction")

    with st.spinner("Loading and processing data..."):
        download_data()
        df_raw = load_data()
        df_clean = clean_data(df_raw)
        df_features = feature_engineering(df_clean)

    # Country selection — now that df is loaded
    countries = sorted(df_features["Country"].unique())
    country = st.sidebar.selectbox("Select a Country", countries, index=countries.index("US"))

    # Filter data
    df_country = df_features[df_features["Country"] == country]

    # Section: Line chart of cumulative cases
    st.subheader(f"📈 Confirmed Cases Over Time — {country}")
    df_plot = df_country.groupby("Date")["ConfirmedCases"].sum().reset_index()
    st.line_chart(df_plot.set_index("Date"))

    # Section: Predict daily new cases
    st.subheader(f"🔮 Predicted vs Actual Daily New Cases — {country}")
    model, X_test, y_test, y_pred = train_predictive_model(df_features, country=country)

    result_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    }).reset_index(drop=True)

    st.line_chart(result_df)

if __name__ == "__main__":
    main()
