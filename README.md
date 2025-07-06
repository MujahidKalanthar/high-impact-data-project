# 🦠 COVID-19 Analytical Dashboard & Predictor

An end-to-end data analytics platform that ingests real-world COVID-19 data, cleans it, builds useful features, trains a predictive ML model, and visualizes it all in an interactive Streamlit dashboard.

🔗 **Live App:** [Click here to view](https://high-impact-data-project-jqfdxdtwngvhcwbhm9gxz3.streamlit.app/)

---

## 💡 Features

- 📊 Ingests daily COVID data from Johns Hopkins University
- 🧼 Automated data cleaning & transformation
- 🔍 Feature engineering: new cases, lag variables, date-based signals
- 🔮 Predicts daily new COVID-19 cases per country (Random Forest)
- 📈 Interactive visualizations and prediction results using Streamlit

---

## 🛠️ Tech Stack

- Python 3.x
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit
- Git + GitHub

---

## 📁 Folder Structure

high-impact-data-project/
│
├── app.py # Main Streamlit app
├── requirements.txt # Package list
├── README.md # Project overview
└── src/
├── data_pipeline.py # Data ingestion, cleaning, feature engineering
└── modeling.py # Model training & prediction

---

## 🚀 Run Locally

```bash
git clone https://github.com/MujahidKalanthar/high-impact-data-project.git
cd high-impact-data-project
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
