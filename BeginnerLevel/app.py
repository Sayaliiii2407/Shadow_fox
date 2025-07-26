import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# @st.cache_data
def load_and_train_model():
    df = pd.read_csv("HousingData.csv")

    df_cleaned = df.copy()
    columns_to_clean = df.columns[df.dtypes == 'object']
    for col in columns_to_clean:
        df_cleaned[col] = df_cleaned[col].astype(str).str.replace('%', '', regex=False)
        df_cleaned[col] = df_cleaned[col].str.replace('?', '', regex=False)
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

    df_cleaned.dropna(inplace=True)


    X = df_cleaned.drop(columns=['Median Home Value'])
    y = df_cleaned['Median Home Value']

    model = LinearRegression()
    model.fit(X, y)

    return model, X.columns.tolist()

model, feature_names = load_and_train_model()


st.title("🏠 Interactive House Price Predictor")

import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 Housing Data Visualizations")


df = pd.read_csv("HousingData.csv")
df_cleaned = df.copy()
columns_to_clean = df.columns[df.dtypes == 'object']
for col in columns_to_clean:
    df_cleaned[col] = df_cleaned[col].astype(str).str.replace('%', '', regex=False)
    df_cleaned[col] = df_cleaned[col].str.replace('?', '', regex=False)
    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

df_cleaned.dropna(inplace=True)
st.subheader("📌 Preview of Cleaned Data")
st.write(df_cleaned.head())


st.write("Fill in the details below to get the estimated **median home value**.")

user_input = []
for feature in feature_names:
    val = st.number_input(f"{feature}", value=0.0)
    user_input.append(val)

if st.button("🔮 Predict House Price"):
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    st.success(f"🏡Estimated Median Home Value: $**{prediction:.2f}**")
