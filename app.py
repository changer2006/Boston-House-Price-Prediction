import streamlit as st
import pandas as pd
import joblib

model = joblib.load("boston_model.pkl")

st.title("🏠 Boston House Price Prediction")

rm = st.number_input("Average Rooms (RM)", value=6.0)
lstat = st.number_input("LSTAT", value=10.0)
nox = st.number_input("NOX", value=0.5)
tax = st.number_input("TAX", value=300)
ptratio = st.number_input("PTRATIO", value=15.0)

if st.button("Predict House Price"):

    input_data = pd.DataFrame({
        "indus":[5],
        "chas":[0],
        "nox":[nox],
        "rm":[rm],
        "age":[50],
        "dis":[4],
        "rad":[5],
        "tax":[tax],
        "ptratio":[ptratio],
        "lstat":[lstat]
    })

    prediction = model.predict(input_data)

    st.success(
        f"Predicted House Price: {prediction[0]:.2f}"
    )