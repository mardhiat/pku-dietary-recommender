import joblib
import pandas as pd
import streamlit as st

# Load the trained model and the fitted scaler
model = joblib.load("pku_severity_model.pkl")  # Load the trained model
scaler = joblib.load("scaler.pkl")  # Load the fitted scaler

# Streamlit app setup
st.title("PKU Severity Predictor")

# Input features via Streamlit sidebar
st.sidebar.header("Input Patient Data")
age = st.sidebar.slider("Age", 1, 50, 25)
phe_level = st.sidebar.slider("Phe Level (µmol/L)", 100, 2400, 500)
adherence = st.sidebar.slider("Adherence (%)", 50, 100, 80)
kuvan_use = st.sidebar.selectbox("Kuvan Use", ["Yes", "No"])

# Prepare the input data for prediction
input_data = pd.DataFrame({
    "Age": [age],
    "Phe Level (µmol/L)": [phe_level],
    "Adherence (%)": [adherence],
    "Kuvan Use": [1 if kuvan_use == "Yes" else 0]
})

# Scale the input data using the loaded scaler
scaled_data = scaler.transform(input_data[['Age', 'Phe Level (µmol/L)', 'Adherence (%)']])

# Create a copy of input_data to store the scaled values
input_data_scaled = input_data.copy()
input_data_scaled[['Age', 'Phe Level (µmol/L)', 'Adherence (%)']] = scaled_data

# Make prediction when button is pressed
if st.sidebar.button("Predict"):
    prediction = model.predict(input_data_scaled.drop(columns=["Kuvan Use"]))  # Drop 'Kuvan Use' since it's not a feature for prediction
    st.write("Predicted PKU Severity:", prediction[0])
