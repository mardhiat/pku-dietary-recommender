import streamlit as st
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('final_model.h5')

# Define the prediction function
def predict_pku(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0][0]

# Streamlit app
st.title("PKU Severity Predictor")
st.write("Predict the severity of PKU based on phenotype data.")

# User input form
st.header("Input Patient Data")
num_features = 5  # Adjust this based on your model's input features
inputs = []

for i in range(num_features):
    value = st.number_input(f"Feature {i+1}", step=0.01, format="%.2f")
    inputs.append(value)

# Predict and display results
if st.button("Predict"):
    if all(inputs):
        result = predict_pku(inputs)
        st.success(f"Predicted Severity: {result:.2f}")
    else:
        st.error("Please fill in all the inputs.")
