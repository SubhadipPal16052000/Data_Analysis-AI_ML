import streamlit as st
import pandas as pd
import joblib  #

# Load the trained Random Forest model
try:
    model = joblib.load("E:\PYCHARM_PROJ\Breast_Cancer_Prediction_Model.joblib")  # Adjust the filename if needed
except FileNotFoundError:
    st.error("Error: Trained model file not found. Please make sure 'breast_cancer_model.joblib' is in the same directory.")
    st.stop()

# Define the features that the model expects
feature_names = ['radius_mean', 'mean texture', 'mean perimeter', 'mean area',
                 'mean smoothness', 'mean compactness', 'mean concavity',
                 'mean concave points', 'mean symmetry', 'mean fractal dimension',
                 'radius error', 'texture error', 'perimeter error', 'area error',
                 'smoothness error', 'compactness error', 'concavity error',
                 'concave points error', 'symmetry error', 'fractal dimension error',
                 'worst radius', 'worst texture', 'worst perimeter', 'worst area',
                 'worst smoothness', 'worst compactness', 'worst concavity',
                 'worst concave points', 'worst symmetry'] #'worst fractal dimension']

st.title('Breast Cancer Prediction WebApp')
st.write('Please enter the medical measurements to predict if the tumor is benign or malignant.')
st.write('This BCP_web application Represented by SUBHADIP_PAL')

# Create input fields for each feature
features = {}
for name in feature_names:
    features[name] = st.number_input(f'{name}:', format="%.4f")

# Create a DataFrame from the user input
input_df = pd.DataFrame([features])

# Make prediction when the user clicks the button
if st.button('Predict'):
    if model is not None:
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[:, 1]  # Probability of being malignant

        st.subheader('Prediction Result:')
        if prediction[0] == 0:
            st.success('The model predicts the tumor is Benign.')
        else:
            st.error('The model predicts the tumor is Malignant.')

        st.write(f'Probability of being Malignant: {probability[0]:.4f}')
    else:
        st.warning("Model not loaded. Please check the file path.")