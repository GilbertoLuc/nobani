import os       #streamlit run UI.py
import sys
sys.path.append(os.path.abspath('..'))
from src import config
import streamlit as st
import pickle

# Load the model and vectorizer (random forest e logistico)
with open(os.path.join(config.MODELS_PATH, "random_forest.pickle"), "rb") as file:
    modelRF = pickle.load(file)

with open(f"{config.MODELS_PATH}vectorizerRF.pickle", "rb") as f:
    vectorizerRF = pickle.load(f)

with open(os.path.join(config.MODELS_PATH, "logistic_regression.pickle"), "rb") as file:
    modelLR = pickle.load(file)

with open(f"{config.MODELS_PATH}vectorizerLR.pickle", "rb") as f:
    vectorizerLR = pickle.load(f)

selectedModel = st.selectbox(
         'Quale modello vorresti utilizzare per la sentiment analysis?',
         ('Random Forest', 'Logistic Regression'))

st.title("Text Classification")

# Text input
user_input = st.text_area("Enter text to classify", "")

# Predict when button is clicked
if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform input and predict
        if selectedModel == "Random Forest":
            x = vectorizerRF.transform([user_input])
            prediction = modelRF.predict(x)[0]
        elif selectedModel == "Logistic Regression":
            x = vectorizerLR.transform([user_input])
            prediction = modelLR.predict(x)[0]             
        if prediction == "positive":
            st.success(f"Predicted class: {prediction}")
        elif prediction == 'negative':
            st.warning(f"Predicted class: {prediction}")
