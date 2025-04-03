import os       #streamlit run UI.py
import sys
sys.path.append(os.path.abspath('..'))
from src import config
import streamlit as st
import pickle

# Load the models
with open(os.path.join(config.MODELS_PATH, "random_forest_l.pickle"), "rb") as file:
    modelRF_l = pickle.load(file)

with open(os.path.join(config.MODELS_PATH, "random_forest_a.pickle"), "rb") as file:
    modelRF_a = pickle.load(file)

st.title("House Price Prediction")

# Step 1: User selects the type of input
selectedModel = st.selectbox(
    'Quale modello vorresti utilizzare per la previsione?',
    ('Latitudine-Longitudine', 'eta, distanza e numero negozi')
)

# Step 2: User provides input values
if selectedModel == 'Latitudine-Longitudine':
    latitude = st.text_input("inserire Latitude (utilizzare . per decimali):")
    longitude = st.text_input("inserire Longitude (utilizzare . per decimali):")
elif selectedModel == 'eta, distanza e numero negozi':
    età_casa = st.text_input("inserire anni dalla data di costruzione:")
    dist_MRT = st.text_input("inserire distanza dalla metro più vicina (valore in m):")
    n_negozi_convenienti = st.text_input("inserire n. minimarket vicini:")

# Step 3: Predict when button is clicked
if st.button("Predict"):
    try:
        if selectedModel == 'Latitudine-Longitudine':
            # Ensure inputs are valid
            if not latitude or not longitude:
                st.warning("Please enter both latitude and longitude.")
            else:
                # Convert inputs to float
                latitude = float(latitude)
                longitude = float(longitude)
                # Prepare input for the model
                x = [[latitude, longitude]]
                # Predict using the Random Forest model
                prediction = modelRF_l.predict(x)[0]
                st.success(f"Predicted price: {prediction}")
        elif selectedModel == 'eta, distanza e numero negozi':
            # Ensure inputs are valid
            if not età_casa or not dist_MRT or not n_negozi_convenienti:
                st.warning("Please enter all three features.")
            else:
                # Convert inputs to float
                età_casa = float(età_casa)
                dist_MRT = float(dist_MRT)
                n_negozi_convenienti = float(n_negozi_convenienti)
                # Prepare input for the model
                x = [[età_casa, dist_MRT, n_negozi_convenienti]]
                # Predict using the Random Forest model
                prediction = modelRF_a.predict(x)[0]
                st.success(f"Predicted price: {prediction}")
    except ValueError:
        st.error("Please enter valid numeric values.")