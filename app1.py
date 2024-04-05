import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

import os

# Get the absolute path to the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'model_pickle1.pkl')

# Open the file
with open(file_path, 'rb') as file:
# # Load the trained model
# with open('model_pickle1.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to preprocess input data
def preprocess_input(data):
    # Perform label encoding for categorical variables
    label_encoder = LabelEncoder()
    for col in data.columns:
        if data[col].dtype == 'O':
            data[col] = label_encoder.fit_transform(data[col])
    return data

# Function to predict habitability score
def predict_habitability_score(input_data):
    input_df = pd.DataFrame(input_data, index=[0])
    input_df = preprocess_input(input_df)
    
    # Get feature names
    feature_names = input_df.columns.tolist()
    
    # Convert input data to numpy array
    input_array = input_df.to_numpy()
    
    # Predict habitability score
    prediction = model.predict(input_array)
    return prediction[0]


# UI layout
st.title('Habitability Score Predictor')

# Input form
st.header('Enter Property Details')

# Define input options for categorical columns
property_types = ['Apartment', 'Bungalow', 'Single-family home', 'Duplex', 'Container Home']
furnishing_options = ['Semi_Furnished', 'Unfurnished', 'Fully Furnished']
power_backup_options = ['No', 'Yes', 'NOT MENTIONED']
water_supply_options = ['Once in a day - Morning', 'Once in a day - Evening', 'All time', 'NOT MENTIONED', 'Once in two days']
crime_rate_options = ['Slightly below average', 'Well below average', 'Well above average', 'Slightly above average']
dust_and_noise_options = ['Medium', 'High', 'Low']

# Input fields for categorical columns
property_type = st.selectbox('Property Type', options=property_types, index=0)
furnishing = st.selectbox('Furnishing', options=furnishing_options, index=0)
power_backup = st.selectbox('Power Backup', options=power_backup_options, index=0)
water_supply = st.selectbox('Water Supply', options=water_supply_options, index=0)
crime_rate = st.selectbox('Crime Rate', options=crime_rate_options, index=0)
dust_and_noise = st.selectbox('Dust and Noise', options=dust_and_noise_options, index=0)

# Input fields for numerical columns
property_area = st.number_input('Property Area', value=0, step=1)
number_of_windows = st.number_input('Number of Windows', value=0, step=1)
number_of_doors = st.number_input('Number of Doors', value=0, step=1)
frequency_of_powercuts = st.number_input('Frequency of Powercuts', value=0, step=1)
traffic_density_score = st.number_input('Traffic Density Score', value=0, step=1)
air_quality_index = st.number_input('Air Quality Index', value=0, step=1)
neighborhood_review = st.number_input('Neighborhood Review', value=0, step=1)

input_data = {
    'property_type': property_type,
    'furnishing': furnishing,
    'power_backup': power_backup,
    'water_supply': water_supply,
    'crime_rate': crime_rate,
    'dust_and_noise': dust_and_noise,
    'property_area': property_area,
    'number_of_windows': number_of_windows,
    'number_of_doors': number_of_doors,
    'frequency_of_powercuts': frequency_of_powercuts,
    'traffic_density_score': traffic_density_score,
    'air_quality_index': air_quality_index,
    'neighborhood_review': neighborhood_review
}

# Predict button
if st.button('Predict'):
    # Predict habitability score
    prediction = predict_habitability_score(input_data)
    st.success(f'Predicted Habitability Score: {prediction:.2f}')
