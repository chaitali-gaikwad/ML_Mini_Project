from flask import Flask, render_template, request
import numpy as np
import joblib
import warnings
from sklearn.preprocessing import LabelEncoder

# Suppress all warnings (not recommended unless you're sure)
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the trained model
model = joblib.load('C:/Users/admin/Desktop/ML_Mini_Project/Python codes (ipynb files)/model_pickle1.pkl')

# Define label encoders for categorical variables
label_encoders = {}
categorical_features = ['property_type', 'furnishing', 'power_backup', 'water_supply', 'crime_rate', 'dust_and_noise']

for feature in categorical_features:
    label_encoders[feature] = LabelEncoder()

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    property_area = float(request.form['property_area'])
    num_windows = int(request.form['number_of_windows'])
    num_doors = int(request.form['number_of_doors'])
    freq_powercuts = float(request.form['frequency_of_powercuts'])
    traffic_density = float(request.form['traffic_density_score'])
    air_quality_index = float(request.form['air_quality_index'])
    neighborhood_review = float(request.form['neighborhood_review'])

    # Convert categorical variables to numerical values
    features = []
    for feature in categorical_features:
        label_encoder = label_encoders[feature]
        encoded_value = label_encoder.transform([request.form[feature]])[0]
        features.append(encoded_value)

    # Add numerical features
    features.extend([property_area, num_windows, num_doors, freq_powercuts, traffic_density,
                     air_quality_index, neighborhood_review])

    # Convert to numpy array
    features_array = np.array(features).reshape(1, -1)

    # Make prediction
    habitability_score = model.predict(features_array)[0]

    return render_template('result.html', habitability_score=habitability_score)

if __name__ == '__main__':
    # Fit label encoders on dummy data to avoid NotFittedError
    dummy_data = {'property_type': ['Apartment', 'Bungalow', 'Single-family home', 'Duplex', 'Container Home'],
                  'furnishing': ['Semi_Furnished', 'Unfurnished', 'Fully Furnished'],
                  'power_backup': ['No', 'Yes'],
                  'water_supply': ['Once in a day - Morning', 'Once in a day - Evening', 'All time', 'Once in two days'],
                  'crime_rate': ['Slightly below average', 'Well below average', 'Well above average', 'Slightly above average'],
                  'dust_and_noise': ['Medium', 'High', 'Low']}
    
    for feature, values in dummy_data.items():
        label_encoders[feature].fit(values)

    app.run(debug=True, port=5500)
