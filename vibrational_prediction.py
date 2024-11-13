# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt

# Load data
vibrational_data = pd.read_csv('Vibrational_Data.csv')
molecular_data = pd.read_excel('Molecular_Coordinates.xlsx')

# Data Aggregation and Merging
# Aggregate molecular data by molecule
molecular_agg = molecular_data.groupby('Molecule').agg({
    'X': 'mean',
    'Y': 'mean',
    'Z': 'mean',
    'Number of Bonds': 'sum',
    'Dipole Moment': 'first',
    'Point Group': 'first'
}).reset_index()

# Aggregate vibrational data by molecule
vibrational_agg = vibrational_data.groupby('Molecule').agg({
    'Frequency (cm-1)': list,
    'IR Intensity': list
}).reset_index()

# Merge datasets on 'Molecule'
merged_data = pd.merge(molecular_agg, vibrational_agg, on='Molecule', how='inner')

# Encode the 'Point Group' feature
label_encoder = LabelEncoder()
merged_data['Point Group Encoded'] = label_encoder.fit_transform(merged_data['Point Group'])

# Prepare Features (X) and Targets (y)
X = merged_data[['X', 'Y', 'Z', 'Number of Bonds', 'Dipole Moment', 'Point Group Encoded']]
target_length = 50  # Set target length for consistent output

# Pad or truncate frequency and intensity data
y_frequencies = np.array([np.pad(freq, (0, max(0, target_length - len(freq))), 'constant')[:target_length] 
                          for freq in merged_data['Frequency (cm-1)']])
y_intensities = np.array([np.pad(intensity, (0, max(0, target_length - len(intensity))), 'constant')[:target_length] 
                          for intensity in merged_data['IR Intensity']])

# Split data into training and testing sets
X_train, X_test, y_train_frequencies, y_test_frequencies = train_test_split(X, y_frequencies, test_size=0.2, random_state=42)
_, _, y_train_intensities, y_test_intensities = train_test_split(X, y_intensities, test_size=0.2, random_state=42)

# Train MultiOutputRegressor models for frequency and intensity prediction
frequency_model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
intensity_model = MultiOutputRegressor(RandomForestRegressor(random_state=42))

frequency_model.fit(X_train, y_train_frequencies)
intensity_model.fit(X_train, y_train_intensities)

# Function to predict and plot for a given molecule
def predict_and_plot(molecule_name):
    # Find the molecule in the data
    molecule_index = merged_data[merged_data['Molecule'] == molecule_name].index[0]
    X_molecule = pd.DataFrame([X.iloc[molecule_index]], columns=X.columns)  # Ensure feature names are included
    
    # Predict frequency and intensity
    predicted_frequencies = frequency_model.predict(X_molecule)[0]
    predicted_intensities = intensity_model.predict(X_molecule)[0]
    
    # Actual values for comparison
    actual_frequencies = np.array(merged_data.loc[molecule_index, 'Frequency (cm-1)'])
    actual_intensities = np.array(merged_data.loc[molecule_index, 'IR Intensity'])
    
    # Scale predicted intensities to match the actual intensity range
    max_actual_intensity = actual_intensities.max()
    scaled_predicted_intensities = (predicted_intensities / predicted_intensities.max()) * max_actual_intensity
    
    # Pad actual values to match plot length
    actual_frequencies = np.pad(actual_frequencies, (0, target_length - len(actual_frequencies)), 'constant')
    actual_intensities = np.pad(actual_intensities, (0, target_length - len(actual_intensities)), 'constant')

    # Plot the actual vs predicted IR Intensity vs Frequency graph
    plt.figure(figsize=(10, 6))
    plt.stem(actual_frequencies, actual_intensities, linefmt="C0-", markerfmt="C0o", basefmt=" ", label="Actual Intensity")
    plt.stem(predicted_frequencies, scaled_predicted_intensities, linefmt="C1--", markerfmt="C1x", basefmt=" ", label="Scaled Predicted Intensity")
    plt.title(f"IR Intensity vs Frequency for {molecule_name}")
    plt.xlabel("Frequency (cm⁻¹)")
    plt.ylabel("IR Intensity")
    plt.legend()
    plt.show()

# Example usage
predict_and_plot("CNH3O")
