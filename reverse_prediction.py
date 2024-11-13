import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image

# Load data
vibrational_data = pd.read_csv('C:\\Users\\Adithya\\OneDrive\\Desktop\\new_chem_python\\Vibrational_Data.csv')
molecular_coordinates = pd.read_excel('C:\\Users\\Adithya\\OneDrive\\Desktop\\new_chem_python\\Molecular_Coordinates.xlsx')
molecule_images = pd.read_excel('C:\\Users\\Adithya\\OneDrive\\Desktop\\new_chem_python\\molecules_with_actual_image_paths(1).xlsx')

# Encoding and feature preparation
label_encoder_pg = LabelEncoder()
label_encoder_atom = LabelEncoder()
molecular_coordinates['Point Group Encoded'] = label_encoder_pg.fit_transform(molecular_coordinates['Point Group'])
molecular_coordinates['Dominant Atom Encoded'] = label_encoder_atom.fit_transform(molecular_coordinates['dominant_atom'])

# Aggregate vibrational data
vibrational_aggregated = vibrational_data.groupby('Molecule').agg({
    'Frequency (cm-1)': ['mean', 'median', 'std'],
    'IR Intensity': ['mean', 'median', 'std']
}).reset_index()
vibrational_aggregated.columns = ['Molecule', 'Freq_Mean', 'Freq_Median', 'Freq_Std', 'IR_Mean', 'IR_Median', 'IR_Std']

# Merge data
vibrational_full_data = pd.merge(vibrational_aggregated,
                                 molecular_coordinates[['Molecule', 'Dipole Moment', 'Point Group Encoded', 'Dominant Atom Encoded']].drop_duplicates(),
                                 on='Molecule', how='left')
vibrational_full_data.fillna(vibrational_full_data.select_dtypes(include='number').median(), inplace=True)

# Feature scaling and weighting
scaler = StandardScaler()
X_final = scaler.fit_transform(vibrational_full_data[['Freq_Mean', 'Freq_Median', 'Freq_Std', 'IR_Mean', 'IR_Median', 'IR_Std', 
                                                      'Dipole Moment', 'Point Group Encoded', 'Dominant Atom Encoded']])
X_final[:, 7] *= 2  # Emphasizing Point Group
X_final[:, 8] *= 2  # Emphasizing Dominant Atom

# Model preparation
y_final = vibrational_full_data['Molecule']
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.3, random_state=42)

# Training the classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Making predictions and storing results
y_pred = rf_model.predict(X_test)

# Creating DataFrame for actual vs predicted results
results_df = pd.DataFrame({
    'Actual Molecule': y_test.values,
    'Predicted Molecule': y_pred
})

# Adding coordinates of the predicted molecules
coordinates_list = []
for molecule in y_pred:
    coords = molecular_coordinates[molecular_coordinates['Molecule'] == molecule][['Atom', 'X', 'Y', 'Z']].reset_index(drop=True)
    coordinates_list.append(coords)

results_df['Predicted Molecule Coordinates'] = coordinates_list

# Adding image path to the predicted results based on the image file
results_df = results_df.merge(molecule_images, left_on='Predicted Molecule', right_on='Molecule', how='left')

# Function to display a specific test case with image
def display_test_case_with_image(test_index):
    if test_index >= len(results_df):
        print("Invalid index! Please select a valid test case index.")
        return

    row = results_df.iloc[test_index]
    print(f"Actual Molecule: {row['Actual Molecule']}")
    print(f"Predicted Molecule: {row['Predicted Molecule']}")
    print("Predicted Molecule Coordinates:")
    print(row['Predicted Molecule Coordinates'])
    
    # Display the image if available
    image_path = row['Image (Link)']  # Replace with actual column name if different
    if pd.notna(image_path):
        img = Image.open(image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Predicted Molecule: {row['Predicted Molecule']}")
        plt.show()
    else:
        print("Image not available for this molecule.")

# Example usage - Replace '0' with the desired index
display_test_case_with_image(3)