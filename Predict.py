import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit import DataStructs
import joblib

# Load the trained model and scaler
model_path = "F:\\ML Prediction Project\\Source Code\\CuMOF_XGBoost_best_model.pkl" # Replace your own
scaler_path = "F:\\ML Prediction Project\\Source Code\\scaler.pkl" # Replace your own
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Load label mappings
label_mapping = {'Paddle-wheel': 0, 'Other': 1, 'rod': 2}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Function to compute molecular feature vector
def featurize_smiles(smiles):
    """
    Converts a SMILES string into a feature vector for model input.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # Calculate chemical descriptors
        descriptors = [desc[1](mol) for desc in Descriptors._descList]
        
        # Calculate molecular fingerprints
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fingerprint_array = np.zeros((2048,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
        
        # Calculate SMILES string length
        smiles_length = len(smiles)
        
        # Combine features
        features = np.concatenate([descriptors, fingerprint_array, [smiles_length]])
        return features
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return None

# Prediction function
def predict_smiles(smiles):
    """
    Takes a SMILES string as input and returns the predicted label.
    """
    # Compute features
    features = featurize_smiles(smiles)
    if features is None:
        return f"Prediction failed for SMILES '{smiles}'"
    
    # Standardize features
    features_scaled = scaler.transform([features])
    
    # Perform model prediction
    prediction = model.predict(features_scaled)[0]
    predicted_label = reverse_label_mapping[prediction]
    return predicted_label

# Example inputs
example_smiles = [
    "O=C(O)C=1C=CC(=CC1)C2=C(C(C=3C=CC(=CC3)C(=O)O)=C(C(C=4C=CC(=CC4)C(=O)O)=C2C)C)C",  # Example 1
    "O=C(O)C=1C=CC(=CC1)C2=C(C3=CC=C(C=C3)C(C)(C)C)C(C=4C=CC(=CC4)C(=O)O)=C(C5=CC=C(C=C5)C(C)(C)C)C(C=6C=CC(=CC6)C(=O)O)=C2C7=CC=C(C=C7)C(C)(C)C"  # Example 2
]

# Batch predictions
print("Prediction results:")
for smiles in example_smiles:
    result = predict_smiles(smiles)
    print(f"SMILES: {smiles} -> Predicted Type: {result}")
