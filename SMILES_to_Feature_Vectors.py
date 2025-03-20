import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit import DataStructs
import numpy as np
from tqdm import tqdm

# Function to calculate chemical descriptors and molecular fingerprints
def featurize_molecule(smiles):
    """
    Computes the chemical descriptors, molecular fingerprints, and SMILES string length for a given molecule.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Failed to parse SMILES: {smiles}")
        
        # Compute chemical descriptors
        descriptors = [desc[1](mol) for desc in Descriptors._descList]
        
        # Compute molecular fingerprints
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fingerprint_array = np.zeros((2048,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
        
        # Compute SMILES string length
        smiles_length = len(smiles)
        
        # Combine all features
        features = np.concatenate([descriptors, fingerprint_array, [smiles_length]])
        return features
    except Exception as e:
        raise ValueError(f"Feature extraction failed for SMILES: {smiles}, Error: {e}")

# Batch processing of SMILES
def process_smiles(smiles_list):
    """
    Processes a batch of SMILES strings and computes their corresponding feature vectors.
    """
    results = []
    failed_smiles = []  # Store SMILES that failed processing

    for smiles in tqdm(smiles_list, desc="Processing SMILES"):
        try:
            features = featurize_molecule(smiles)
            if features is not None:
                # Convert the feature vector into a single string
                features_string = ",".join(map(str, features))
                results.append(features_string)
        except Exception as e:
            failed_smiles.append(smiles)
            print(f"Failed to process SMILES: {smiles}, Error: {e}")

    return results, failed_smiles

# Save the results as a CSV file
def save_features_to_csv(results, file_path):
    """
    Saves the feature vectors as a CSV file.
    """
    df = pd.DataFrame(results, columns=["Total_Feature_Vector"])
    df.to_csv(file_path, index=False, encoding='utf-8')
    print(f"Feature vectors saved to: {file_path}")

# Main function
def main():
    # Read SMILES information from an Excel file
    input_excel = "F:\\ML Prediction Project\\SMILES.xlsx"  # Specify the path to the Excel file, Replace your own
    smiles_column = "SMILES"  # Specify the column containing SMILES strings, Replace your own
    smiles_df = pd.read_excel(input_excel)
    
    if smiles_column not in smiles_df.columns:
        raise ValueError(f"The specified column '{smiles_column}' was not found in the Excel file. Please check the file structure.")
    
    smiles_list = smiles_df[smiles_column].dropna().tolist()
    
    # Process SMILES and compute features
    results, failed_smiles = process_smiles(smiles_list)

    # Specify the output path
    output_csv = "F:\\ML Prediction Project\\chemical_features_output.csv"  # Specify the path for saving the output, Replace your own
    save_features_to_csv(results, output_csv)

    # Print failed SMILES strings, if any
    if failed_smiles:
        print("\nThe following SMILES could not be processed:")
        for smiles in failed_smiles:
            print(smiles)

if __name__ == "__main__":
    main()
