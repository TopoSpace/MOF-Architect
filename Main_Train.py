import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
import joblib
from collections import Counter

# Load dataset
file_path = "F:\\ML Prediction Project\\chemical_features_output.csv" # Replace your own
data = pd.read_csv(file_path)

# Parse the 'Total_Feature_Vector' column into numerical vectors
data['Total_Feature_Vector'] = data['Total_Feature_Vector'].apply(
    lambda x: np.array(list(map(float, x.split(','))))
)

# Extract features and labels
X = np.stack(data['Total_Feature_Vector'].values)
y = data['Node_Type']

# Convert categorical labels to numerical encoding
label_mapping = {label: idx for idx, label in enumerate(y.unique())}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}
y = y.map(label_mapping)

# Print label mappings
print("Label Mapping:", label_mapping)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle any invalid numerical values
X_train = np.nan_to_num(X_train, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)

# Clip feature values to prevent extreme outliers
X_train = np.clip(X_train, -1e6, 1e6)
X_test = np.clip(X_test, -1e6, 1e6)

# Examine the distribution of the original training data
original_counts = Counter(y_train)
print("Original training data distribution:", original_counts)

# Define a custom sampling strategy
sampling_strategy = {
    0: original_counts[0],  # Retain the number of samples for 'Paddle-wheel'
    1: original_counts[1],  # Retain the number of samples for 'Other'
    2: original_counts[2]   # Retain the number of samples for 'rod'
}

# Apply SMOTETomek for data balancing
print("Applying SMOTETomek with custom sampling strategy...")
smote_tomek = SMOTETomek(sampling_strategy=sampling_strategy, random_state=42)
X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)

# Display the distribution after balancing
print("Balanced training data distribution:", Counter(y_train_resampled))

# Feature scaling using StandardScaler
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Save the scaler for future use
scaler_path = "F:\\ML Prediction Project\\scaler.pkl" # Replace your own
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to: {scaler_path}")

# Define the hyperparameter grid for optimization
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Initialize the XGBoost classifier
xgb = XGBClassifier(
    random_state=42,
    eval_metric='mlogloss'
)

# Perform hyperparameter tuning using GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=1,
    n_jobs=-1
)

# Execute the grid search
print("Starting Grid Search...")
grid_search.fit(X_train_resampled, y_train_resampled)

# Output the best parameters and scores
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Train the final model using the best parameters
best_model = grid_search.best_estimator_

# Evaluate the model on the test set
y_pred = best_model.predict(X_test)

# Compute overall performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

# Output overall performance metrics
print("\nOverall Model Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")

# Generate and display the classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_mapping.keys()))

# Save the trained model
model_path = "F:\\ML Prediction Project\\CuMOF_XGBoost_best_model.pkl" # Replace your own
joblib.dump(best_model, model_path)
print(f"Best Model saved to: {model_path}")
