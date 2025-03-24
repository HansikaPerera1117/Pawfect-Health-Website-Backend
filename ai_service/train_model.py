# import pandas as pd
# import numpy as np
# import pickle
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import LabelEncoder
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# # Load Dataset
# data = pd.read_csv("data/dog_data.csv")

# # Check for missing values
# print("Missing Values:\n", data.isnull().sum())
# # Handle missing values
# data.fillna(method="ffill", inplace=True)  # Forward fill
# # Alternative: data.fillna(method="bfill", inplace=True)  # Backward fill
# # Handling missing values intelligently
# for col in data.columns:
#     if data[col].dtype == "object":  
#         # Fill categorical NaN with most frequent value (mode)
#         data[col].fillna(data[col].mode()[0], inplace=True)
#     else:  
#         # Fill numeric NaN with median
#         data[col].fillna(data[col].median(), inplace=True)

# # Encode categorical features
# label_encoders = {}
# for col in [
#     "Breed",
#     "Gender",
#     "Spayed/Neutered",
#     "Vaccination Status",
#     "Fever",
#     "Eating Normally",
#     "Lethargy",
#     "Vomiting",
#     "Coughing/Sneezing",
#     "Breathing Issue",
#     "Rash/Swelling",
#     "Wounds/Cuts",
#     "Hair Loss",
#     "Excessive Scratching",
#     "Red/Watery Eyes",
#     "Eye Discharge",
#     "Ear Odor",
#     "Shakes Head",
#     "Eating Less",
#     "Excessive Drinking",
#     "Abnormal Stools",
#     "Aggressive/Irritable",
#     "Avoiding Interaction",
#     "Pacing/Whining",
#     "Pain Symptoms",
#     "Food Changes",
#     "Exposure to Sick Dogs",
#     "Predicted Sickness",
# ]:
#     le = LabelEncoder()
#     data[col] = le.fit_transform(data[col])
#     label_encoders[col] = le

# # Save label encoders
# with open("model/label_encoders.pkl", "wb") as f:
#     pickle.dump(label_encoders, f)

# # Define Features and Target
# X = data.drop(columns=["Predicted Sickness"])  # Features
# y = data["Predicted Sickness"]  # Target

# # Split Data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Hyperparameter Tuning using GridSearchCV
# param_grid = {
#     "n_estimators": [100, 200, 300],
#     "max_depth": [10, 20, 30],
#     "min_samples_split": [2, 5, 10],
# }

# model = RandomForestClassifier(random_state=42)
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
# grid_search.fit(X_train, y_train)

# # Best model
# best_model = grid_search.best_estimator_

# # Model Evaluation
# y_pred = best_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Model Accuracy: {accuracy:.2f}")
# # Feature importance plot
# importances = best_model.feature_importances_
# feature_names = X.columns

# plt.figure(figsize=(12, 6))
# plt.barh(feature_names, importances, color="skyblue")
# plt.xlabel("Importance Score")
# plt.ylabel("Feature")
# plt.title("Feature Importance in RandomForest Model")
# plt.show()


# # Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot(cmap="Blues")
# plt.title("Confusion Matrix")
# plt.show()


# # Save the trained model
# with open("model/trained_model.pkl", "wb") as f:
#     pickle.dump(best_model, f)
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
 
# Load Dataset
data = pd.read_csv("data/dog_data.csv")
 
# Check for missing values
print("Missing Values:\n", data.isnull().sum())
 
# Encode categorical features
label_encoders = {}
for col in ["Breed", "Gender", "Spayed/Neutered", "Vaccination Status", "Fever", "Eating Normally", "Lethargy",
            "Vomiting", "Coughing/Sneezing", "Breathing Issue", "Rash/Swelling", "Wounds/Cuts", "Hair Loss",
            "Excessive Scratching", "Red/Watery Eyes", "Eye Discharge", "Ear Odor", "Shakes Head", "Eating Less",
            "Excessive Drinking", "Abnormal Stools", "Aggressive/Irritable", "Avoiding Interaction", "Pacing/Whining",
            "Pain Symptoms", "Food Changes", "Exposure to Sick Dogs", "Predicted Sickness"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
 
# Save label encoders
with open("model/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)
 
# Define Features and Target
X = data.drop(columns=["Predicted Sickness"])  # Features
y = data["Predicted Sickness"]  # Target
 
# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Hyperparameter Tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
 
model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_search.fit(X_train, y_train)
 
# Best model
best_model = grid_search.best_estimator_
 
# Model Evaluation
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
 
# Save the trained model
with open("model/trained_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
 