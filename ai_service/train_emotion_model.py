# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# import matplotlib.pyplot as plt

# # Load dataset
# df = pd.read_csv("data/dog_bark_emotions_large.csv")  # Use the correct path

# # Check for missing values
# print("Missing values:", df.isnull().sum().sum())

# # Extract features and labels
# X = df.drop(columns=["label"]).values  # MFCC features
# y = df["label"].values  # Labels

# # Normalize features (Standard Scaling)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Encode labels as numerical values
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# # Split dataset (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled, y_encoded, test_size=0.2, random_state=42
# )

# print("Training set shape:", X_train.shape)
# print("Testing set shape:", X_test.shape)


# # Define model
# model = Sequential(
#     [
#         Dense(128, activation="relu", input_shape=(40,)),
#         Dropout(0.3),
#         Dense(64, activation="relu"),
#         Dense(10, activation="softmax"),  # 10 classes
#     ]
# )

# # Compile model
# model.compile(
#     optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
# )

# # Train model
# history = model.fit(
#     X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test)
# )

# # Save the model
# model.save("dog_emotion_model.h5")


# # Plot training & validation accuracy
# plt.plot(history.history["accuracy"], label="Train Accuracy")
# plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.title("Model Accuracy")
# plt.show()

# # Plot training & validation loss
# plt.plot(history.history["loss"], label="Train Loss")
# plt.plot(history.history["val_loss"], label="Validation Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()
# plt.title("Model Loss")
# plt.show()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/dog_bark_emotions.csv")  # Use the correct path

# Check for missing values
print("Missing values:", df.isnull().sum().sum())

# Extract features and labels
X = df.drop(columns=["label"]).values  # MFCC features
y = df["label"].values  # Labels

# Normalize features (Standard Scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode labels as numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Check class distribution to see if imbalance exists
class_distribution = pd.Series(y_encoded).value_counts()
print("Class distribution:\n", class_distribution)

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Define model
model = Sequential([
    Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.5),  # Increase dropout to avoid overfitting
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(len(np.unique(y_encoded)), activation="softmax")  # Dynamically adjust output size
])

# Compile model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Add Early Stopping callback to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
history = model.fit(
    X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

# Save the model
model.save("dog_emotion_model.h5")

# Plot training & validation accuracy
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Model Accuracy")
plt.show()

# Plot training & validation loss
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Model Loss")
plt.show()
