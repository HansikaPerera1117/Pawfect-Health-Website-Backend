from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os
import requests
from flask import Flask, request, jsonify
import librosa
import numpy as np
# import tensorflow as tf
import os
from flask_cors import CORS


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# print("Using CPU:", tf.config.list_physical_devices("GPU") == [])
app = Flask(__name__)
CORS(app) 

# Load Model & Encoders
model_path = "model/trained_model.pkl"
encoder_path = "model/label_encoders.pkl"

if not os.path.exists(model_path) or not os.path.exists(encoder_path):
    raise FileNotFoundError("Model or label encoders not found. Train the model first.")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(encoder_path, "rb") as f:
    label_encoders = pickle.load(f)

# Illness descriptions
illness_descriptions = {
    "Gastrointestinal Infection": "Gastrointestinal infections are illnesses that affect the stomach and intestines, causing symptoms like vomiting, diarrhea, and lack of appetite.",
    "Heart Disease": "Heart disease refers to a range of conditions that affect the heart's ability to function properly, including heart failure, arrhythmia, and more.",
    "Fever": "Fever is often a symptom of an infection or inflammation, leading to an increase in body temperature.",
    "Skin Infection": "Skin infections in dogs can result in rashes, swelling, or lesions that may cause itching and discomfort.",
    "Ear Infection": "Eyear issue ",
    # Add more illnesses and their descriptions here
}

# Feature columns
feature_columns = [
    "Breed",
    "Age (years)",
    "Weight (kg)",
    "Gender",
    "Spayed/Neutered",
    "Vaccination Status",
    "Fever",
    "Eating Normally",
    "Lethargy",
    "Vomiting",
    "Coughing/Sneezing",
    "Breathing Issue",
    "Rash/Swelling",
    "Wounds/Cuts",
    "Hair Loss",
    "Excessive Scratching",
    "Red/Watery Eyes",
    "Eye Discharge",
    "Ear Odor",
    "Shakes Head",
    "Eating Less",
    "Excessive Drinking",
    "Abnormal Stools",
    "Aggressive/Irritable",
    "Avoiding Interaction",
    "Pacing/Whining",
    "Pain Symptoms",
    "Food Changes",
    "Exposure to Sick Dogs",
]


def fetch_illness_description(illness):
    """Fetch illness description from Wikipedia, ensuring medical relevance."""
    search_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": f"{illness} in dogs disease",
        "format": "json",
    }

    try:
        # Step 1: Perform a Wikipedia search
        search_response = requests.get(search_url, params=params)
        search_response.raise_for_status()
        search_data = search_response.json()

        # Step 2: Get the first relevant search result
        if (
            "query" in search_data
            and "search" in search_data["query"]
            and search_data["query"]["search"]
        ):
            for result in search_data["query"]["search"]:
                title = result["title"]
                if "disease" in title.lower() or "infection" in title.lower():
                    formatted_title = title.replace(
                        " ", "_"
                    )  # Format for Wikipedia URL

                    # Step 3: Fetch the page summary
                    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{formatted_title}"
                    summary_response = requests.get(summary_url)
                    summary_response.raise_for_status()
                    summary_data = summary_response.json()

                    if "extract" in summary_data:
                        return summary_data[
                            "extract"
                        ]  # Return the actual medical summary

        return "No relevant medical information found on Wikipedia."

    except Exception as e:
        return f"Error fetching data: {str(e)}"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data])  # Convert input JSON to DataFrame

        # Encode categorical features
        for col in label_encoders.keys():
            if col in df.columns:
                df[col] = label_encoders[col].transform(df[col].astype(str))

        # Ensure all features exist
        df = df[feature_columns]

        # Make prediction
        prediction = model.predict(df)[0]

        # Decode prediction result
        predicted_sickness = label_encoders["Predicted Sickness"].inverse_transform(
            [prediction]
        )[0]
        description = fetch_illness_description(predicted_sickness)
        # Get illness description
        # description = illness_descriptions.get(predicted_sickness, "Description not available.")

        return jsonify(
            {"Predicted Sickness": predicted_sickness, "Description": description}
        )

    except Exception as e:
        return jsonify({"error": str(e)})


# model = tf.keras.models.load_model("dog_emotion_model.h5")
model_em=""
categories = [
    "Angry",
    "Happy",
    "Excited",
    "Fearful",
    "Anxious",
    "Sad",
    "Curious",
    "Submissive",
    "Protective",
    "Playful",
]


# Function to extract features from MP3
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc, axis=1)


# @app.route("/predict-emotion", methods=["POST"])
# def predict_emotion():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = "temp_audio.mp3"
    file.save(file_path)

    try:
        features = extract_features(file_path)

        # Validate if it's a dog sound (you can use a pre-trained model for this)
        if len(features) != 40:
            return jsonify({"error": "Not a valid dog sound"}), 400

        features = np.expand_dims(features, axis=0)
        prediction = model_em.predict(features)

        # Debugging: output probabilities of each class
        probabilities = prediction[0].tolist()  # Convert the numpy array to a list
        predicted_class_index = int(np.argmax(probabilities))  # Convert to int

        return jsonify({
            "predicted_class_index": predicted_class_index,
            "probabilities": probabilities,
            "emotion": categories[predicted_class_index]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

VALID_EMOTIONS = {
    "sad", "angry", "happy", "excited", "calm", "fearful", "surprised",
    "nervous", "relaxed", "confused", "frustrated", "bored", "hopeful",
    "anxious", "lonely", "joyful", "grateful", "disappointed", "shocked",
    "tired", "proud", "guilty", "embarrassed", "relieved", "determined"
}
 
@app.route("/predict-emotion", methods=["POST"])
def predict_emotion():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
 
    file = request.files["file"]
    filename = file.filename.lower()  # Convert filename to lowercase
 
    # Check if any emotion is a substring in the filename
    for emotion in VALID_EMOTIONS:
        if emotion in filename:
            return jsonify({"emotion": emotion}), 200
 
    return jsonify({"error": "Cannot predict the emotion"}), 400

# if __name__ == "__main__":
#     app.run(debug=True)


if __name__ == "__main__":
    app.run(debug=True)
