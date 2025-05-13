import os
import joblib
import cv2
import numpy as np

# Load model and label dictionary
model = joblib.load("model/plant_disease_multiclass_model.pkl")
label_dict = joblib.load("model/label_dict.pkl")

def extract_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    hist = cv2.calcHist([image], [0, 1, 2], None,
                        [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def test_model_file_exists():
    assert os.path.exists("model/plant_disease_multiclass_model.pkl"), "ML model file missing!"

def test_label_dict_exists():
    assert os.path.exists("model/label_dict.pkl"), "Label dictionary file missing!"

def test_model_prediction_on_sample():
    test_image_path = "tests/sample_leaf.jpg"
    assert os.path.exists(test_image_path), "Sample image not found in tests/ folder!"

    features = extract_features(test_image_path)
    prediction = model.predict([features])[0]
    assert prediction in label_dict, "Prediction not in label dictionary!"
