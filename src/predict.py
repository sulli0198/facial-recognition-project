import tensorflow as tf
import numpy as np
from PIL import Image
import os
from tensorflow import keras

# --- Configuration ---
# These must match the settings used during training
IMAGE_HEIGHT = 250
IMAGE_WIDTH = 250

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'model', 'face_recognition_model.h5')

# Class labels (must match your folder names during training)
CLASSES = ['me', 'not_me']

def load_and_preprocess_image(image_path):
    """
    Loads an image from a given path, resizes it, and prepares it for the model.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Load the image
    img = Image.open(image_path)

    # Resize to match training dimensions
    img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))

    # Convert to NumPy array
    img_array = np.asarray(img, dtype=np.float32)

    # Add batch dimension (model expects shape: (1, height, width, channels))
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize to match training (0–1 range)
    img_array /= 255.0

    return img_array

def main():
    """
    Loads the trained model, takes user input for image path,
    makes prediction, and prints results.
    """
    # --- Step 1: Load trained model ---
    try:
        print(f"Loading model from: {MODEL_PATH}")
        model = keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Step 2: Prompt for image path ---
    print("\nEnter the full path to the image you want to predict:")
    image_to_predict_path = input(">>> ")

    # --- Step 3: Preprocess image ---
    try:
        preprocessed_image = load_and_preprocess_image(image_to_predict_path)
    except Exception as e:
        print(f"Error: {e}")
        return

    # --- Step 4: Make prediction ---
    print("\nMaking a prediction...")
    prediction = model.predict(preprocessed_image)

    # Model outputs probability for 'not_me' (if trained with binary_crossentropy + sigmoid)
    score = prediction[0][0]

    # --- Step 5: Interpret result ---
    print("-" * 30)
    if score >= 0.5:
        predicted_class_index = 1  # 'not_me'
        confidence = score * 100
    else:
        predicted_class_index = 0  # 'me'
        confidence = (1 - score) * 100

    predicted_class = CLASSES[predicted_class_index]
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    main()
