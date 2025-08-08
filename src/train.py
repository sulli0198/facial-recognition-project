import os
import tensorflow as tf
from tensorflow import keras

# ========================
# CONFIGURATION
# ========================

# Image size (all images will be resized to this)
IMAGE_HEIGHT = 250
IMAGE_WIDTH = 250

# Training parameters
BATCH_SIZE = 9   # Small batch size for small datasets
EPOCHS = 3      # Increase for better results

# Project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'model', 'face_recognition_model.h5')

# Make sure the model directory exists
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)


# ========================
# LOAD DATASET
# ========================
# This will automatically label images based on folder names: 'me' and 'not_me'

print("Loading training dataset...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'train'),
    labels='inferred',         # Infer labels from folder names
    label_mode='binary',       # Binary classification: output will be 0 or 1
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,              # Shuffle for better training
    seed=42
)

print("\nLoading validation dataset...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'validation'),
    labels='inferred',
    label_mode='binary',
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False,             # Validation data should not be shuffled
    seed=42
)

# Optional: confirm classes
print("\nDetected classes:", train_ds.class_names)


# ========================
# BUILD THE MODEL
# ========================
# A simple CNN for binary classification

model = keras.Sequential([
    # Normalize pixel values from [0,255] to [0,1]
    keras.layers.Rescaling(1./255, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),

    # First convolutional block
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(),

    # Second convolutional block
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(),

    # Third convolutional block
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(),

    # Flatten output before dense layers
    keras.layers.Flatten(),

    # Fully connected layer
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),  # Dropout to prevent overfitting

    # Output layer (sigmoid gives probability between 0 and 1)
    keras.layers.Dense(1, activation='sigmoid')
])

# ========================
# COMPILE THE MODEL
# ========================
# Adam optimizer: adaptive learning rate, works well in most cases
# Binary crossentropy: standard loss for binary classification
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Show model summary
model.summary()


# ========================
# TRAIN THE MODEL
# ========================
print("\nStarting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)


# ========================
# SAVE THE MODEL
# ========================
print(f"\nTraining complete. Saving model to {MODEL_SAVE_PATH}...")
model.save(MODEL_SAVE_PATH)
print("Model saved successfully!")
