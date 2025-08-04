import tensorflow as tf
from tensorflow import keras
import os


layers = tf.keras.layers

# --- Configuration ---
IMAGE_HEIGHT = 250
IMAGE_WIDTH = 250
BATCH_SIZE = 50
EPOCHS = 2 # You can increase this if needed, but start with a small number
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'model', 'face_recognition_model.h5')

# Ensure the model save directory exists
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# --- Step 1: Load the Dataset ---
# Keras's image_dataset_from_directory is the perfect tool for our folder structure.
# It automatically infers labels from the subdirectory names ('me', 'not_me').
print("Loading training data...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'train'),
    labels='inferred',
    label_mode='binary', # Use 'binary' for 2 classes
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True, # Shuffle the training data
    seed=123 # Use a seed for reproducible results
)

print("\nLoading validation data...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'validation'),
    labels='inferred',
    label_mode='binary',
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False, # Don't need to shuffle validation data
    seed=123
)

# Optional: Print the class names to confirm labels
class_names = train_ds.class_names
print(f"\nFound classes: {class_names}")
print(f"  - 'me' will likely be label 0")
print(f"  - 'not_me' will likely be label 1")

# --- Step 2: Build the Model ---
# This is a simple Convolutional Neural Network (CNN) architecture.
model = keras.Sequential([
    # Rescaling layer to normalize pixel values from [0, 255] to [0, 1]
    layers.Rescaling(1./255, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),

    # Convolutional Block 1
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    # Convolutional Block 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    # Convolutional Block 3
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    # Flatten the 3D output to 1D for the Dense layers
    layers.Flatten(),

    # Dense layers for classification
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Dropout to prevent overfitting

    # Final output layer with a sigmoid activation for binary classification
    # It outputs a single value between 0 and 1, representing the probability.
    layers.Dense(1, activation='sigmoid')
])

# --- Step 3: Compile the Model ---
# Adam is a good all-purpose optimizer.
# binary_crossentropy is the standard loss function for binary classification.
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Print a summary of the model's layers
model.summary()

# --- Step 4: Train the Model ---
print("\nStarting model training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# --- Step 5: Save the Trained Model ---
print(f"\nTraining finished. Saving model to: {MODEL_SAVE_PATH}")
model.save(MODEL_SAVE_PATH)

print("Model saved successfully!")