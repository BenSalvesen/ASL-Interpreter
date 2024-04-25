# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Set the size of the images that our model will expect. This must match the size of the images used during training.
image_size = (140, 140)
batch_size = 10

train_dir = 'images'

# Set the seed for reproducibility
seed = 123

# Create a dataset from image files in a directory for training and validation
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels = 'inferred',
    label_mode = 'categorical',
    class_names = ['a','b','c','d','e'],
    color_mode = 'grayscale',
    batch_size = batch_size,
    image_size = image_size,
    shuffle = True,
    seed = seed,
    validation_split = 0.2,
    subset = "training",
    interpolation = 'bilinear',
    follow_links = False,
    crop_to_aspect_ratio = False
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels = 'inferred',
    label_mode = 'categorical',
    class_names = ['a','b','c','d','e'],
    color_mode = 'grayscale',
    batch_size = batch_size,
    image_size = image_size,
    shuffle = True,
    seed = seed,
    validation_split = 0.2,
    subset = "validation",
    interpolation = 'bilinear',
    follow_links = False,
    crop_to_aspect_ratio = False
)


# Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size = AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size = AUTOTUNE)

# Define the CNN model architecture
model = Sequential([
    Input(shape=(image_size[0], image_size[1], 1)),  # Only one channel for grayscale
    Conv2D(32, (3, 3), activation = 'relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation = 'relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation = 'relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation = 'relu'),
    Dropout(0.5),
    Dense(5, activation = 'softmax')
])

# Compile the model
model.compile(
    optimizer = Adam(),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

# Early stopping callback
early_stopping = EarlyStopping(
    monitor = 'val_loss',  # Monitors the validation loss
    patience = 5,          # Number of epochs with no improvement after which training will be stopped
    restore_best_weights = True  # Restores model weights from the epoch with the best value of the monitored quantity
)

# Add ModelCheckpoint callback
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.keras',
    save_best_only = True,
    monitor = 'val_accuracy',
    mode = 'max'
)

# Train the model with early stopping and model checkpoint
history = model.fit(
    train_dataset,
    validation_data = validation_dataset,
    epochs = 20,
    callbacks = [early_stopping, model_checkpoint]  # Include both callbacks
)

# Save the trained model
model.save('hand_gesture_model.keras')