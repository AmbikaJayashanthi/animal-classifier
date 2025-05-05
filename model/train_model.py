# train_model.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
import json

# Paths
dataset_path = "dataset"

# Data preparation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Save class indices
with open("model/class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f)

# Load base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False

# Custom classification model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Checkpoint
checkpoint = ModelCheckpoint(
    'model/best_model_transfer_learning.keras', monitor='val_accuracy', save_best_only=True
)

# Train
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=7,
    callbacks=[checkpoint]
)

# Save final accuracy/loss
final_val_acc = history.history['val_accuracy'][-1]
final_val_loss = history.history['val_loss'][-1]
with open("model/metrics.txt", "w") as f:
    f.write(f"val_accuracy={final_val_acc}\n")
    f.write(f"val_loss={final_val_loss}\n")
