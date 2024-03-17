import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import matplotlib.pyplot as plt 


def face_detection_model(path):
    # Define paths to your data
    base_dir = path
    train_dir = os.path.join(base_dir, "train_face_model")
    validation_dir = os.path.join(base_dir, "validation_face_model")

    # Data augmentation and generator setup
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(150, 150), batch_size=32, class_mode="binary"
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir, target_size=(150, 150), batch_size=32, class_mode="binary"
    )

    # Model architecture
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(512, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=100,
        epochs=15,  
        validation_data=validation_generator,
        validation_steps=50,
    )
    
     # Plotting Training and Validation Accuracy
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Face Detection Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plotting Training and Validation Loss
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Face Detection Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()  # Making room for the legends
    plt.show()

    # Save the model
    model_dir = os.path.join(base_dir, "pre-trained-models/face_model.h5")
    model.save(model_dir)

base_dir = Path(__file__).resolve().parent.parent
face_detection_model(base_dir)