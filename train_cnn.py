# train_cnn.py
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

if __name__ == "__main__":

    # -------------------------
    # 1️⃣ Set Dataset Paths
    # -------------------------
    train_dir = 'data/train'
    val_dir   = 'data/val'
    test_dir  = 'data/test'

    # -------------------------
    # 2️⃣ Hyperparameters
    # -------------------------
    img_size   = (224, 224)
    batch_size = 32
    epochs     = 5  # Change to 10 for better accuracy

    # -------------------------
    # 3️⃣ Data Preprocessing
    # -------------------------
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    val_data = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    test_data = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    # -------------------------
    # 4️⃣ Build CNN Model
    # -------------------------
    model = Sequential()
    # Convolution + Pooling layers
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Flatten
    model.add(Flatten())

    # Fully Connected Layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Output Layer (Binary Classification)
    model.add(Dense(1, activation='sigmoid'))

    # Model Summary
    model.summary()

    # -------------------------
    # 5️⃣ Compile Model
    # -------------------------
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # -------------------------
    # 6️⃣ Train Model
    # -------------------------
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs
    )

    # -------------------------
    # 7️⃣ Save Model
    # -------------------------
    os.makedirs('models', exist_ok=True)
    model.save('models/cnn_model.h5')
    print("✅ Model saved successfully!")

    # -------------------------
    # 8️⃣ Plot Accuracy & Loss
    # -------------------------
    os.makedirs('outputs', exist_ok=True)

    # Accuracy Plot
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.savefig('outputs/accuracy_plot.png')
    plt.close()

    # Loss Plot
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    plt.savefig('outputs/loss_plot.png')
    plt.close()

    # -------------------------
    # 9️⃣ Evaluate on Test Data
    # -------------------------
    loss, accuracy = model.evaluate(test_data)
    print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")