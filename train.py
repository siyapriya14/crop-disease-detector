import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Small Dataset / Testing Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 2  # Kam images ke liye batch size small rakha hai
EPOCHS = 5
DATASET_PATH = 'dataset'
MODEL_SAVE_PATH = 'models/crop_model.tflite'

def build_and_train():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset directory '{DATASET_PATH}' not found. Add class folders with images.")

    # Data Generator (Simple Rescaling without split for tiny dataset testing)
    datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    num_classes = train_gen.num_classes
    if num_classes == 0:
        raise ValueError("No classes found in dataset directory. Ensure folders like Healthy, Early_Blight exist inside 'dataset'.")

    # MobileNetV2 Transfer Learning Setup
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("--- Starting Training ---")
    model.fit(train_gen, epochs=EPOCHS)

    # Convert Trained Model to Optimized TFLite
    os.makedirs('models', exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(MODEL_SAVE_PATH, 'wb') as f:
        f.write(tflite_model)
    
    print(f"--- TFLite Model Saved Successfully at {MODEL_SAVE_PATH} ---")

if __name__ == '__main__':
    build_and_train()