# utils.py
%%writefile utils.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import cv2
import time

# Define image dimensions and number of classes for each task
IMG_WIDTH, IMG_HEIGHT = 48, 48
NUM_EMOTION_CLASSES = 7
NUM_GENDER_CLASSES = 2
NUM_AGE_CLASSES = 8

# Define batch size
BATCH_SIZE = 64

# Mapping for emotion, gender, and age labels
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
GENDER_LABELS = ['male', 'female']
AGE_LABELS = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']

def preprocess_image_for_vgg16(image_array):
    """
    Preprocesses a single image array for VGG16 input.
    - Ensures 3 channels (duplicates grayscale if needed).
    - Rescales pixel values to [0, 1].
    - Resizes to VGG16 expected input size.
    """
    resized_image = cv2.resize(image_array, (IMG_WIDTH, IMG_HEIGHT))
    if len(resized_image.shape) == 2:
        processed_image = np.stack([resized_image, resized_image, resized_image], axis=-1)
    elif resized_image.shape[-1] == 1:
        processed_image = np.concatenate([resized_image, resized_image, resized_image], axis=-1)
    else:
        processed_image = resized_image
    processed_image = processed_image.astype('float32') / 255.0
    return processed_image

def load_data_generators(base_data_dir, img_width, img_height, batch_size):
    """
    Loads data using ImageDataGenerator for emotion, gender, and age.
    Assumes your dataset is already structured correctly at base_data_dir.
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    print(f"Loading data from: {base_data_dir}")

    # --- Emotion Data Generator ---
    emotion_train_dir = os.path.join(base_data_dir, 'emotion_data', 'train')
    emotion_val_dir = os.path.join(base_data_dir, 'emotion_data', 'validation')
    emotion_test_dir = os.path.join(base_data_dir, 'emotion_data', 'test')

    train_emotion_generator = train_datagen.flow_from_directory(
        emotion_train_dir, target_size=(img_width, img_height), color_mode='rgb',
        batch_size=batch_size, class_mode='categorical', shuffle=True, classes=EMOTION_LABELS
    )
    val_emotion_generator = val_test_datagen.flow_from_directory(
        emotion_val_dir, target_size=(img_width, img_height), color_mode='rgb',
        batch_size=batch_size, class_mode='categorical', shuffle=False, classes=EMOTION_LABELS
    )
    test_emotion_generator = val_test_datagen.flow_from_directory(
        emotion_test_dir, target_size=(img_width, img_height), color_mode='rgb',
        batch_size=batch_size, class_mode='categorical', shuffle=False, classes=EMOTION_LABELS
    )

    # --- Gender Data Generator ---
    gender_train_dir = os.path.join(base_data_dir, 'gender_data', 'train')
    gender_val_dir = os.path.join(base_data_dir, 'gender_data', 'validation')
    gender_test_dir = os.path.join(base_data_dir, 'gender_data', 'test')

    train_gender_generator = train_datagen.flow_from_directory(
        gender_train_dir, target_size=(img_width, img_height), color_mode='rgb',
        batch_size=batch_size, class_mode='categorical', shuffle=True, classes=GENDER_LABELS
    )
    val_gender_generator = val_test_datagen.flow_from_directory(
        gender_val_dir, target_size=(img_width, img_height), color_mode='rgb',
        batch_size=batch_size, class_mode='categorical', shuffle=False, classes=GENDER_LABELS
    )
    test_gender_generator = val_test_datagen.flow_from_directory(
        gender_test_dir, target_size=(img_width, img_height), color_mode='rgb',
        batch_size=batch_size, class_mode='categorical', shuffle=False, classes=GENDER_LABELS
    )

    # --- Age Data Generator ---
    age_train_dir = os.path.join(base_data_dir, 'age_data', 'train')
    age_val_dir = os.path.join(base_data_dir, 'age_data', 'validation')
    age_test_dir = os.path.join(base_data_dir, 'age_data', 'test')

    train_age_generator = train_datagen.flow_from_directory(
        age_train_dir, target_size=(img_width, img_height), color_mode='rgb',
        batch_size=batch_size, class_mode='categorical', shuffle=True, classes=AGE_LABELS
    )
    val_age_generator = val_test_datagen.flow_from_directory(
        age_val_dir, target_size=(img_width, img_height), color_mode='rgb',
        batch_size=batch_size, class_mode='categorical', shuffle=False, classes=AGE_LABELS
    )
    test_age_generator = val_test_datagen.flow_from_directory(
        age_test_dir, target_size=(img_width, img_height), color_mode='rgb',
        batch_size=batch_size, class_mode='categorical', shuffle=False, classes=AGE_LABELS
    )

    return (train_emotion_generator, val_emotion_generator, test_emotion_generator,
            train_gender_generator, val_gender_generator, test_gender_generator,
            train_age_generator, val_age_generator, test_age_generator)

def create_combined_generator(emotion_gen, gender_gen, age_gen):
    """
    Combines outputs from individual generators into a format suitable for multi-output model training.
    """
    while True:
        x_e, y_e = next(emotion_gen)
        _, y_g = next(gender_gen)
        _, y_a = next(age_gen)

        min_batch_size = min(len(x_e), len(y_g), len(y_a))

        yield x_e[:min_batch_size], {
            'emotion_output': y_e[:min_batch_size],
            'gender_output': y_g[:min_batch_size],
            'age_output': y_a[:min_batch_size]
        }

if __name__ == '__main__':
    print("Running utils.py directly for demonstration of data loading.")
    # UPDATED PATH for dataset
    TEST_DATA_PATH = '/content/drive/MyDrive/Project/Face Emotion Recogization/face/data'

    (train_emotion_gen, val_emotion_gen, test_emotion_gen,
     train_gender_gen, val_gender_gen, test_gender_gen,
     train_age_gen, val_age_gen, test_age_gen) = load_data_generators(
        TEST_DATA_PATH, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE
    )

    print("\nIndividual generators loaded.")
    print("Emotion Train Samples:", train_emotion_gen.samples)
    print("Gender Train Samples:", train_gender_gen.samples)
    print("Age Train Samples:", train_age_gen.samples)

    print("\nTesting combined generator (should yield actual data):")
    combined_gen = create_combined_generator(train_emotion_gen, train_gender_gen, train_age_gen)
    try:
        sample_x, sample_y = next(combined_gen)
        print("Sample combined batch input shape:", sample_x.shape)
        print("Sample combined batch emotion output shape:", sample_y['emotion_output'].shape)
        print("Sample combined batch gender output shape:", sample_y['gender_output'].shape)
        print("Sample combined batch age output shape:", sample_y['age_output'].shape)
    except Exception as e:
        print("Could not get sample from combined generator. Ensure data directories have images. Error:", e)