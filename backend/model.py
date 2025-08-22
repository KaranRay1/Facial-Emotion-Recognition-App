# model.py
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import os
import matplotlib.pyplot as plt

# Import constants and data generators from utils.py
from utils import (
    NUM_EMOTION_CLASSES, NUM_GENDER_CLASSES, NUM_AGE_CLASSES,
    IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE,
    load_data_generators, create_combined_generator
)

def build_multi_task_vgg16_model(
    learning_rate=0.001,
    freeze_base=True
):
    """Builds a VGG16-based multi-task learning model."""
    input_tensor = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3), name='input_image')
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

    for layer in base_model.layers:
        layer.trainable = not freeze_base

    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu', name='shared_dense_1')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', name='shared_dense_2')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    emotion_output = Dense(NUM_EMOTION_CLASSES, activation='softmax', name='emotion_output')(x)
    gender_output = Dense(NUM_GENDER_CLASSES, activation='softmax', name='gender_output')(x)
    age_output = Dense(NUM_AGE_CLASSES, activation='softmax', name='age_output')(x)

    model = Model(inputs=input_tensor, outputs=[emotion_output, gender_output, age_output])

    losses = {
        'emotion_output': 'categorical_crossentropy',
        'gender_output': 'categorical_crossentropy',
        'age_output': 'categorical_crossentropy'
    }
    loss_weights = {
        'emotion_output': 1.0,
        'gender_output': 1.0,
        'age_output': 1.0
    }
    metrics = {
        'emotion_output': ['accuracy'],
        'gender_output': ['accuracy'],
        'age_output': ['accuracy']
    }

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)
    return model

def train_and_save_model(
    base_data_dir, # This argument is now mandatory for Colab use
    epochs_phase1=20,
    epochs_phase2=60,
    learning_rate_phase1=0.001,
    learning_rate_phase2=0.00001,
    model_save_dir_colab='/content/drive/MyDrive/Face Emotion Recogization' # UPDATED PATH for model save
):
    """Trains and saves the multi-task VGG16 model."""
    TOTAL_EPOCHS = epochs_phase1 + epochs_phase2
    model_save_path = os.path.join(model_save_dir_colab, 'final_multi_task_vgg16_model.h5')
    os.makedirs(model_save_dir_colab, exist_ok=True) # Ensure save directory exists

    # --- 1. Load Data Generators ---
    (train_emotion_gen, val_emotion_gen, test_emotion_gen,
     train_gender_gen, val_gender_gen, test_gender_gen,
     train_age_gen, val_age_gen, test_age_gen) = load_data_generators(
        base_data_dir, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE
    )

    train_combined_gen = create_combined_generator(train_emotion_gen, train_gender_gen, train_age_gen)
    val_combined_gen = create_combined_generator(val_emotion_gen, val_gender_gen, val_age_gen)
    test_combined_gen = create_combined_generator(test_emotion_gen, test_gender_gen, test_age_gen)

    steps_per_epoch_train = min(len(train_emotion_gen), len(train_gender_gen), len(train_age_gen))
    steps_per_epoch_val = min(len(val_emotion_gen), len(val_gender_gen), len(val_age_gen))
    steps_per_epoch_test = min(len(test_emotion_gen), len(test_gender_gen), len(test_age_gen))

    print(f"\nTraining steps per epoch: {steps_per_epoch_train}")
    print(f"Validation steps per epoch: {steps_per_epoch_val}")

    # --- 2. Build Model (Phase 1) ---
    print("\n--- Building Model for Phase 1 (Frozen VGG16 Base) ---")
    model = build_multi_task_vgg16_model(
        learning_rate=learning_rate_phase1,
        freeze_base=True
    )
    model.summary()

    checkpoint_filepath = os.path.join(model_save_dir_colab, 'multi_task_vgg16_best_model_phase1.h5')
    model_checkpoint_callback_phase1 = ModelCheckpoint(
        filepath=checkpoint_filepath, monitor='val_loss', save_best_only=True,
        save_weights_only=False, mode='min', verbose=1
    )
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=0.0000001, verbose=1)

    # --- 3. Train Phase 1 ---
    print(f"\n--- Starting Training Phase 1: {epochs_phase1} Epochs (Training custom top layers) ---")
    history_phase1 = model.fit(
        train_combined_gen, steps_per_epoch=steps_per_epoch_train, epochs=epochs_phase1,
        validation_data=val_combined_gen, validation_steps=steps_per_epoch_val,
        callbacks=[early_stopping_callback, reduce_lr_callback, model_checkpoint_callback_phase1]
    )

    # --- 4. Fine-tuning Phase 2 ---
    print(f"\n--- Starting Training Phase 2: {epochs_phase2} Epochs (Fine-tuning VGG16 last blocks) ---")
    if os.path.exists(checkpoint_filepath):
        print(f"Loading best model from Phase 1: {checkpoint_filepath}")
        model = load_model(checkpoint_filepath)
    else:
        print("Warning: Best model from Phase 1 not found. Continuing with current model state.")

    model = build_multi_task_vgg16_model(
        learning_rate=learning_rate_phase2,
        freeze_base=False
    )

    vgg16_layer = None
    for layer in model.layers:
        if layer.name == 'vgg16':
            vgg16_layer = layer
            break
    if vgg16_layer:
        for sub_layer in vgg16_layer.layers:
            sub_layer.trainable = False
        for sub_layer in vgg16_layer.layers:
            if sub_layer.name.startswith('block4') or sub_layer.name.startswith('block5'):
                sub_layer.trainable = True
        print("\n--- Model Summary for Phase 2 (Fine-tuning) ---")
        model.summary()
    else:
        print("Error: VGG16 base layer not found in the model for fine-tuning. Exiting.")
        return

    early_stopping_callback_phase2 = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    reduce_lr_callback_phase2 = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=0.0000001, verbose=1)
    model_checkpoint_callback_phase2 = ModelCheckpoint(
        filepath=model_save_path, monitor='val_loss', save_best_only=True,
        save_weights_only=False, mode='min', verbose=1
    )

    history_phase2 = model.fit(
        train_combined_gen, steps_per_epoch=steps_per_epoch_train, epochs=epochs_phase2,
        validation_data=val_combined_gen, validation_steps=steps_per_epoch_val,
        callbacks=[early_stopping_callback_phase2, reduce_lr_callback_phase2, model_checkpoint_callback_phase2]
    )

    # --- 5. Save the final best model ---
    if os.path.exists(model_save_path):
        print("\n--- Training Complete. Final best model saved to {} ---".format(model_save_path))
    else:
        print("\n--- Training Complete. Best model was not saved by checkpoint. Saving current state. ---")
        model.save(model_save_path)

    # --- 6. Plot Training History ---
    def plot_history(history1, history2):
        history = {}
        for key in history1.history.keys():
            history[key] = history1.history[key] + history2.history[key]
        epochs_range = range(TOTAL_EPOCHS)
        plt.figure(figsize=(18, 12))

        plt.subplot(2, 2, 1)
        plt.plot(epochs_range, history['loss'], label='Training Total Loss')
        plt.plot(epochs_range, history['val_loss'], label='Validation Total Loss')
        plt.title('Total Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(epochs_range, history['emotion_output_accuracy'], label='Training Emotion Accuracy')
        plt.plot(epochs_range, history['val_emotion_output_accuracy'], label='Validation Emotion Accuracy')
        plt.title('Emotion Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot(epochs_range, history['gender_output_accuracy'], label='Training Gender Accuracy')
        plt.plot(epochs_range, history['val_gender_output_accuracy'], label='Validation Gender Accuracy')
        plt.title('Gender Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(epochs_range, history['age_output_accuracy'], label='Training Age Accuracy')
        plt.plot(epochs_range, history['val_age_output_accuracy'], label='Validation Age Accuracy')
        plt.title('Age Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    plot_history(history_phase1, history_phase2)

    # --- 7. Evaluate on Test Data ---
    final_model = load_model(model_save_path)
    print("\n--- Evaluating the final model on test data ---")
    test_results = final_model.evaluate(test_combined_gen, steps=steps_per_epoch_test)
    results_dict = dict(zip(final_model.metrics_names, test_results))
    print("\nFinal Test Results:")
    for metric_name, value in results_dict.items():
        print("  {}: {:.4f}".format(metric_name, value))

    print("\nOverall Test Accuracy (Emotion): {:.4f}".format(results_dict.get('emotion_output_accuracy', 'N/A')))
    print("Overall Test Accuracy (Gender): {:.4f}".format(results_dict.get('gender_output_accuracy', 'N/A')))
    print("Overall Test Accuracy (Age): {:.4f}".format(results_dict.get('age_output_accuracy', 'N/A')))
    print("Overall Test Loss: {:.4f}".format(results_dict.get('loss', 'N/A')))

    return final_model

if __name__ == '__main__':
    print("Running model.py directly to train and save the model.")
    # UPDATED PATH for dataset and model save location
    ACTUAL_DATASET_PATH = '/content/drive/MyDrive/Face Emotion Recogization/data' # Unzipped 'data' folder
    MAIN_PROJECT_SAVE_PATH = '/content/drive/MyDrive/Face Emotion Recogization' # Project root for model save

    trained_model = train_and_save_model(
        base_data_dir=ACTUAL_DATASET_PATH,
        model_save_dir_colab=MAIN_PROJECT_SAVE_PATH
    )
    print("Model training complete and saved.")