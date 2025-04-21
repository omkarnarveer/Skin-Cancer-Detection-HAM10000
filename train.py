import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Constants
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 7
LEARNING_RATE = 1e-4

BASE_DIR = 'dataset'
RAW_DATA_DIR = os.path.join(BASE_DIR, 'raw')
MODEL_DIR = 'static/models'
HISTORY_PATH = os.path.join(MODEL_DIR, 'training_history.json')
MODEL_PATH = os.path.join(MODEL_DIR, 'skin_cancer_model.h5')
os.makedirs(MODEL_DIR, exist_ok=True)

def prepare_data():
    metadata_path = os.path.join(RAW_DATA_DIR, 'HAM10000_metadata.csv')
    df = pd.read_csv(metadata_path)
    df['path'] = df['image_id'].apply(lambda x: os.path.join(RAW_DATA_DIR, 'images', f'{x}.jpg'))
    df = df[df['path'].apply(os.path.exists)].reset_index(drop=True)

    train_df, test_df = train_test_split(df, test_size=0.3, stratify=df['dx'], random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=0.5, stratify=test_df['dx'], random_state=42)
    return train_df, val_df, test_df

def create_generators(train_df, val_df, test_df):
    datagen = ImageDataGenerator(rescale=1./255)

    train_gen = datagen.flow_from_dataframe(
        train_df, x_col='path', y_col='dx',
        target_size=IMAGE_SIZE, class_mode='categorical',
        batch_size=BATCH_SIZE, shuffle=True
    )

    val_gen = datagen.flow_from_dataframe(
        val_df, x_col='path', y_col='dx',
        target_size=IMAGE_SIZE, class_mode='categorical',
        batch_size=BATCH_SIZE, shuffle=False
    )

    test_gen = datagen.flow_from_dataframe(
        test_df, x_col='path', y_col='dx',
        target_size=IMAGE_SIZE, class_mode='categorical',
        batch_size=BATCH_SIZE, shuffle=False
    )

    return train_gen, val_gen, test_gen

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(*IMAGE_SIZE, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train():
    train_df, val_df, test_df = prepare_data()
    train_gen, val_gen, test_gen = create_generators(train_df, val_df, test_df)

    model = build_model()
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(filepath=MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max')
    ]

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    save_model(model, MODEL_PATH)
    with open(HISTORY_PATH, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)

if __name__ == "__main__":
    tf.keras.utils.set_random_seed(42)
    train()