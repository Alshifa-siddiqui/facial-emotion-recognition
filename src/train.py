import os, numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout, BatchNormalization)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# paths
train_dir = os.path.join("..","data","train")
val_dir   = os.path.join("..","data","val")
save_keras = os.path.join("..","models","emotion_model.keras")
save_h5    = os.path.join("..","models","emotion_model.h5")  # optional legacy

# data
train_gen = ImageDataGenerator(rescale=1./255,
                               rotation_range=10,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               zoom_range=0.1,
                               horizontal_flip=True)
val_gen   = ImageDataGenerator(rescale=1./255)

train = train_gen.flow_from_directory(train_dir, target_size=(48,48),
                                      color_mode="grayscale",
                                      batch_size=64, class_mode="categorical")
val   = val_gen.flow_from_directory(val_dir, target_size=(48,48),
                                    color_mode="grayscale",
                                    batch_size=64, class_mode="categorical")

# model
model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(48,48,1)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64,(3,3),activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128,(3,3),activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

cbs = [
    EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
    ModelCheckpoint(save_keras, monitor="val_accuracy", save_best_only=True, verbose=1)
]

history = model.fit(train, validation_data=val, epochs=40, callbacks=cbs)

# save both (new + legacy)
model.save(save_keras)
model.save(save_h5)
print("Saved:", save_keras, "and", save_h5)
