import numpy as np
import os
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau

print("[INFO] Loading prepared FER2013 dataset...")
train_data = np.load("prepared/train_data.npz")
val_data = np.load("prepared/val_data.npz")
test_data = np.load("prepared/test_data.npz")

X_train, y_train = train_data["X"], train_data["y"]
X_val, y_val = val_data["X"], val_data["y"]
X_test, y_test = test_data["X"], test_data["y"]

print(f"[INFO] Train: {X_train.shape}, {y_train.shape}")
print(f"[INFO] Validation: {X_val.shape}, {y_val.shape}")
print(f"[INFO] Test: {X_test.shape}, {y_test.shape}")


def residual_block(x, filters):
    shortcut = x

    # main path
    x = layers.Conv2D(filters, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3,3), padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)

    # project shortcut if filters differ
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1,1), padding='same', activation=None)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def build_fernet_v4(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = layers.MaxPooling2D((2,2))(x)

    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = layers.MaxPooling2D((2,2))(x)

    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name="FERNet_v4")
    return model


model = build_fernet_v4(X_train.shape[1:], y_train.shape[1])
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0003),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

checkpoint = ModelCheckpoint(
    "models/best_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

csv_logger = CSVLogger("logs/training_log.csv", append=True)

print("[INFO] Starting training...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    validation_data=(X_val, y_val),
    epochs=60,
    callbacks=[checkpoint, early_stop, reduce_lr, csv_logger]
)

print("[INFO] Evaluating best model on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f" Test Accuracy: {test_acc * 100:.2f}%")

model.save("models/final_model.h5")
print(" Saved final model to models/final_model.h5")
