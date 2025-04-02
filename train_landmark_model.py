import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

# Load preprocessed landmarks and labels
X = []
y = []
output_dir = "landmark_data"

# Get sorted list of class folders (A, B, ..., Z, del, nothing, space)
class_names = sorted([d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))])

# Ensure there are exactly 29 classes
assert len(class_names) == 29, f"Expected 29 classes, found {len(class_names)}"

for class_idx, class_name in enumerate(class_names):  # class_idx ranges 0–28
    class_dir = os.path.join(output_dir, class_name)
    for landmark_file in os.listdir(class_dir):
        landmark_path = os.path.join(class_dir, landmark_file)
        landmarks = np.load(landmark_path)
        X.append(landmarks)
        y.append(class_idx)  # Assign correct label (0–28)

X = np.array(X)
y = np.array(y)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Simple dense model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(63,)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(29, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5),
        tf.keras.callbacks.ModelCheckpoint('landmark_model.h5', save_best_only=True)
    ]
)