import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

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

# Train the model and store history
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5),
        tf.keras.callbacks.ModelCheckpoint('landmark_model.h5', save_best_only=True)
    ]
)

# 1. Plot epoch vs accuracy graph
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Load the best model for evaluation
best_model = tf.keras.models.load_model('landmark_model.h5')

# Get predictions
y_pred = np.argmax(best_model.predict(X_val), axis=1)

# 2. Create and plot confusion matrix
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(15, 15))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# 3. Analyze per-class performance
classification_rep = classification_report(y_val, y_pred, target_names=class_names, output_dict=True)

# Extract performance metrics per class
classes_precision = {k: v['precision'] for k, v in classification_rep.items() if k in class_names}
classes_recall = {k: v['recall'] for k, v in classification_rep.items() if k in class_names}
classes_f1 = {k: v['f1-score'] for k, v in classification_rep.items() if k in class_names}

# Sort classes by F1 score to identify best and worst performing
sorted_classes = sorted(classes_f1.items(), key=lambda x: x[1])
worst_classes = sorted_classes[:5]  # 5 worst performing
best_classes = sorted_classes[-5:]  # 5 best performing

# Plot the performance of all classes
plt.figure(figsize=(15, 8))
classes = list(classes_f1.keys())
f1_scores = list(classes_f1.values())
precision_scores = list(classes_precision.values())
recall_scores = list(classes_recall.values())

# Sort by F1 score
sorted_indices = np.argsort(f1_scores)
sorted_classes = [classes[i] for i in sorted_indices]
sorted_f1 = [f1_scores[i] for i in sorted_indices]
sorted_precision = [precision_scores[i] for i in sorted_indices]
sorted_recall = [recall_scores[i] for i in sorted_indices]

x = np.arange(len(sorted_classes))
width = 0.25

fig, ax = plt.subplots(figsize=(15, 8))
rects1 = ax.bar(x - width, sorted_precision, width, label='Precision')
rects2 = ax.bar(x, sorted_recall, width, label='Recall')
rects3 = ax.bar(x + width, sorted_f1, width, label='F1-Score')

ax.set_title('Performance Metrics by Class')
ax.set_xlabel('Class')
ax.set_ylabel('Score')
ax.set_xticks(x)
ax.set_xticklabels(sorted_classes, rotation=90)
ax.legend()
ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig('class_performance.png')
plt.show()

# Print best and worst performing classes
print("\nBest performing classes (F1-Score):")
for cls, score in reversed(sorted_classes[-5:]):
    print(f"{cls}: {score:.4f}")

print("\nWorst performing classes (F1-Score):")
for cls, score in sorted_classes[:5]:
    print(f"{cls}: {score:.4f}")

# Additional analysis for misclassifications
# Find the most common misclassifications
misclassifications = {}
for true_label, pred_label in zip(y_val, y_pred):
    if true_label != pred_label:
        true_class = class_names[true_label]
        pred_class = class_names[pred_label]
        key = f"{true_class} → {pred_class}"
        misclassifications[key] = misclassifications.get(key, 0) + 1

# Sort misclassifications by frequency
sorted_misclassifications = sorted(misclassifications.items(), key=lambda x: x[1], reverse=True)

# Print the top misclassifications
print("\nTop misclassifications:")
for pair, count in sorted_misclassifications[:10]:
    print(f"{pair}: {count}")

# Save model performance summary
with open('model_performance_summary.txt', 'w') as f:
    f.write("Sign Language Detection Model Performance Summary\n")
    f.write("=" * 50 + "\n\n")

    f.write("Overall Performance:\n")
    f.write(f"Accuracy: {classification_rep['accuracy']:.4f}\n")
    f.write(f"Macro Avg Precision: {classification_rep['macro avg']['precision']:.4f}\n")
    f.write(f"Macro Avg Recall: {classification_rep['macro avg']['recall']:.4f}\n")
    f.write(f"Macro Avg F1-Score: {classification_rep['macro avg']['f1-score']:.4f}\n\n")

    f.write("Best performing classes (F1-Score):\n")
    for cls, score in reversed(sorted_classes[-5:]):
        f.write(f"{cls}: {score:.4f}\n")

    f.write("\nWorst performing classes (F1-Score):\n")
    for cls, score in sorted_classes[:5]:
        f.write(f"{cls}: {score:.4f}\n")

    f.write("\nTop misclassifications:\n")
    for pair, count in sorted_misclassifications[:10]:
        f.write(f"{pair}: {count}\n")