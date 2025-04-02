import os
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)


def extract_landmarks(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(img_rgb)
    if results.multi_hand_landmarks:
        landmarks = []
        for lm in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])  # 63 values (21 landmarks * 3 coordinates)
        return np.array(landmarks)
    else:
        return None


# Create directories for saving landmarks
dataset_dir = "data/train"  # Original ASL dataset (A-Z folders)
output_dir = "landmark_data"
os.makedirs(output_dir, exist_ok=True)

# Process all images
for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)

    # Skip if it's not a directory (e.g., .DS_Store)
    if not os.path.isdir(class_dir):
        continue

    output_class_dir = os.path.join(output_dir, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)

        # Skip non-image files (e.g., .DS_Store)
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        landmarks = extract_landmarks(img_path)
        if landmarks is not None:
            np.save(os.path.join(output_class_dir, img_file[:-4] + ".npy"), landmarks)