import os
from dotenv import load_dotenv
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
from autocorrect import Speller
import google.generativeai as genai

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
chat_model = genai.GenerativeModel('gemini-1.5-flash')

# Class labels (A-Z, del, nothing, space)
class_labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

# Load gesture recognition model
gesture_model = tf.keras.models.load_model('landmark_model.h5')  # Renamed
spell = Speller()
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1)

# Sentence-building variables
sentence = []
current_word = []
last_prediction = None
last_update_time = time.time()
last_space_time = time.time()
DEBOUNCE_TIME = 0.5
COOLDOWN_TIME = 1.0
SPACE_COOLDOWN = 1.0
CONFIDENCE_THRESHOLD = 0.8
SPACE_CLASS = class_labels.index('space')
DEL_CLASS = class_labels.index('del')
NOTHING_CLASS = class_labels.index('nothing')
SENTENCE_TIMEOUT = 2.0

# New variables for better word handling
last_gesture_time = time.time()
is_word_complete = False

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(frame_rgb)

    current_time = time.time()

    if results.multi_hand_landmarks:
        last_gesture_time = current_time

        # Extract landmarks and predict
        landmarks = []
        for lm in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        landmarks = np.array(landmarks).reshape(1, -1)
        prediction = gesture_model.predict(landmarks, verbose=0)[0]  # Updated
        confidence = np.max(prediction)
        predicted_class = np.argmax(prediction) if confidence > CONFIDENCE_THRESHOLD else None

        if predicted_class is not None:
            class_label = class_labels[predicted_class]

            if (current_time - last_update_time) > COOLDOWN_TIME:
                last_update_time = current_time

                if class_label == 'space':
                    if (current_time - last_space_time) > SPACE_COOLDOWN:
                        if current_word:
                            word = ''.join(current_word)
                            sentence.append(word)
                            current_word = []
                            last_space_time = current_time
                            print("Space added")
                        is_word_complete = True

                elif class_label == 'del':
                    if current_word:
                        current_word.pop()
                        print("Letter deleted")
                    elif sentence:
                        sentence.pop()
                        print("Word deleted")

                elif class_label != 'nothing':
                    current_word.append(class_label)
                    print(f"Recognized: {class_label}")
                    is_word_complete = False

    else:
        if (current_time - last_gesture_time) > SENTENCE_TIMEOUT and (current_word or sentence):
            if current_word:
                word = ''.join(current_word)
                sentence.append(word)

            if sentence:
                final_sentence = ' '.join(sentence)
                print("Final Sentence:", final_sentence)

                if final_sentence:
                    try:
                        response = chat_model.generate_content(final_sentence)  # Updated
                        chatbot_response = response.text
                        print("Chatbot Response:", chatbot_response)
                    except Exception as e:
                        print("Chatbot Error:", str(e))

                sentence = []
                current_word = []
                is_word_complete = False
                last_gesture_time = current_time

    # Display
    current_display = ''.join(current_word)
    sentence_display = ' '.join(sentence)
    if current_word:
        sentence_display = sentence_display + (' ' if sentence_display else '') + current_display

    display_text = f"Current: {current_display} | Sentence: {sentence_display}"
    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('ASL Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()