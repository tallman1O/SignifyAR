from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO
import os
from dotenv import load_dotenv
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
from autocorrect import Speller
import google.generativeai as genai
import threading

load_dotenv()

app = Flask(__name__, static_folder='static')
socketio = SocketIO(app, cors_allowed_origins="*")

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
print("Loading gesture recognition model...")
gesture_model = tf.keras.models.load_model('landmark_model.h5')
spell = Speller()
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1)
print("Model loaded successfully")

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

is_running = True


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


@app.route('/models/<path:path>')
def serve_models(path):
    return send_from_directory('models', path)


@socketio.on('connect')
def handle_connect():
    print("Client connected")
    socketio.emit('connection_status', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")


@socketio.on('test_ar')
def handle_test_ar():
    print("Testing AR display with letter A")
    socketio.emit('update_ar', 'A')


def process_video():
    global sentence, current_word, last_update_time, last_space_time, is_running

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        socketio.emit('error', {'message': 'Failed to open camera'})
        return

    print("Camera opened successfully, starting sign language recognition")

    last_detected_letter = None
    letter_cooldown = 0

    while is_running:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            landmarks = []
            for lm in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks).reshape(1, -1)
            prediction = gesture_model.predict(landmarks, verbose=0)[0]
            confidence = np.max(prediction)
            predicted_class = np.argmax(prediction) if confidence > CONFIDENCE_THRESHOLD else None

            if predicted_class is not None:
                class_label = class_labels[predicted_class]

                # Only process letters A-Z
                if class_label in class_labels[:26]:  # A-Z are the first 26 labels
                    # Debounce to prevent rapid switching
                    if last_detected_letter != class_label and letter_cooldown <= 0:
                        print(f"Detected sign: {class_label}")

                        # Update UI with recognized sign
                        socketio.emit('update_signs', {
                            'current_word': class_label,
                            'sentence': ' '.join(sentence) + ' ' + ''.join(current_word)
                        })

                        # Send letter to AR display
                        socketio.emit('update_ar', class_label)

                        # Update current word
                        current_word.append(class_label)

                        # Set cooldown and last detected letter
                        last_detected_letter = class_label
                        letter_cooldown = 10  # Wait for 10 frames before accepting new letter
                        last_update_time = time.time()

                # Handle special commands
                elif class_label == 'space' and time.time() - last_space_time > SPACE_COOLDOWN:
                    if current_word:
                        word = ''.join(current_word)
                        sentence.append(word)
                        current_word = []
                        socketio.emit('update_signs', {
                            'current_word': '',
                            'sentence': ' '.join(sentence)
                        })
                        last_space_time = time.time()

                elif class_label == 'del':
                    if current_word:
                        current_word.pop()
                        socketio.emit('update_signs', {
                            'current_word': ''.join(current_word),
                            'sentence': ' '.join(sentence)
                        })

            # Decrement cooldown counter
            if letter_cooldown > 0:
                letter_cooldown -= 1

        time.sleep(0.03)

    cap.release()
    print("Video processing stopped")


@socketio.on('stop_processing')
def handle_stop():
    global is_running
    is_running = False
    print("Stopping video processing")


if __name__ == '__main__':
    try:
        print("=" * 60)
        print("Starting Sign Language AR Chatbot")
        print("=" * 60)

        os.makedirs('static', exist_ok=True)
        os.makedirs('models', exist_ok=True)

        video_thread = threading.Thread(target=process_video)
        video_thread.daemon = True
        video_thread.start()

        print("Starting web server at http://0.0.0.0:8000")
        socketio.run(app, host='0.0.0.0', port=8000, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        is_running = False
        print("Shutting down server")