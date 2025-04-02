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

# New variables for better word handling
last_gesture_time = time.time()
is_word_complete = False
is_running = True

# Debug logging settings
DEBUG_LEVEL = 1  # 0: minimal, 1: normal, 2: verbose


def log(message, level=1):
    """Print log message if its level is less than or equal to DEBUG_LEVEL"""
    if level <= DEBUG_LEVEL:
        print(f"[{time.strftime('%H:%M:%S')}] {message}")


@app.route('/')
def index():
    return render_template('index.html')


# Serve static files
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


@app.route('/models/<path:path>')
def serve_models(path):
    return send_from_directory('models', path)


@socketio.on('connect')
def handle_connect():
    log("Client connected", 0)
    # Send a test message to verify connection
    socketio.emit('connection_status', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    log("Client disconnected", 0)


@socketio.on('check_connection')
def handle_check():
    log("Connection check received", 0)
    # Echo back to confirm server is responding
    socketio.emit('connection_status', {'status': 'active'})


def process_video():
    global sentence, current_word, last_update_time, last_space_time, last_gesture_time, is_word_complete, is_running

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log("ERROR: Could not open camera", 0)
        socketio.emit('error', {'message': 'Failed to open camera'})
        return

    log("Camera opened successfully, starting sign language recognition", 0)
    log(f"Configuration: Confidence threshold: {CONFIDENCE_THRESHOLD}, Cooldown time: {COOLDOWN_TIME}s", 1)

    # Send a test AR response to verify functionality on startup
    socketio.emit('update_ar', "AR System Ready! Make hand gestures to communicate.")

    while is_running:
        success, frame = cap.read()
        if not success:
            log("Failed to grab frame", 0)
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
            prediction = gesture_model.predict(landmarks, verbose=0)[0]
            confidence = np.max(prediction)
            predicted_class = np.argmax(prediction) if confidence > CONFIDENCE_THRESHOLD else None

            if predicted_class is not None:
                class_label = class_labels[predicted_class]
                log(f"Detected gesture: {class_label} (confidence: {confidence:.2f})", 2)

                if (current_time - last_update_time) > COOLDOWN_TIME:
                    last_update_time = current_time

                    if class_label == 'space':
                        if (current_time - last_space_time) > SPACE_COOLDOWN:
                            if current_word:
                                word = ''.join(current_word)
                                sentence.append(word)
                                current_word = []
                                last_space_time = current_time
                                log(f"SPACE detected - Added word '{word}' to sentence", 0)
                            is_word_complete = True

                    elif class_label == 'del':
                        if current_word:
                            deleted = current_word.pop()
                            log(f"DEL detected - Deleted letter '{deleted}' from current word", 0)
                        elif sentence:
                            deleted = sentence.pop()
                            log(f"DEL detected - Deleted word '{deleted}' from sentence", 0)

                    elif class_label != 'nothing':
                        current_word.append(class_label)
                        log(f"LETTER detected - Added '{class_label}' to current word", 0)
                        log(f"Current word: {''.join(current_word)}", 1)
                        is_word_complete = False

                    # Update UI with current status
                    current_display = ''.join(current_word)
                    sentence_display = ' '.join(sentence)
                    full_display = sentence_display + (
                        ' ' if sentence_display and current_display else '') + current_display

                    log(f"Current input state: {full_display}", 1)

                    socketio.emit('update_signs', {
                        'current_word': current_display,
                        'sentence': sentence_display
                    })

        else:
            if (current_time - last_gesture_time) > SENTENCE_TIMEOUT and (current_word or sentence):
                if current_word:
                    word = ''.join(current_word)
                    sentence.append(word)
                    log(f"Timeout - Added word '{word}' to sentence", 0)

                if sentence:
                    final_sentence = ' '.join(sentence)
                    log("=" * 50, 0)
                    log(f"FINAL SENTENCE: \"{final_sentence}\"", 0)
                    log("=" * 50, 0)

                    if final_sentence:
                        try:
                            log(f"Sending to Gemini: \"{final_sentence}\"", 0)
                            response = chat_model.generate_content(final_sentence)
                            chatbot_response = response.text
                            log(f"CHATBOT RESPONSE: \"{chatbot_response}\"", 0)
                            log("=" * 50, 0)

                            # Send the response to the AR interface
                            socketio.emit('update_ar', chatbot_response)
                            # Send a debug message as well to verify data flow
                            socketio.emit('debug', {
                                'message': 'Sent AR response',
                                'response': chatbot_response[:30] + ('...' if len(chatbot_response) > 30 else '')
                            })
                        except Exception as e:
                            error_msg = f"ERROR with Gemini API: {str(e)}"
                            log(error_msg, 0)
                            socketio.emit('update_ar', f"Sorry, I couldn't process that. Error: {str(e)}")
                            socketio.emit('error', {'message': error_msg})

                    sentence = []
                    current_word = []
                    is_word_complete = False
                    last_gesture_time = current_time

                    # Update UI
                    socketio.emit('update_signs', {
                        'current_word': '',
                        'sentence': ''
                    })

        # Sleep a bit to reduce CPU usage
        time.sleep(0.03)

    cap.release()
    log("Video processing stopped", 0)


@socketio.on('stop_processing')
def handle_stop():
    global is_running
    is_running = False
    log("Stopping video processing", 0)


@socketio.on('test_ar')
def handle_test_ar():
    """Handle test request from client to verify AR display works"""
    log("Received test AR request", 0)
    socketio.emit('update_ar', "This is a test AR message. If you can see this, AR is working!")


if __name__ == '__main__':
    try:
        log("=" * 60, 0)
        log("Starting Sign Language AR Chatbot", 0)
        log("=" * 60, 0)

        # Create directories if they don't exist
        os.makedirs('static', exist_ok=True)
        os.makedirs('models', exist_ok=True)

        # Start video processing in a separate thread
        video_thread = threading.Thread(target=process_video)
        video_thread.daemon = True
        video_thread.start()

        # Start the web server
        log("Starting web server at http://0.0.0.0:8000", 0)
        socketio.run(app, host='0.0.0.0', port=8000, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        is_running = False
        log("Shutting down server", 0)