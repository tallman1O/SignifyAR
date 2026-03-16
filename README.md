# SignifyAR

**SignifyAR** is a real-time American Sign Language (ASL) recognition system with augmented reality display and AI chatbot integration. It uses your webcam to detect hand gestures, recognizes ASL letters (A–Z) using a trained machine learning model, and responds intelligently through a Google Gemini-powered chatbot — all rendered in a 3D augmented reality interface in the browser.

---

## Features

- 🤟 **Real-time ASL Recognition** — Detects and classifies 29 ASL gestures (A–Z, space, delete, nothing) from a live webcam feed
- 🖐️ **Hand Landmark Tracking** — Uses [MediaPipe](https://mediapipe.dev/) to extract 21 3D hand landmarks per frame
- 🧠 **Deep Learning Classification** — TensorFlow/Keras dense neural network trained on hand landmark data
- 🌐 **3D AR Display** — Three.js renders recognized letters as 3D animated text overlaid on the webcam feed
- 🤖 **AI Chatbot** — Sends formed sentences to Google Gemini (`gemini-1.5-flash`) and displays AI-generated responses
- ⚡ **Real-time Communication** — Flask-SocketIO provides bidirectional real-time updates between the backend and browser
- ✏️ **Autocorrect Support** — Corrects spelling of detected words using the `autocorrect` library

---

## Architecture & Workflow

```
User Performs ASL Gesture
         │
         ▼
Webcam Captures Video (OpenCV)
         │
         ▼
Hand Landmark Extraction (MediaPipe)
         │
         ▼
Gesture Classification (TensorFlow Model)
         │
         ▼
Flask-SocketIO Backend (server.py)
         │
    ┌────┴────┐
    ▼         ▼
3D AR Display   Google Gemini AI
(Three.js)      Chatbot Response
    │               │
    └────────┬───────┘
             ▼
      Browser UI (HTML/JS)
```

---

## Tech Stack

| Layer        | Technology                                    |
|--------------|-----------------------------------------------|
| Backend      | Python, Flask, Flask-SocketIO                 |
| ML / Vision  | TensorFlow / Keras, MediaPipe                 |
| AI           | Google Gemini API (`gemini-1.5-flash`)        |
| Frontend     | HTML, CSS, JavaScript, Three.js, Socket.IO    |
| NLP          | `autocorrect` (Speller)                       |

---

## Project Structure

```
SignifyAR/
├── server.py                  # Flask server + real-time video processing + gesture recognition
├── preprocess_dataset.py      # Extracts MediaPipe hand landmarks from ASL image dataset
├── train_landmark_model.py    # Trains the TensorFlow gesture classification model
├── landmark_model.h5          # Trained gesture recognition model (generated after training)
├── templates/
│   └── index.html             # Web UI template
├── static/
│   ├── app.js                 # Frontend JavaScript (Three.js AR, Socket.IO)
│   └── style.css              # Styles
├── models/
│   └── avatar.glb             # 3D avatar model for AR display
├── temp/
│   └── real_time_inference.py # Standalone real-time inference script (no web server)
├── flowchart.mmd              # Mermaid architecture flowchart
├── confusion_matrix.png       # Model evaluation: confusion matrix (generated after training)
├── training_history.png       # Model evaluation: accuracy/loss curves (generated after training)
├── .env                       # Environment variables (GEMINI_API_KEY)
└── .gitignore
```

---

## Prerequisites

- Python 3.8+
- A webcam
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)
- An ASL image dataset (e.g., [ASL Alphabet dataset on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)) for training

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/tallman1O/SignifyAR.git
cd SignifyAR
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv myenv
source myenv/bin/activate       # On Windows (cmd): myenv\Scripts\activate.bat

pip install flask flask-socketio opencv-python mediapipe tensorflow \
            autocorrect google-generativeai python-dotenv
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_google_gemini_api_key_here
```

---

## Dataset Preparation & Model Training

> Skip this section if you already have a `landmark_model.h5` file.

### Step 1 — Prepare the ASL dataset

Download an ASL image dataset and place it in the following directory structure:

```
data/
└── train/
    ├── A/
    │   ├── image1.jpg
    │   └── ...
    ├── B/
    │   └── ...
    ├── ...
    ├── del/
    ├── nothing/
    └── space/
```

### Step 2 — Extract hand landmarks

Run the preprocessing script to extract MediaPipe hand landmarks from all images. This saves `.npy` files to a `landmark_data/` directory:

```bash
python preprocess_dataset.py
```

### Step 3 — Train the gesture recognition model

Train the TensorFlow model on the extracted landmarks:

```bash
python train_landmark_model.py
```

This will:
- Train a dense neural network for 50 epochs (with early stopping)
- Save the best model to `landmark_model.h5`
- Generate `training_history.png` (accuracy/loss curves)
- Generate `confusion_matrix.png` (per-class performance)

---

## Running the Application

Start the Flask-SocketIO web server:

```bash
python server.py
```

Then open your browser and navigate to:

```
http://localhost:8000
```

The server will:
1. Load the trained `landmark_model.h5`
2. Open your webcam for real-time hand tracking
3. Emit recognized signs and AR updates via WebSocket

---

## Using the Web Interface

| Control              | Description                                              |
|----------------------|----------------------------------------------------------|
| **Current Word**     | Displays letters being signed in real time               |
| **Current Sentence** | Displays the full sentence formed so far                 |
| **Chatbot Response** | Displays the AI response from Google Gemini              |
| **Test 3D Display**  | Emits a test letter to verify the 3D AR overlay          |
| **Toggle 3D View**   | Shows/hides the Three.js AR overlay on the webcam feed   |

### Signing Tips

- Sign **letters A–Z** to build words letter by letter
- Sign **space** to finalize the current word and add it to the sentence
- Sign **del** to delete the last letter (or last word if the current word is empty)
- Hold a gesture steady for a moment — a debounce/cooldown system prevents duplicate detections
- After a pause (`SENTENCE_TIMEOUT = 2.0s`), the sentence is sent to the Gemini chatbot

---

## Standalone Real-time Inference

A standalone inference script (without the web server) is available in `temp/real_time_inference.py`. It opens a local OpenCV window and prints recognized letters to the terminal:

```bash
python temp/real_time_inference.py
```

Press **Q** to quit.

---

## Model Details

The gesture recognition model is a simple dense (fully connected) neural network:

| Layer       | Units | Activation |
|-------------|-------|------------|
| Input       | 63    | —          |
| Dense       | 128   | ReLU       |
| Dropout     | 0.5   | —          |
| Dense       | 64    | ReLU       |
| Dense (out) | 29    | Softmax    |

- **Input**: 63 features — 21 hand landmarks × 3 coordinates (x, y, z) from MediaPipe
- **Output**: 29 classes — A–Z (26), `del`, `nothing`, `space`
- **Confidence threshold**: 0.8 (predictions below this are ignored)

---

## Configuration

Key parameters can be adjusted in `server.py`:

| Parameter            | Default | Description                                      |
|----------------------|---------|--------------------------------------------------|
| `DEBOUNCE_TIME`      | 0.5s    | Minimum time between repeated letter detections |
| `COOLDOWN_TIME`      | 1.0s    | Cooldown between any gesture updates            |
| `SPACE_COOLDOWN`     | 1.0s    | Cooldown between space gestures                 |
| `CONFIDENCE_THRESHOLD` | 0.8   | Minimum model confidence to accept a prediction |
| `SENTENCE_TIMEOUT`   | 2.0s    | Idle time before sending sentence to chatbot    |

---

## License

This project is open source. See the repository for license details.
