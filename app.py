from flask import Flask, request, jsonify
from PIL import Image
import os
import uuid
import cv2
import numpy as np
import mediapipe as mp
from flask_cors import CORS
import io

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Create a folder for sessions
SESSIONS_FOLDER = './sessions'
os.makedirs(SESSIONS_FOLDER, exist_ok=True)

# Mediapipe Face Mesh initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to calculate the Eye Aspect Ratio (EAR) to detect blinks
def calculate_ear(landmarks, image_width, image_height):
    # Left eye landmarks
    left_eye_top = np.array([landmarks[159].x * image_width, landmarks[159].y * image_height])  # Top of left eye
    left_eye_bottom = np.array([landmarks[145].x * image_width, landmarks[145].y * image_height])  # Bottom of left eye
    left_eye_left = np.array([landmarks[33].x * image_width, landmarks[33].y * image_height])  # Left corner
    left_eye_right = np.array([landmarks[133].x * image_width, landmarks[133].y * image_height])  # Right corner

    # Right eye landmarks
    right_eye_top = np.array([landmarks[386].x * image_width, landmarks[386].y * image_height])  # Top of right eye
    right_eye_bottom = np.array([landmarks[374].x * image_width, landmarks[374].y * image_height])  # Bottom of right eye
    right_eye_left = np.array([landmarks[362].x * image_width, landmarks[362].y * image_height])  # Left corner
    right_eye_right = np.array([landmarks[263].x * image_width, landmarks[263].y * image_height])  # Right corner

    # Calculate the distances between the vertical eye landmarks (top and bottom) and horizontal landmarks (left and right)
    left_ear = np.linalg.norm(left_eye_top - left_eye_bottom) / np.linalg.norm(left_eye_left - left_eye_right)
    right_ear = np.linalg.norm(right_eye_top - right_eye_bottom) / np.linalg.norm(right_eye_left - right_eye_right)

    # Average the EAR for both eyes
    ear = (left_ear + right_ear) / 2.0
    return ear

# Function to detect smile based on teeth visibility with dynamic threshold
def detect_smile(landmarks, image_width, image_height):
    upper_lip_inner = landmarks[13]  # Upper lip midpoint (inner)
    lower_lip_inner = landmarks[14]  # Lower lip midpoint (inner)
    chin = landmarks[152]  # Use chin point to measure face height

    lip_distance = np.linalg.norm(np.array([upper_lip_inner.x * image_width, upper_lip_inner.y * image_height]) - 
                                  np.array([lower_lip_inner.x * image_width, lower_lip_inner.y * image_height]))

    face_height = np.linalg.norm(np.array([upper_lip_inner.x * image_width, upper_lip_inner.y * image_height]) - 
                                 np.array([chin.x * image_width, chin.y * image_height]))

    smile_threshold = face_height * 0.15

    return lip_distance > smile_threshold

# Function to detect face orientation
def detect_face_orientation(landmarks, image_width):
    nose_tip = landmarks[1]
    left_cheek = landmarks[234]  # Left side of the face
    right_cheek = landmarks[454]  # Right side of the face
    
    face_center = (left_cheek.x + right_cheek.x) / 2
    nose_pos = nose_tip.x
    
    if abs(nose_pos - face_center) < 0.08:
        return "Front"
    elif nose_pos < face_center - 0.08:
        return "Right"
    else:
        return "Left"

# Liveliness Test API
@app.route('/liveliness_test', methods=['POST'])
def liveliness_test():
    session_id = request.form.get('session_id', str(uuid.uuid4()))
    
    # Process the image first, create session folder only if the image is valid
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    
    # Convert the image file to a numpy array
    try:
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        image_np = np.array(image)
    except Exception as e:
        return jsonify({"error": f"Image processing failed: {str(e)}"}), 500

    # Convert the image to BGR format for OpenCV
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Get the dimensions of the image
    image_height, image_width, _ = image_np.shape

    # Convert the image to RGB for Mediapipe processing
    rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # Perform face detection and landmarking using Mediapipe
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return jsonify({"error": "No face detected"}), 400

    # If face is detected, create the session folder (only now, after successful detection)
    session_folder = os.path.join(SESSIONS_FOLDER, session_id)
    os.makedirs(session_folder, exist_ok=True)

    # Process the detected face
    for face_landmarks in results.multi_face_landmarks:
        landmarks = face_landmarks.landmark

        # Detect face orientation (left, right, or front)
        orientation = detect_face_orientation(landmarks, image_width)

        # Detect smile using the existing detect_smile function
        smile_detected = detect_smile(landmarks, image_width, image_height)

        # Detect blink using Eye Aspect Ratio (EAR)
        ear = calculate_ear(landmarks, image_width, image_height)
        blink_detected = ear < 0.28  # Blink detected if EAR is less than a certain threshold (typically 0.2)

        # Save the image only if the orientation is "Front"
        if orientation == "Front":
            frontal_image_path = os.path.join(session_folder, "frontal_image.jpg")
            cv2.imwrite(frontal_image_path, image_np)
            
            # Generate a public URL-friendly link (relative path to serve through your API)
            frontal_image_url = f"/sessions/{session_id}/frontal_image.jpg"
        else:
            frontal_image_path = None
            frontal_image_url = None

        # Return the results as JSON, including session_id and the fixed link
        result = {
            "session_id": session_id,
            "orientation": orientation,
            "smile": "Yes" if smile_detected else "No",
            "blink": "Yes" if blink_detected else "No",
            "frontal_image_url": frontal_image_url
        }
        return jsonify(result)

    return jsonify({"error": "Face detection failed", "session_id": session_id}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
