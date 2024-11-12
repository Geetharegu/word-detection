import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the pre-trained model
model = load_model('action.h5')

# Define the actions (labels) you trained the model on
actions = np.array(['Hello', 'Thanks', 'Love You'])

# Initialize Mediapipe holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Initialize variables
sequence = []
sentence = []
predictions = []
threshold = 0.5

# Function to extract keypoints from the holistic results
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

# Function to perform Mediapipe detection and convert to BGR
def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image, results

# Function to visualize probabilities (optional)
def prob_viz(res, actions, image, colors):
    output_frame = image.copy()
    for i, action in enumerate(actions):
        cv2.rectangle(output_frame, (0, 60 + i * 40), (int(res[i] * 100), 90 + i * 40), colors[i], -1)
        cv2.putText(output_frame, f'{action}: {res[i]:.2f}', (5, 85 + i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

# Set colors for probability visualization
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (245, 16, 117)]

# OpenCV Video capture (Live feed)
cap = cv2.VideoCapture(0)

# Initialize Mediapipe Holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Make detections
        image, results = mediapipe_detection(frame, holistic)

        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Extract keypoints for this frame
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]  # Keep only the last 30 frames

        # Prediction logic: Make predictions after 30 frames are collected
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))

            # Visualization logic (show confidence)
            if np.unique(predictions[-10:])[0] == np.argmax(res):  # Ensure consistency in predictions
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]  # Limit sentence length

            # Visualize probabilities (optional)
            image = prob_viz(res, actions, image, colors)

        # Display recognized action in a rectangle on the frame
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('Real-time Action Recognition', image)

        # Break gracefully on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
