import cv2
import time
import numpy as np
from tensorflow.keras.models import load_model

def create_hand_silhouette(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
        return mask
    return gray  # Fallback to gray if no contours found

# Load the trained CNN model
model = load_model('hand_gesture_model.keras')

# Map gesture indices to their respective letters.
index_to_letter = {
    0: 'a',
    1: 'b',
    2: 'c',
    3: 'd',
    4: 'e'
}

# Define constants for the ROI and capture settings
roi_size = (140, 140)
border_thickness = 2
prediction_interval = 1

# Start video capture
cap = cv2.VideoCapture(0)
last_prediction_time = time.time()
gesture_label = "Waiting for gesture..."

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    upper_left = ((width - roi_size[0]) // 2, (height - roi_size[1]) // 2)
    bottom_right = (upper_left[0] + roi_size[0], upper_left[1] + roi_size[1])

    cv2.rectangle(frame, upper_left, bottom_right, (255, 0, 0), 2)

    current_time = time.time()
    if current_time - last_prediction_time >= prediction_interval:
        last_prediction_time = current_time

        cropped_frame = frame[upper_left[1] + border_thickness:bottom_right[1] - border_thickness,
                              upper_left[0] + border_thickness:bottom_right[0] - border_thickness]

        # Apply silhouette filtering
        filtered_roi = create_hand_silhouette(cropped_frame)

        # Resize and prepare for model prediction
        resized_frame = cv2.resize(filtered_roi, roi_size)
        normalized_frame = resized_frame / 255.0
        input_frame = np.expand_dims(normalized_frame, axis=0)
        input_frame = np.expand_dims(input_frame, axis=-1)

        prediction = model.predict(input_frame)
        gesture_index = np.argmax(prediction)
        gesture_confidence = np.max(prediction)
        gesture_letter = index_to_letter[gesture_index]
        gesture_label = f'Gesture: {gesture_letter.upper()}, Confidence: {gesture_confidence:.2f}'

    cv2.rectangle(frame, (10, 30), (400, 60), (50, 50, 50), -1)
    cv2.putText(frame, gesture_label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Real-Time Gesture Recognition', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()