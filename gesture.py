import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Initialize MediaPipe drawing
mp_drawing = mp.solutions.drawing_utils

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Flip the frame horizontally for a laterals view
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Count fingers based on landmarks
            finger_count = 0
            # Check which fingers are up
            # Landmark index (using MediaPipe's indexing)
            # Thumb (4), Index (8), Middle (12), Ring (16), Pinky (20)

            # Thumb
            if hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y:  # Thumb up
                finger_count += 1

            # Index
            if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:  # Index up
                finger_count += 1

            # Middle
            if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y:  # Middle up
                finger_count += 1

            # Ring
            if hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y:  # Ring up
                finger_count += 1

            # Pinky
            if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y:  # Pinky up
                finger_count += 1

            # Define gesture based on finger count
            if finger_count == 1:
                gesture = "One Finger"
            elif finger_count == 2:
                gesture = "Two Fingers"
            elif finger_count == 5:
                gesture = "Five Fingers"
            elif finger_count == 0:
                gesture = "Fist"
            else:
                gesture = "Other"

            # Display the number of fingers detected
            cv2.putText(frame, f'Fingers: {finger_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # Display the detected gesture
            cv2.putText(frame, f'Gesture: {gesture}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Print to console
            print(f'Fingers: {finger_count}, Gesture: {gesture}')

    # Display the frame
    cv2.imshow('Hand Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
