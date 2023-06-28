import os
import cv2
import mediapipe as mp
import csv
import numpy as np
from Analyze import process_coord
import time

# Prompt user for file name
projName = input("Enter project name: ")

# Create a directory with the project name
os.makedirs(projName, exist_ok=True)

# Open the file for writing
file = open(os.path.join(projName, projName + "_Coordinates.csv"), 'w', newline='')
writer = csv.writer(file)

# Define columns
landmark_positions = []
for i in range(21):
    landmark_positions.extend(['Landmark ' + str(i+1)])
landmark_positions.extend(["Time"])
writer.writerow(landmark_positions)

# Initialize MediaPipe Hands
hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.5, static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Open video capture
cap = cv2.VideoCapture(0)

# Record start time
startTime = time.time()

# Get the video width and height
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define video writers
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
landmarks_video = cv2.VideoWriter(os.path.join(projName, projName + '_LandmarksOnly.mp4'), fourcc, 20.0, (width, height))
velocity_video = cv2.VideoWriter(os.path.join(projName, projName + '_VelocityField.mp4'), fourcc, 20.0, (width, height))

# Initialize previous landmark positions
previous_landmarks = None

while True:
    # Read frame from video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    # Create a black background image
    black_frame = np.zeros_like(frame)

    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        # Iterate over each detected hand
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(black_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Array for storing each position instance (row)
            landmark_row = []

            # Display position of each landmark
            for idx, landmark in enumerate(hand_landmarks.landmark):
                # Get landmark position in 2D
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                # Get landmark position in 3D
                z = landmark.z
                # Save position to landmark_row
                landmark_row.append((x, y, z))

                # Display position as text
                cv2.putText(frame, f"Landmark {idx+1}: ({x}, {y}, {z})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Append timestamp to the landmark_row
            landmark_row.append(time.time()-startTime) #current time - start time for calibration

            # Write row to csv file
            writer.writerow(landmark_row)

            # Calculate velocity field
            if previous_landmarks is not None:
                velocity_field = []
                for prev_landmark, curr_landmark in zip(previous_landmarks, landmark_row[:-1]):
                    prev_x, prev_y, _ = prev_landmark
                    curr_x, curr_y, _ = curr_landmark
                    delta_x = curr_x - prev_x
                    delta_y = curr_y - prev_y
                    velocity_field.append((delta_x, delta_y))

                # Draw velocity field on the black frame
                for velocity_vector, curr_landmark in zip(velocity_field, landmark_row[:-1]):
                    curr_x, curr_y, _ = curr_landmark
                    delta_x, delta_y = velocity_vector
                    end_point = (curr_x + delta_x, curr_y + delta_y)
                    cv2.arrowedLine(black_frame, (curr_x, curr_y), end_point, (0, 0, 255), 2)

            # Update previous_landmarks
            previous_landmarks = landmark_row[:-1]

    # Write frame to output videos
    landmarks_video.write(black_frame)
    velocity_video.write(frame)

    # Show the frame
    cv2.imshow('MediaPipe Hands', frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Run analysis
process_coord(projName, os.path.join(projName, projName + "_Coordinates.csv"))  #passing it as a path rather than object

# Release resources
cap.release()
landmarks_video.release()
velocity_video.release()
cv2.destroyAllWindows()
hands.close()

# Close file
file.close()
