import os
import cv2
import mediapipe as mp
import csv
import numpy as np
from Analyze import process_coord
import time

# Prompt user for the file name
projName = input("Enter project name: ")

# Prompt user for input type
input_choice = 0
while input_choice != "1" and input_choice != "2":
    input_choice = input("Enter 1 for live feed and 2 for video: ")

# Establish input type
if input_choice == "1":
    input_type = 0
else:  # 2
    input_type = input("Enter video file path: ")

# Create a directory with the project name
os.makedirs(projName, exist_ok=True)
os.makedirs(projName + "/Left", exist_ok=True)
os.makedirs(projName + "/Right", exist_ok=True)

# Initialize MediaPipe Hands
hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.5, static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Open video capture
cap = cv2.VideoCapture(input_type)

# Get the video width and height
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define video writers
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
normal_video = cv2.VideoWriter(os.path.join(projName, projName + '_NoLandmarks.mp4'), fourcc, 20.0,
                               (width, height))
abstract_video = cv2.VideoWriter(os.path.join(projName, projName + '_AbstractLandmarks.mp4'), fourcc, 20.0,
                                 (width, height))
landmarks_video = cv2.VideoWriter(os.path.join(projName, projName + '_Landmarks.mp4'), fourcc,
                                  20.0, (width, height))
combined_video = cv2.VideoWriter(os.path.join(projName, projName + '_SideBySide.mp4'), fourcc,
                                 20.0, (width * 4, height * 1))

angle_video = cv2.VideoWriter(os.path.join(projName, projName + '_AngleSample.mp4'), fourcc, 20.0, (width, height))

# Open files for writing
file_left = open(os.path.join(projName + "/Left", projName + "_LeftHand_Coordinates.csv"), 'w', newline='')
writer_left = csv.writer(file_left)

file_right = open(os.path.join(projName + "/Right", projName + "_RightHand_Coordinates.csv"), 'w', newline='')
writer_right = csv.writer(file_right)

# Define columns
landmark_positions = []
for i in range(21):
    landmark_positions.extend(['Landmark ' + str(i + 1)])
landmark_positions.extend(["Time"])
writer_left.writerow(landmark_positions)
writer_right.writerow(landmark_positions)

# Record start time
startTime = time.time()

# Initialize previous landmark positions
previous_landmarks = {}  # For left and right hands

while True:
    # Read frame from video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    if input_type != 0:  # Video input -> flip again
        frame = cv2.flip(frame, 1)
    normal_frame = frame.copy()

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    # Create a black background image (for joint angle display)
    ang_frame = np.zeros_like(frame)

    # Create a black background image
    black_frame = np.zeros_like(frame)

    # Generate tuples for joint locations on fingers
    joint_tuples = [(2, 3, 4), (5, 6, 7), (6, 7, 8), (9, 10, 12), (10, 11, 12), (13, 14, 15), (14, 15, 16),
                    (17, 18, 19), (18, 19, 20)]

    # Generate tuples for joint locations of knuckles
    knuckle_tuples = [(1, 2, 3), (0, 5, 6), (0, 9, 10), (0, 13, 14), (0, 17, 18)]

    prev_theta = 0
    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        # Iterate over each detected hand
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

            # Get landmarks for the hand
            hand_x = [landmark.x * frame.shape[1] for landmark in hand_landmarks.landmark]
            hand_y = [landmark.y * frame.shape[0] for landmark in hand_landmarks.landmark]

            # Calculate the bounding box coordinates
            min_x = min(hand_x)
            max_x = max(hand_x)
            min_y = min(hand_y)
            max_y = max(hand_y)

            # Draw the bounding box with the correct color argument placement
            cv2.rectangle(frame, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)

            # Determine if it's the left or right hand
            handedness = results.multi_handedness[hand_idx].classification[0].label
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(black_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(ang_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

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
                cv2.putText(frame, f"Landmark {idx + 1}: ({x}, {y}, {z})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)

            # Compute joint angle per frame
            for joint in joint_tuples:
                # Generate first vector for line segment formed by landmarks
                vec1 = np.subtract(landmark_row[joint[0]], landmark_row[joint[1]])

                # Generate second vector for line segment formed by landmarks
                vec2 = np.subtract(landmark_row[joint[2]], landmark_row[joint[1]])

                # Compute cosine via dot product:
                X = np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)

                # Compute sine via cross product
                Y = np.linalg.norm(np.cross(vec1, vec2)) / np.linalg.norm(vec1) / np.linalg.norm(vec2)

                # Round off the angle
                theta = int(np.arctan2(Y, X) * 180 / np.pi)

                # Display joint angle on angle frame
                cv2.putText(ang_frame, f"{theta}", (landmark_row[joint[1]][0], landmark_row[joint[1]][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                '''# Test angular velocity vector 
                # generate "finger plane"
                norm = np.cross(vec2, vec1)
                # cross product between normal vector and vec2
                ang_vel_direction = np.cross(vec2, norm) 
                # calculate magnitude of angular velocity, scale vector accordingly
                # assuming that there is equal time between each frame, the difference in angle should be proportional to angular velocity!
                ang_vel_magnitude = np.linalg.norm(prev_theta - theta) / 5
                # Normalize
                ang_vel = ang_vel_direction / np.linalg.norm(ang_vel_direction) * ang_vel_magnitude
                # Calculate endpoint
                end_point = np.subtract(np.array(landmark_row[joint[2]]), ang_vel)
                start_point = landmark_row[joint[2]]
                if prev_theta != 0:
                    # Draw arrow 
                    cv2.arrowedLine(ang_frame, (int(start_point[0]), int(start_point[1])), (int(end_point[0]), int(end_point[1])), (0, 255, 0), 2)
                prev_theta = theta'''

            for joint in knuckle_tuples:
                # Generate first vector for line segment formed by landmarks (pointing towards the base of the hand)
                vec1 = np.subtract(landmark_row[joint[0]], landmark_row[joint[1]])

                # Generate second vector for line segment formed by landmarks
                vec2 = np.subtract(landmark_row[joint[2]], landmark_row[joint[1]])

                # Generate third vector for line segment formed by the next two landmarks after vec2
                vec3 = np.subtract(landmark_row[joint[2] + 1], landmark_row[joint[2]])

                # Generate plane formed by vec2 and vec3
                normal_vec = np.cross(vec2, vec3)

                # Project vec1 onto the plane
                vec1_proj = vec1 - (np.dot(normal_vec, vec1) / (np.linalg.norm(normal_vec) ** 2) * normal_vec)

                # Compute cosine via dot product:
                X = np.dot(vec1_proj, vec2) / np.linalg.norm(vec1_proj) / np.linalg.norm(vec2)

                # Compute sine via cross product:
                Y = np.linalg.norm(np.cross(vec1_proj, vec2)) / np.linalg.norm(vec1_proj) / np.linalg.norm(vec2)

                # Compute angle from dot product formula
                theta = int(np.arctan2(Y, X) * 180 / np.pi)

                # Display joint angle on angle frame
                cv2.putText(ang_frame, f"{theta}", (landmark_row[joint[1]][0], landmark_row[joint[1]][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Append timestamp to the landmark_row
            landmark_row.append(time.time() - startTime)  # current time - start time for calibration

            # Write row to the respective CSV file based on handedness
            if handedness == "Left":
                writer_left.writerow(landmark_row)
                if hand_idx not in previous_landmarks:
                    previous_landmarks[hand_idx] = None  # Initialize previous landmarks for the left hand
            elif handedness == "Right":
                writer_right.writerow(landmark_row)
                if hand_idx not in previous_landmarks:
                    previous_landmarks[hand_idx] = None  # Initialize previous landmarks for the right hand

            # Calculate velocity field
            if previous_landmarks[hand_idx] is not None:
                velocity_field = []
                for prev_landmark, curr_landmark in zip(previous_landmarks[hand_idx], landmark_row[:-1]):
                    prev_x, prev_y, prev_z = prev_landmark
                    curr_x, curr_y, curr_z = curr_landmark
                    delta_x = curr_x - prev_x
                    delta_y = curr_y - prev_y
                    velocity_field.append((delta_x, delta_y))

                # Draw velocity field on the black frame
                for velocity_vector, curr_landmark in zip(velocity_field, landmark_row[:-1]):
                    curr_x, curr_y, _ = curr_landmark
                    delta_x, delta_y = velocity_vector
                    end_point = (curr_x + delta_x, curr_y + delta_y)
                    cv2.arrowedLine(black_frame, (curr_x, curr_y), end_point, (0, 0, 255), 2)

                    # Draw velocity vectors on the frame
                    cv2.arrowedLine(frame, (curr_x, curr_y), end_point, (0, 0, 255), 2)

            # Update previous_landmarks
            previous_landmarks[hand_idx] = landmark_row[:-1]

    # Write frame to output videos based on handedness
    if results.multi_handedness:
        for handedness in results.multi_handedness:
            hand_label = handedness.classification[0].label
            abstract_video.write(black_frame)
            landmarks_video.write(frame)
            angle_video.write(ang_frame)

    # Original frame without landmarks
    normal_video.write(normal_frame)

    # Side by side video
    combined_frame = np.concatenate((black_frame, frame, ang_frame, normal_frame), axis=1)
    combined_video.write(combined_frame)

    angle_video.write(ang_frame)

    # Show the frame
    cv2.imshow('MediaPipe Hands', frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Run analysis on left and right hand coordinates separately
process_coord(projName + "_LeftHand", os.path.join(projName + "/Left", projName + "_LeftHand_Coordinates.csv"),
              projName + "/Left")
process_coord(projName + "_RightHand", os.path.join(projName + "/Right", projName + "_RightHand_Coordinates.csv"),
              projName + "/Right")

# Release resources
cap.release()
normal_video.release()
abstract_video.release()
landmarks_video.release()
combined_video.release()
cv2.destroyAllWindows()
hands.close()

# Close files
file_left.close()
file_right.close()
