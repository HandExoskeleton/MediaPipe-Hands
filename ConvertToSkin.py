import cv2
import numpy as np

# Ask the user to choose between live video feed (1) and a video file (2)
choice = int(input("Enter 1 for live video feed or 2 for video file: "))

if choice == 1:
    # Capture video from your webcam
    cap = cv2.VideoCapture(0)
elif choice == 2:
    # Prompt the user to enter the video file path
    file_path = input("Enter the video file path: ")
    cap = cv2.VideoCapture(file_path)
else:
    print("Invalid choice. Please enter 1 for live video feed or 2 for video file.")
    exit()

# Define the codec for the output video (use 'avc1' for H.264 codec)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Define the output video file name
output_filename = "output.mp4"

# Get the video width and height
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the VideoWriter to save the output video
out = cv2.VideoWriter(output_filename, fourcc, 20.0, (width, height))  # Adjust frame size and frame rate as needed

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video capture failed. Exiting...")
        break

    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the color range for dark objects (adjust these values as needed)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 50])

    # Create a binary mask to isolate dark objects
    dark_mask = cv2.inRange(hsv_frame, lower_dark, upper_dark)

    # Replace the dark regions with a different color (e.g., skin tone in BGR format)
    frame[dark_mask > 0] = [172, 219, 255]  # Change to skin tone (BGR color format)

    # Write the frame with the replaced regions to the output video
    out.write(frame)

    # Display the frame with the modified regions
    cv2.imshow('Replaced Dark Objects', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release the VideoWriter and video capture
out.release()
cap.release()
cv2.destroyAllWindows()
