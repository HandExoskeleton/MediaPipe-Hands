import cv2
import numpy as np

# Ask the user to choose between live video feed (1) and a video file (2)
choice = int(input("Enter 1 for live video feed or 2 for a video file: "))

if choice == 1:
    # Capture video from your webcam
    cap = cv2.VideoCapture(0)
elif choice == 2:
    # Prompt the user to enter the video file path
    file_path = input("Enter the video file path: ")
    cap = cv2.VideoCapture(file_path)
else:
    print("Invalid choice. Please enter 1 for a live video feed or 2 for a video file.")
    exit()

# Define the codec for the output video (use 'avc1' for H.264 codec)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Define the output video file name
output_filename = "output_contour.mp4"

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

    # Define the color range for skin tone (adjust these values as needed)
    lower_skin = np.array([172, 219, 255])
    upper_skin = np.array([172, 219, 255])

    # Create a mask to isolate the skin tone regions
    skin_mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)

    # Create contours based on the skin tone mask
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original frame
    cv2.drawContours(frame, contours, -1, (0, 0, 0), 3)

    # Write the frame with the contours to the output video
    out.write(frame)

    # Display the frame with the modified regions and contours
    cv2.imshow('Skin Tone Regions with Contours', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release the VideoWriter and video capture
out.release()
cap.release()
cv2.destroyAllWindows()
