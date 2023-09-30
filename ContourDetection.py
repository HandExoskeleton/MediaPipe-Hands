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

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video capture failed. Exiting...")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur and thresholding to create a binary image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and draw hand contours
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Adjust the area threshold as needed
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)

    # Display the frame with hand contours
    cv2.imshow('Hand Contours', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
