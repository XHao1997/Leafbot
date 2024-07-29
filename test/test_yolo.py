import numpy as np
import cv2

# Initialize the video capture object to capture from the default camera (usually the webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly, ret is True
    if not ret:
        print("Error: Couldn't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    cv2.imshow('Webcam Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close the window
cap.release()
cv2.destroyAllWindows()
