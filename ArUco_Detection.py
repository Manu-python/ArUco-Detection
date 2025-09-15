# ArUco marker detection code
import cv2
import cv2.aruco as aruco
import numpy as np

# Initialize the camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Check if the camera was opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Load the predefined ArUco dictionary (using a recent OpenCV syntax)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to grayscale for marker detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the markers in the grayscale image
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # If markers are detected
    if ids is not None:
        # Draw detected markers on the original color frame
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Print marker IDs and their corner coordinates
        for i in range(len(ids)):
            print(f"Marker ID: {ids[i][0]}")
            # The corners array has a specific structure: [ [ [x1,y1], [x2,y2], [x3,y3], [x4,y4] ] ]
            print(f"Corners: {corners[i][0]}")

    # Show the output frame with detected markers
    cv2.imshow('Detected ArUco markers', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()