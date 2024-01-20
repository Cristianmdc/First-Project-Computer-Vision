import cv2
import numpy as np

def nothing(x):
    pass

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create a window
cv2.namedWindow('Dynamic Color Filtering')

# Create trackbars for color change
cv2.createTrackbar('Lower-Hue', 'Dynamic Color Filtering', 0, 180, nothing)
cv2.createTrackbar('Lower-Saturation', 'Dynamic Color Filtering', 0, 255, nothing)
cv2.createTrackbar('Lower-Value', 'Dynamic Color Filtering', 0, 255, nothing)
cv2.createTrackbar('Upper-Hue', 'Dynamic Color Filtering', 180, 180, nothing)
cv2.createTrackbar('Upper-Saturation', 'Dynamic Color Filtering', 255, 255, nothing)
cv2.createTrackbar('Upper-Value', 'Dynamic Color Filtering', 255, 255, nothing)
cv2.createTrackbar('Lower-Blue', 'Dynamic Color Filtering', 0, 255, nothing)
cv2.createTrackbar('Upper-Blue', 'Dynamic Color Filtering', 255, 255, nothing)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get trackbar positions for Hue, Saturation, and Value
    lh = cv2.getTrackbarPos('Lower-Hue', 'Dynamic Color Filtering')
    ls = cv2.getTrackbarPos('Lower-Saturation', 'Dynamic Color Filtering')
    lv = cv2.getTrackbarPos('Lower-Value', 'Dynamic Color Filtering')
    uh = cv2.getTrackbarPos('Upper-Hue', 'Dynamic Color Filtering')
    us = cv2.getTrackbarPos('Upper-Saturation', 'Dynamic Color Filtering')
    uv = cv2.getTrackbarPos('Upper-Value', 'Dynamic Color Filtering')

    # Get trackbar positions for Blue color channel
    lb = cv2.getTrackbarPos('Lower-Blue', 'Dynamic Color Filtering')
    ub = cv2.getTrackbarPos('Upper-Blue', 'Dynamic Color Filtering')

    # Define ranges of color in HSV and BGR
    lower_color_hsv = np.array([lh, ls, lv])
    upper_color_hsv = np.array([uh, us, uv])
    lower_color_bgr = np.array([lb, 0, 0])
    upper_color_bgr = np.array([ub, 255, 255])

    # Threshold the HSV image to get only selected colors
    mask_hsv = cv2.inRange(hsv, lower_color_hsv, upper_color_hsv)

    # Threshold the BGR image to get only selected colors
    mask_bgr = cv2.inRange(frame, lower_color_bgr, upper_color_bgr)

    # Bitwise-AND masks with original image for each color space
    res_hsv = cv2.bitwise_and(frame, frame, mask=mask_hsv)
    res_bgr = cv2.bitwise_and(frame, frame, mask=mask_bgr)

    # Display the resulting frames
    cv2.imshow('Dynamic Color Filtering (HSV)', res_hsv)
    cv2.imshow('Dynamic Color Filtering (BGR)', res_bgr)

    # Break the loop with the 'ESC' key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
