import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Read the first frame
ret, first_frame = cap.read()
prev_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Create a blank canvas for the trail effect
canvas = np.zeros_like(first_frame, dtype=np.float32)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference between the current frame and first frame
    frame_diff = cv2.absdiff(prev_frame_gray, gray)

    # Threshold the difference to get the motion
    thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)[1]

    # Dilate the thresholded image to fill in holes
    dilated = cv2.dilate(thresh, None, iterations=2)

    # Find contours of the moving parts
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Apply the trail effect by blending the frames
    overlay = frame.copy()  # Create a copy of the frame for overlaying
    alpha = 0.1  # Adjust the alpha value to control the trail effect intensity

    # Draw contours on the frame and update the overlay
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Filter small contours
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), -1)

    # Apply the trail effect by blending the overlay with the canvas
    canvas = cv2.addWeighted(overlay.astype(float), alpha, canvas.astype(float), 1 - alpha, 0)

    # Convert back to the appropriate data type
    canvas = canvas.astype(np.uint8)

    # Apply filters or color transformations
    blurred = cv2.GaussianBlur(canvas, (15, 15), 0)
    grayscale = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    inverse = cv2.bitwise_not(grayscale)
    artistic_display = cv2.cvtColor(inverse, cv2.COLOR_GRAY2BGR)

    # Display the resulting artistic display
    cv2.imshow('Motion Capture Artistic Display', artistic_display)

    # Update the previous frame
    prev_frame_gray = gray

    # Break the loop with the 'ESC' key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()