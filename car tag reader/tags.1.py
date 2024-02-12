import cv2

# Load the pre-trained license plate detection model (you can replace this with your own model)
plate_cascade = cv2.CascadeClassifier('path/to/haarcascade_russian_plate_number.xml')

# Open the video file (replace 'path/to/your/video.mp4' with the actual path)
video_path = 'c:\Dropbox\Personal\mdc cris\spring 2024\CV\car project\20240212_132205.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize video writer to save the output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = 'output_video_with_plates.avi'
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for license plate detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect license plates
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in plates:
        # Draw bounding box around license plate
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract license plate region
        plate_roi = frame[y:y + h, x:x + w]

        # Save license plate images (you can customize the filenames)
        plate_filename = f'plate_{x}_{y}.jpg'
        cv2.imwrite(plate_filename, plate_roi)

    # Write the frame with bounding boxes to the output video
    out.write(frame)

    # Display the processed frame (optional)
    cv2.imshow('License Plate Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"License plate images saved as {plate_filename}")
print(f"Output video saved as {output_path}")
