import cv2
import requests
from bs4 import BeautifulSoup

# Load license plate detection model (similar to previous code)
plate_cascade = cv2.CascadeClassifier(r'C:\Users\MEELKO101\Documents\GitHub\First-Project-Computer-Vision\car tag reader\haarcascade_license_plate_rus_16stages.xml')

# Initialize video capture
video_path = 'D:\Dropbox\Personal\mdc cris\spring 2024\CV\car project\20240212_132205.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = 'output_video_with_details.avi'
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# Web scraping setup (replace with actual URL)
web_url = 'https://findbyplate.com/'

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # License plate detection (similar to previous code)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in plates:
        plate_roi = frame[y:y + h, x:x + w]

        # Extract license plate number using OCR (Tesseract or other library)
        plate_number = extract_plate_number(plate_roi)  # Implement this function

        # Web scraping: Get car details from the webpage
        response = requests.get(web_url + plate_number)
        soup = BeautifulSoup(response.content, 'html.parser')
        car_brand = soup.find('span', class_='brand').text
        car_model = soup.find('span', class_='model').text
        car_year = soup.find('span', class_='year').text

        # Annotate frame with car details
        cv2.putText(frame, f'{car_brand} {car_model} ({car_year})', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Write frame to output video
    out.write(frame)

    # Display processed frame (optional)
    cv2.imshow('Car Details Annotation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved as {output_path}")
