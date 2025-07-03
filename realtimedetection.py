import cv2
from keras.models import model_from_json
import numpy as np

# Load model
json_file = open("face detection.json", "r")
model_json = json_file.read()
json_file.close()

model = model_from_json(model_json)
model.load_weights("face detection.h5")

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Define labels
labels = {0: 'angry', 1: 'disgust', 2: 'fearful', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprised'}

# Function to process image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    feature = feature / 255.0  # Normalize pixel values
    return feature

# Start webcam
webcam = cv2.VideoCapture(0)

# Check if camera opened successfully
if not webcam.isOpened():
    print("❌ Could not open webcam")
    exit()

while True:
    ret, im = webcam.read()
    if not ret:
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    try:
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            img = extract_features(roi)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]

            # Draw rectangle and label
            cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(im, prediction_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Real-time Facial Emotion Detection", im)
        
        # Exit if ESC is pressed
        if cv2.waitKey(1) == 27:
            break
    except cv2.error:
        pass

# Clean up
webcam.release()
cv2.destroyAllWindows()
