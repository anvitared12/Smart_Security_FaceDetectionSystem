
import cv2
import json

import os
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

cascade_path = os.path.join(
    BASE_DIR,
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(cascade_path)

# Check whether classifier loaded successfully
if face_cascade.empty():
    raise RuntimeError(
        "Error: Haar Cascade XML could not be loaded!"
    )

print("Haar Cascade loaded successfully!")

# Load trained face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read("models/face_model.yml")

# Load resident names
with open("dataset/labels.json", "r") as f:
    names = json.load(f)

# Recognition threshold
# Lower LBPH distance means a closer match
THRESHOLD = 65

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Security system started.")
print("Press Q to exit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Unable to read frame.")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Improve image contrast
    gray = cv2.equalizeHist(gray)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:

        # Extract face region
        face = gray[y:y+h, x:x+w]

        # Resize face
        face = cv2.resize(face, (200, 200))

        # Predict resident ID and distance
        label, confidence = recognizer.predict(face)

        # Check whether person is recognized
        if confidence < THRESHOLD and str(label) in names:

            person_name = names[str(label)]

            status = "ENTRY ALLOWED"
            color = (0, 255, 0)

        else:

            person_name = "Unknown Person"

            status = "ENTRY REJECTED"
            color = (0, 0, 255)

        # Draw rectangle around face
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            color,
            3
        )

        # Display person's name
        cv2.putText(
            frame,
            person_name,
            (x, y-45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        # Display entry status
        cv2.putText(
            frame,
            status,
            (x, y-15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        # Display LBPH distance
        cv2.putText(
            frame,
            f"Distance: {confidence:.2f}",
            (x, y+h+25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

    cv2.imshow(
        "Smart Society Security System",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
