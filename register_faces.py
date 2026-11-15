
import cv2
import os
import json

# Create dataset directory
os.makedirs("dataset", exist_ok=True)

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Get resident information
person_id = input("Enter Resident ID: ").strip()
person_name = input("Enter Resident Name: ").strip()

if not person_id.isdigit() or not person_name:
    raise ValueError("Enter a numeric ID and a valid name.")

# Create folder for resident
person_folder = os.path.join("dataset", person_id)
os.makedirs(person_folder, exist_ok=True)

# Load existing resident names
labels_file = "dataset/labels.json"

if os.path.exists(labels_file):
    with open(labels_file, "r") as f:
        labels = json.load(f)
else:
    labels = {}

# Check ID
if person_id in labels and labels[person_id] != person_name:
    raise ValueError("This ID is already assigned to another resident.")

labels[person_id] = person_name

with open(labels_file, "w") as f:
    json.dump(labels, f, indent=4)

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

count = 0
max_images = 50

print("Press SPACE to capture a face.")
print("Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Improve contrast
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Face Detected",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"Images: {count}/{max_images}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.imshow("Resident Registration", frame)

    key = cv2.waitKey(1) & 0xFF

    # Capture only when SPACE is pressed
    if key == ord(' '):

        if len(faces) == 0:
            print("No face detected. Try again.")
            continue

        if count >= max_images:
            print("Maximum images captured.")
            continue

        # Select largest detected face
        x, y, w, h = max(
            faces,
            key=lambda face: face[2] * face[3]
        )

        face = gray[y:y+h, x:x+w]

        # Resize face to standard dimensions
        face = cv2.resize(face, (200, 200))

        image_path = os.path.join(
            person_folder,
            f"{count}.jpg"
        )

        cv2.imwrite(image_path, face)

        count += 1

        print(f"Captured image {count}/{max_images}")

    if key == ord('q'):
        break

    if count >= max_images:
        print("Registration completed.")
        break

cap.release()
cv2.destroyAllWindows()

print(f"Registration completed for {person_name}")
