
import cv2
import os
import json
import numpy as np

dataset_path = "dataset"
model_path = "models"

os.makedirs(model_path, exist_ok=True)

faces = []
labels = []

# Read resident names
with open("dataset/labels.json", "r") as f:
    names = json.load(f)

# Read all registered face images
for person_id in os.listdir(dataset_path):

    person_folder = os.path.join(
        dataset_path,
        person_id
    )

    if not os.path.isdir(person_folder):
        continue

    if not person_id.isdigit():
        continue

    label = int(person_id)

    for image_name in os.listdir(person_folder):

        if not image_name.lower().endswith(".jpg"):
            continue

        image_path = os.path.join(
            person_folder,
            image_name
        )

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            continue

        # Standardize image size
        image = cv2.resize(image, (200, 200))

        faces.append(image)
        labels.append(label)

# Check dataset
if len(faces) == 0:
    raise ValueError("No training images found!")

# Convert labels into NumPy array
labels = np.array(labels, dtype=np.int32)

# Create LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer.create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

# Train the model
recognizer.train(faces, labels)

# Save trained model
recognizer.save(
    os.path.join(model_path, "face_model.yml")
)

print("Training completed successfully!")
print("Total images:", len(faces))
print("Total residents:", len(names))
print("Model saved successfully!")
