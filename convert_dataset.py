
import cv2
import os
import json

SOURCE = "data"
DESTINATION = "dataset"

os.makedirs(DESTINATION, exist_ok=True)

names = {}

for i in range(1, 21):

    source_folder = os.path.join(SOURCE, f"s{i}")
    destination_folder = os.path.join(
        DESTINATION, str(i)
    )

    os.makedirs(destination_folder, exist_ok=True)

    names[str(i)] = f"Person_{i}"

    for j in range(1, 11):

        image_path = os.path.join(
            source_folder, f"{j}.pgm"
        )

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            print("Missing:", image_path)
            continue

        image = cv2.resize(image, (200, 200))

        cv2.imwrite(
            os.path.join(destination_folder, f"{j}.jpg"),
            image
        )

with open(
    os.path.join(DESTINATION, "labels.json"), "w"
) as f:
    json.dump(names, f, indent=4)

print("Dataset conversion completed!")
