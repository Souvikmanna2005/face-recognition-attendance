import cv2
import os
import numpy as np

DATASET_DIR = "dataset"
TRAINER_DIR = "trainer"
TRAINER_FILE = os.path.join(TRAINER_DIR, "trainer.yml")

def get_images_and_labels(dataset_path):
    faces = []
    ids = []
    names = {}
    current_id = 0

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue

        print(f"[INFO] Processing {person_name}...")
        names[current_id] = person_name

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            faces_detected = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)
            if len(faces_detected) == 0:
                # if no face detected, still add resized full image
                face = cv2.resize(img, (200, 200))
                faces.append(face)
                ids.append(current_id)
            else:
                for (x, y, w, h) in faces_detected:
                    face = img[y:y+h, x:x+w]
                    face = cv2.resize(face, (200, 200))
                    faces.append(face)
                    ids.append(current_id)

        current_id += 1

    return faces, ids, names

def train():
    os.makedirs(TRAINER_DIR, exist_ok=True)

    print("[INFO] Loading dataset...")
    faces, ids, names = get_images_and_labels(DATASET_DIR)

    print(f"[INFO] Training {len(set(ids))} persons with {len(faces)} images...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    recognizer.write(TRAINER_FILE)

    # save names mapping
    with open("labels.txt", "w") as f:
        for i in names:
            f.write(f"{i},{names[i]}\n")

    print("[INFO] Training complete. Model saved.")

if __name__ == "__main__":
    train()
