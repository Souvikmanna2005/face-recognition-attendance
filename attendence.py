import cv2
import os
import csv
import time
from datetime import datetime

TRAINER_FILE = "trainer/trainer.yml"
LABELS_FILE = "labels.txt"
ATTENDANCE_FILE = "attendance.csv"

# Load labels
def load_labels():
    labels = {}
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
            for line in f:
                id_str, name = line.strip().split(",")
                labels[int(id_str)] = name
    return labels

def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])

    # avoid duplicate entries on same date
    with open(ATTENDANCE_FILE, "r") as f:
        rows = f.read().splitlines()
        if any(row.startswith(f"{name},{date}") for row in rows[1:]):
            return

    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, date, time_str])
    print(f"[INFO] Attendance marked for {name}")

def recognize():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_FILE)
    labels = load_labels()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cam = cv2.VideoCapture(0)
    print("[INFO] Starting recognition. Press 'q' to quit.")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            id_pred, conf = recognizer.predict(face)
            if conf < 70:  # lower = better
                name = labels.get(id_pred, "Unknown")
                mark_attendance(name)
            else:
                name = "Unknown"

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, f"{name} ({int(conf)})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize()
