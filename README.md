Face Recognition Attendance System

A Python + OpenCV project to mark attendance using face recognition.

📂 Project Structure

HACKATHON/

├─ dataset/              # Face images (one folder per person, e.g. "Rohit Sharma")

├─ attendence.py         # Runs recognition and marks attendance in CSV

├─ train.py              # Trains LBPH recognizer on dataset

├─ trainer/              # Stores trained model file

├─ attendance.csv        # Attendance log (Name, Date, Time)

└─ labels.txt            # ID ↔ Name mapping

🚀 How to Run

Install requirements

pip install opencv-python opencv-contrib-python numpy


Prepare dataset

Create a folder inside dataset/ with the person’s name.

Add their face images inside that folder.

Train model

python train.py


Start attendance

python attendence.py

📊 Output

Recognized faces are logged in attendance.csv with:

Name, Date, Time

Virat Kohli, 2025-09-11, 19:20:15

✨ Features

Real-time face recognition

Automatic CSV attendance marking

Easy to extend with new people

<img width="796" height="642" alt="Screenshot 2025-09-11 181251" src="https://github.com/user-attachments/assets/8ac3cb85-7286-4da0-883c-0d40307a8000" />
