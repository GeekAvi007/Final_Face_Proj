import cv2
import numpy as np
import sqlite3
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

conn = sqlite3.connect('face_database.db')
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS faces
             (id INTEGER PRIMARY KEY, name TEXT)''')

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces

def store_face(name, face):
    c.execute("INSERT INTO faces (name) VALUES (?)", (name,))
    conn.commit()

    face_id = c.lastrowid
    cv2.imwrite("faces/{}.jpg".format(face_id), face)

def recognize_face(image):
    faces = detect_faces(image)

    for (x, y, w, h) in faces:
        face_roi = image[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)

        c.execute("SELECT id, name FROM faces")
        for row in c.fetchall():
            face_id, name = row
            face_image = cv2.imread("faces/{}.jpg".format(face_id), cv2.IMREAD_GRAYSCALE)

            match = cv2.matchTemplate(face_image, face_gray, cv2.TM_CCOEFF_NORMED)
            confidence = np.max(match)
            threshold = 0.8
            if confidence > threshold:
                cv2.putText(image, name, (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

directory = "faces"
if not os.path.exists(directory):
    os.makedirs(directory)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    recognize_face(frame)

    cv2.putText(frame, "Press 'r' to register a new face with a name.", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'):
        name = input("Enter the name to register: ")
        store_face(name, frame)

cap.release()
cv2.destroyAllWindows()

conn.close()
