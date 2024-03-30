import cv2
import numpy as np
import subprocess
import sqlite3
import os
import time

from time import sleep

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

conn = sqlite3.connect('face_database.db')
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS faces
             (id INTEGER PRIMARY KEY, name TEXT)''')
def image_capture():
    rm_command = "rm -rf faces/cache.jpg"
    libcamera_command = ["libcamera-still", "-o", "faces/cache.jpg", "--nopreview", "--timeout","10"]
    try:
        result = subprocess.run(rm_command, shell=True)
        if result.returncode != 0:
            raise RuntimeError(f"Error removing cache image: {result.stderr}")
    except Exception as e:
        raise RuntimeError(f"Error during cleanup: {e}")
    try:
        result = subprocess.run(libcamera_command)
        if result.returncode != 0:
            raise RuntimeError(f"Error capturing image: {result.stderr}")
    except Exception as e:
        raise RuntimeError(f"Error during image capture: {e}")
        
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
        face_roi = image[y:y + h, x:x + w]
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

        c.execute("SELECT id, name FROM faces")
        for row in c.fetchall():
            face_id, name = row
            face_path = "faces/{}.jpg".format(face_id)

            try:
                face_image = cv2.imread(face_path, cv2.IMREAD_GRAYSCALE)
            except FileNotFoundError:
                print(f"Error: Face image not found: {face_path}")
                continue
                
            if face_image is None:
                print(f"Error: Could not read face image: {face_path}")
                continue  
                
            if face_image.shape[0] > face_gray.shape[0] or face_image.shape[1] > face_gray.shape[1]:
                print("Template is larger than image. Resizing...")
                template = cv2.resize(face_image, (face_gray.shape[1], face_gray.shape[0]))
            else:
                template = face_image 

            match = cv2.matchTemplate(template, face_gray, cv2.TM_CCOEFF_NORMED)
            confidence = np.max(match)
            threshold = 0.8

            if confidence > threshold:
                cv2.putText(image, name, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    return image


directory = "faces"
if not os.path.exists(directory):
    os.makedirs(directory)


while True:
    image_capture()
    image_path = "faces/cache.jpg"
    frame = cv2.imread(image_path)

    recognize_face(frame)

    cv2.putText(frame, "Press 'r' to register a new face with a name.", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)
    #time.sleep(0.1)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'):
        name = input("Enter the name to register: ")
        store_face(name, frame)

cv2.destroyAllWindows()

conn.close()
