import cv2
import sqlite3
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN

# Connect to SQLite database
conn = sqlite3.connect('faces.db')
c = conn.cursor()

# Create table to store faces if not exists
c.execute('''CREATE TABLE IF NOT EXISTS faces
             (name TEXT, embedding TEXT)''')
conn.commit()

# Load pre-trained MTCNN for face detection
detector = MTCNN()

# Load pre-trained FaceNet model for face recognition
facenet_model = load_model('facenet_keras.h5')

# Function to draw rectangle around face and display name
def draw_name_and_border(frame, x, y, w, h, name):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow rectangle around face
    cv2.putText(frame, name, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # Name below face

# Function to register a new face
def register_face(name, embedding):
    c.execute("INSERT INTO faces (name, embedding) VALUES (?, ?)", (name, embedding))
    conn.commit()

# Function to recognize face
def recognize_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (160, 160))  # Resize image for FaceNet input
    face = face / 255.0  # Normalize pixel values

    # Extract face embedding using FaceNet model
    embedding = facenet_model.predict(np.expand_dims(face, axis=0))[0]

    # Compare face embedding with embeddings in the database
    c.execute("SELECT name, embedding FROM faces")
    for row in c.fetchall():
        saved_embedding = np.fromstring(row[1], dtype=float, sep=',')
        distance = np.linalg.norm(embedding - saved_embedding)
        if distance < 0.7:  # Set threshold for similarity
            return row[0]
    return None

# Main loop for face recognition
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    faces = detector.detect_faces(frame)
    for face in faces:
        x, y, w, h = face['box']
        name = recognize_face(frame[y:y+h, x:x+w])
        if name:
            draw_name_and_border(frame, x, y, w, h, name)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
