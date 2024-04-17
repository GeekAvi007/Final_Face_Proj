import cv2
import numpy as np

net = cv2.dnn.readNet("yolov6.weights", "yolov6.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def recognize_faces(face_embedding, known_embeddings, known_labels):
    threshold = 0.6
    for i, known_embedding in enumerate(known_embeddings):
        distance = np.linalg.norm(face_embedding - known_embedding)
        if distance < threshold:
            return known_labels[i]
    return "Unknown"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                face_roi = frame[y:y+h, x:x+w]

                face_embedding = extract_face_embedding(face_roi)

                recognized_label = recognize_faces(face_embedding, known_embeddings, known_labels)

                color = (0, 255, 0) if recognized_label != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, recognized_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
