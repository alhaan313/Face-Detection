import cv2
import os
import numpy as np

# Function to load images and labels for training
def load_training_data(data_folder):
    faces = []
    labels = []
    label_dict = {}
    current_label = 0

    for person_name in os.listdir(data_folder):
        person_folder = os.path.join(data_folder, person_name)
        if not os.path.isdir(person_folder):
            continue

        label_dict[current_label] = person_name
        current_label += 1

        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            faces.append(image)
            labels.append(current_label - 1)

    return faces, labels, label_dict

# Load training data from your dataset folder
dataset_folder =  r'C:\Users\Alhaan\Desktop\Coding\Face Detection Project\Dataset'

faces, labels, label_dict = load_training_data(dataset_folder)


# Create LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the recognizer using the training data
face_recognizer.train(faces, np.array(labels))

# Initialize the camera for live video stream
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video feed
    ret, frame = cap.read()

    # Convert the frame to grayscale for face recognition
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face_roi = gray[y:y+h, x:x+w]

        # Recognize the face using the trained model
        label, confidence = face_recognizer.predict(face_roi)

        # Get the name of the recognized person from the label_dict
        recognized_person = label_dict[label]

        # Display the name and confidence on the frame
        text = f'{recognized_person} (Confidence: {confidence:.2f})'
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame with bounding boxes and recognized names
    cv2.imshow('Face Recognition', frame)

    # Exit loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()