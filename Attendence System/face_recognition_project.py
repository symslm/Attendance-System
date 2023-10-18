# Building a Facial rcognition attendence system.
"""
i used 'pip install opencv-python' for the 
i used 'pip install cmake' for the 
i used 'pip install --use-feature=2020-resolver' for the
i used 'pip install dlib' for the 
"""
import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

# Load known faces
sayem_image = face_recognition.load_image_file(r"C:\Users\ASUS\OneDrive\Desktop\CODING\projects\Attendence System\Faces\sayem.jpg")
sayem_encoding = face_recognition.face_encodings(sayem_image)[0]

mohsin_image = face_recognition.load_image_file(r"\Users\ASUS\OneDrive\Desktop\CODING\projects\Attendence System\Faces\mohsin.jpg")
mohsin_encoding = face_recognition.face_encodings(mohsin_image)[0]

sujal_image = face_recognition.load_image_file(r"\Users\ASUS\OneDrive\Desktop\CODING\projects\Attendence System\Faces\sujal.jpg")
sujal_encoding = face_recognition.face_encodings(sujal_image)[0]

known_face_encoding = [sayem_encoding, mohsin_encoding, sujal_encoding]
known_face_names = ["sayem", "mohsin", "sujal"]

# list of expected students
student = known_face_names.copy()

face_location = []
face_encoding = []

# get the attendence date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
current_time = now.strftime("%H:%M:%S")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)
lnwriter.writerow(["Name", "Time"])

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_location = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_location)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encoding, face_encoding)
        best_match_index = np.argmin(face_distances)
    

        if (matches[best_match_index]):
            name = known_face_names[best_match_index]
            lnwriter.writerow([name, current_time])

            cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Attendence", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


video_capture.release()
cv2.destroyAllWindows()  