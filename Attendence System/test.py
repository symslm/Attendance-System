import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

# Load known faces
sayem_image = face_recognition.load_image_file(r"projects\Faces\sayem.jpg")
sayem_encoding = face_recognition.face_encodings(sayem_image)[0]

mohsin_image = face_recognition.load_image_file(r"projects\Faces\mohsin.jpg")
mohsin_encoding = face_recognition.face_encodings(mohsin_image)[0]

sujal_image = face_recognition.load_image_file(r"projects\Faces\sujal.jpg")
sujal_encoding = face_recognition.face_encodings(sujal_image)[0]

known_face_encoding = [sayem_encoding, mohsin_encoding, sujal_encoding]
known_face_names = ["sayem", "mohsin", "sujal"]

# Get the attendance date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
current_time = now.strftime("%H:%M:%S")

# Open the CSV file for writing attendance
f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)
lnwriter.writerow(["Name", "Time"])

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encoding, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            lnwriter.writerow([name, current_time])

            # Display the recognized name on the frame
            cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
