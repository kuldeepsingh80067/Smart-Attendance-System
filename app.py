import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

path = 'dataset'

images = []
classNames = []

for person in os.listdir(path):
    person_path = os.path.join(path, person)
    
    if os.path.isdir(person_path):
        for img_name in os.listdir(person_path)[:5]:
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            
            if img is not None:
                img = cv2.resize(img, (200, 200))
                images.append(img)
                classNames.append(person)

def findEncodings(images, names):
    encodeList = []
    validNames = []

    for img, name in zip(images, names):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)

        if len(encodings) > 0:
            encodeList.append(encodings[0])
            validNames.append(name)

    return encodeList, validNames

encodeListKnown, classNames = findEncodings(images, classNames)

def markAttendance(name):
    with open('Attendance.csv', 'a+') as f:
        f.seek(0)
        data = f.readlines()
        nameList = []

        for line in data:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            time = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{time}')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faces = face_recognition.face_locations(imgS)
    encodes = face_recognition.face_encodings(imgS, faces)

    for encodeFace in encodes:
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            markAttendance(name)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == 27:
        break