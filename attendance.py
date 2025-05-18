import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Path to the images folder
path = r"C:\Users\Dell\source\repos\image class\attendance"
images = []
classNames = []

# Load all images
myList = os.listdir(path)
print("Images Found:", myList)

for cl in myList:
    full_path = os.path.join(path, cl)
    curImg = cv2.imread(full_path)

    if curImg is None:
        print(f"⚠ Error: Could not load {full_path}")
        continue

    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print("Class Names:", classNames)

# Function to find encodings
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:  # Ensure face encoding exists
            encodeList.append(encode[0])
    return encodeList

# Function to mark attendance
# Dictionary to track last recorded time
last_attendance_time = {}

# Function to mark attendance with time gap
# Dictionary to track last recorded time
last_attendance_time = {}

# Function to mark attendance with date, day, and time gap
def markAttendance(name):
    global last_attendance_time
    now = datetime.now()
    
    # Get date, day, and time
    date = now.strftime('%Y-%m-%d')  # Format: YYYY-MM-DD
    day = now.strftime('%A')         # Full day name (Monday, Tuesday, etc.)
    time = now.strftime('%H:%M:%S')  # Format: HH:MM:SS

    # Check if name exists and if enough time has passed (5 min gap)
    if name in last_attendance_time:
        last_time = last_attendance_time[name]
        time_diff = (now - last_time).total_seconds() / 60  # Convert to minutes
        if time_diff < 1:
            return  # Skip if less than 5 minutes since last entry

    # Update last recorded time
    last_attendance_time[name] = now

    # Append to CSV file
    with open('attendance.csv', 'a') as f:  # 'a' for append mode
        f.writelines(f'\n{name},{date},{day},{time}')



# Encode known faces
encodeListKnown = findEncodings(images)
print('✅ Encoding Complete')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Resize for faster processing
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Detect faces
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()

            # Get face location & scale up
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            # Draw Green Bounding Box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw Filled Green Box for Text
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)

            # Put Text (Bold Uppercase)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Mark Attendance
            markAttendance(name)

    # Show Webcam Feed
    cv2.imshow('Webcam', img)
    cv2.waitKey(1)