import os
import cv2
import numpy as np
import face_recognition

# Define paths
elon_path = r"C:\Users\Dell\source\repos\image class\ImageBasics\elon muskk.jpeg"
test_path = r"C:\Users\Dell\source\repos\image class\ImageBasics\elon bhai.webp"

# Check if files exist
if not os.path.exists(elon_path) or not os.path.exists(test_path):
    print("Error: One or both image files not found!")
    exit()

# Load images using face_recognition (returns images in RGB format)
imgElon_orig = face_recognition.load_image_file(elon_path)
imgTest_orig = face_recognition.load_image_file(test_path)

# Preprocessing Step 1: Resize images to a consistent size for processing and display
desired_width, desired_height = 800, 800
imgElon = cv2.resize(imgElon_orig, (desired_width, desired_height))
imgTest = cv2.resize(imgTest_orig, (desired_width, desired_height))

# --- Additional Preprocessing: Grayscale Conversion and Histogram Equalization ---

# Convert the resized images to grayscale
grayElon = cv2.cvtColor(imgElon, cv2.COLOR_RGB2GRAY)
grayTest = cv2.cvtColor(imgTest, cv2.COLOR_RGB2GRAY)

# Apply histogram equalization on the grayscale images to enhance contrast
equalizedElon = cv2.equalizeHist(grayElon)
equalizedTest = cv2.equalizeHist(grayTest)

# Convert the equalized images back to RGB format (three channels)
imgElon = cv2.cvtColor(equalizedElon, cv2.COLOR_GRAY2RGB)
imgTest = cv2.cvtColor(equalizedTest, cv2.COLOR_GRAY2RGB)

# Optionally, you can add a slight blur to reduce noise
# imgElon = cv2.GaussianBlur(imgElon, (5, 5), 0)
# imgTest = cv2.GaussianBlur(imgTest, (5, 5), 0)

# Run face detection on the preprocessed images using the CNN model for higher accuracy
faceLocsElon = face_recognition.face_locations(imgElon, model="cnn")
faceLocsTest = face_recognition.face_locations(imgTest, model="cnn")

# Ensure that at least one face was detected in each image
if not faceLocsElon or not faceLocsTest:
    print("Error: No face detected in one or both images!")
    exit()

# Extract face encodings from the detected face regions; take the first face found in each image
encodesElon = face_recognition.face_encodings(imgElon, known_face_locations=faceLocsElon)[0]
encodesTest = face_recognition.face_encodings(imgTest, known_face_locations=faceLocsTest)[0]

# Optional: Scale the encodings if required (as in your original script)
encodesElon = np.array(encodesElon) * 1.1
encodesTest = np.array(encodesTest) * 1.1

# Compute the Euclidean distance between the two face encodings
face_distance = np.linalg.norm(encodesElon - encodesTest)

# Use dynamic thresholding: A lower distance implies a more confident match.
threshold = 0.55 if face_distance < 0.5 else 0.65
match = face_distance < threshold

# Draw rectangles around detected faces in each image using the correct coordinate order:
# face_recognition.face_locations returns (top, right, bottom, left)
for (top, right, bottom, left) in faceLocsElon:
    cv2.rectangle(imgElon, (left, top), (right, bottom), (255, 0, 255), 2)

for (top, right, bottom, left) in faceLocsTest:
    cv2.rectangle(imgTest, (left, top), (right, bottom), (255, 0, 255), 2)

# Annotate the test image with the match result and face distance
info_text = f'Match: {match}, Distance: {round(face_distance, 2)}'
cv2.putText(imgTest, info_text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

# OpenCV displays images in BGR format, so convert the images from RGB to BGR
imgElon_BGR = cv2.cvtColor(imgElon, cv2.COLOR_RGB2BGR)
imgTest_BGR = cv2.cvtColor(imgTest, cv2.COLOR_RGB2BGR)

# Display the processed images
cv2.imshow('Elon Musk', imgElon_BGR)
cv2.imshow('Test Image', imgTest_BGR)
cv2.waitKey(0)
cv2.destroyA