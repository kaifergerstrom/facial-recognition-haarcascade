import cv2
import face_recognition
import os
import pickle

def save_pickle(var, file_path):
	f = open(file_path, 'wb')
	pickle.dump(var, f)
	f.close()

def load_pickle(file_path):
	f = open(file_path, 'rb')
	obj = pickle.load(f)
	f.close()
	return obj

def load_training_faces():

	print("Loading training faces...")

	known_faces = {}

	for subdir, dirs, files in os.walk(KNOWN_FACES_DIR):

		label = os.path.basename(subdir)  # Get name of folder
		known_faces[label] = []

		for file in files:  # Loop through each file in the folder
			
			image_dir = os.path.join(subdir, file)  # Create full path for each file

			# Load and create encoding for each image
			image = face_recognition.load_image_file(image_dir)
			encoding = face_recognition.face_encodings(image)[0]
			
			# Create reference dictionary for known faces
			known_faces[label].append(encoding)

	return known_faces


PICKLE_FILE = "pickles/known_faces.pckl"
KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.6


if not os.path.exists(PICKLE_FILE):
	known_faces = load_training_faces()
	save_pickle(known_faces, PICKLE_FILE)
else:
	known_faces = load_pickle(PICKLE_FILE)

cam = cv2.VideoCapture(0)

while True:
	
	_, img = cam.read()

	cv2.imshow('my webcam', img)

	if cv2.waitKey(1) == 27: 
		break  # esc to quit

cv2.destroyAllWindows()

