from PIL import Image as Img
import numpy as np
from keras_facenet import FaceNet
import pickle
import cv2
import argparse

# Function to recognize faces in an image
def recognize_faces(image_path, database_path, output_path):
    

    # Load the database of known face embeddings
    with open(database_path, "rb") as myfile:
        database = pickle.load(myfile)

    # Load and process the input image
    gbr1 = cv2.imread(image_path)
    wajah = HaarCascade.detectMultiScale(gbr1, 1.1, 4)

    detected_faces = []
    data=""
    for face_coordinates in wajah:
        x1, y1, width, height = face_coordinates
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        # Draw a rectangle around the detected face
        cv2.rectangle(gbr1, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle

        gbr = cv2.cvtColor(gbr1, cv2.COLOR_BGR2RGB)
        gbr = Img.fromarray(gbr)
        gbr_array = np.array(gbr)

        face = gbr_array[y1:y2, x1:x2]

        face = Img.fromarray(face)
        face = face.resize((160, 160))
        face = np.array(face)

        face = np.expand_dims(face, axis=0)
        query_embedding = MyFaceNet.embeddings(face)

        # Compare the query face embedding with known face embeddings in the database
        def find_face(image_embedding, database):
            min_dist = float("inf")
            identity = None
            for name, stored_embedding in database.items():
                dist = np.linalg.norm(image_embedding - stored_embedding)
                if dist < min_dist:
                    min_dist = dist
                    identity = name
            return identity, min_dist

        # Find the most similar face in the database
        identity, min_dist = find_face(query_embedding, database)

        # Set a threshold for face recognition
        threshold = 0.7  # You can adjust this threshold as needed

        if min_dist <= threshold:
            detected_faces.append((identity, min_dist, (x1, y1, x2, y2)))
        else:
            detected_faces.append(("Unknown", min_dist, (x1, y1, x2, y2)))

    
    # Write the identity on the rectangle and save the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    for identity, min_dist, (x1, y1, x2, y2) in detected_faces:
        cv2.putText(gbr1, identity, (x1, y1 - 10), font, 0.9, (255, 255, 255), 2)
        data+=identity+'\r'
        # print(f"Face recognized as {identity} with distance {min_dist:.2f}.")

    # Specify the file path where you want to save the data
    file_path = "C:/Users/saite/Desktop/major_project/backend/output_names.txt"

    # Open the file in write mode
    with open(file_path, "w") as file:
        # Write the data to the file
        file.write(data)
    # Save the image with rectangles and identities
    cv2.imwrite(output_path, gbr1)

    if not detected_faces:
        print("No faces detected in the image.")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Recognize faces in an image")
    parser.add_argument("image_path", help="Path to the input image")
    # parser.add_argument("model_path", help="Path to the FaceNet model")
    parser.add_argument("database_path", help="Path to the database of known face embeddings")
    parser.add_argument("output_path", help="Path to save the output image with rectangles and identities")
    args = parser.parse_args()

    # Load HaarCascade classifier
    HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
    # Load the FaceNet model

    MyFaceNet = FaceNet()

    # Call the recognize_faces function with the provided arguments
    recognize_faces(args.image_path, args.database_path, args.output_path)
