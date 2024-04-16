import deepface
import os
import cv2
import json

embeddings_folder = r'./extracted/embeddings'

ortho_face_folder = r'./extracted/orthogonalized'

face_image_paths = [os.path.join(ortho_face_folder, f) for f in os.listdir(ortho_face_folder) if os.path.isfile(os.path.join(ortho_face_folder, f))]

def extract_face_embeddings(face_image_path, embeddings_folder):
    file_name = os.path.splitext(os.path.basename(face_image_path))[0]
    face_image = cv2.imread(face_image_path)
    face_embedding = deepface.DeepFace.represent(face_image, model_name='Facenet')
    face_embedding_path = os.path.join(embeddings_folder, f'{file_name}.json')
    with open(face_embedding_path, 'w') as f:
        json.dump(face_embedding.tolist(), f)