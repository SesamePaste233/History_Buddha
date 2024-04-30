from deepface import DeepFace
import os
from PIL import Image
import numpy as np
import json

embeddings_folder = r'./extracted/embeddings'

ortho_face_folder = r'./extracted/faces'

face_image_paths = [os.path.join(ortho_face_folder, f) for f in os.listdir(ortho_face_folder) if os.path.isfile(os.path.join(ortho_face_folder, f))]

def extract_face_embeddings(face_image_path, embeddings_folder):
    file_name = os.path.splitext(os.path.basename(face_image_path))[0]
    face_embedding = DeepFace.represent(face_image_path, model_name='Facenet', detector_backend='retinaface')
    face_embedding_path = os.path.join(embeddings_folder, f'{file_name}.json')

    assert len(face_embedding) == 1, f'Expected 1 face embedding, but got {len(face_embedding)} for {face_image_path}.'

    with open(face_embedding_path, 'w') as f:
        json.dump(face_embedding[0]['embedding'], f)

for face_image_path in face_image_paths:
    try:
        extract_face_embeddings(face_image_path, embeddings_folder)
    except Exception as e:
        print(f'Failed to extract face embedding for {face_image_path}.')
        print(e)
        continue