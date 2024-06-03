from deepface import DeepFace
import os
from PIL import Image
import numpy as np
import json
import cv2

embeddings_folder = r'./extracted/embeddings_greyscale'

extra_embeddings_folder = r'./extraneous_buddha_statues/extra_embeddings_greyscale'

ortho_face_folder = r'./extracted/faces'

extra_face_folder = r'./extraneous_buddha_statues/extra_faces'

# mkdir
os.makedirs(embeddings_folder, exist_ok=True)
os.makedirs(extra_embeddings_folder, exist_ok=True)


face_image_paths = [os.path.join(ortho_face_folder, f) for f in os.listdir(ortho_face_folder) if os.path.isfile(os.path.join(ortho_face_folder, f))]

extra_face_image_paths = [os.path.join(extra_face_folder, f) for f in os.listdir(extra_face_folder) if os.path.isfile(os.path.join(extra_face_folder, f)) and not f.endswith('.rtf')]

def extract_face_embeddings(face_image_path, embeddings_folder, use_greyscale=False):
    file_name = os.path.splitext(os.path.basename(face_image_path))[0]

    image = cv2.imread(face_image_path)
    if image is None:
        raise ValueError(f'Failed to read image from {face_image_path}.')

    if use_greyscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    face_embedding = DeepFace.represent(image, model_name='Facenet', detector_backend='retinaface')
    face_embedding_path = os.path.join(embeddings_folder, f'{file_name}.json')

    assert len(face_embedding) == 1, f'Expected 1 face embedding, but got {len(face_embedding)} for {face_image_path}.'

    with open(face_embedding_path, 'w') as f:
        json.dump(face_embedding[0]['embedding'], f)

if __name__ == '__main__':
    from tqdm import tqdm
    for extra_face_image_path in tqdm(extra_face_image_paths):
        try:
            extract_face_embeddings(extra_face_image_path, extra_embeddings_folder, use_greyscale=True)
        except Exception as e:
            print(f'Failed to extract face embedding for {extra_face_image_path}.')
            print(e)
            continue
    '''
    for face_image_path in tqdm(face_image_paths):
        try:
            extract_face_embeddings(face_image_path, embeddings_folder)
        except Exception as e:
            print(f'Failed to extract face embedding for {face_image_path}.')
            print(e)
            continue
    '''
    
    