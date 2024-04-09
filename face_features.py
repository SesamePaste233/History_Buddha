from deepface import DeepFace
import os
import matplotlib.pyplot as plt
import PIL.Image as Image

import check_files as Loader

dataset = Loader.LoadAllFiles()

images_with_head = dataset.file_paths_if(lambda d: 'head' in d.parts)

# Plot images
def plot_faces(face_objs, original_img_path):
    num_faces = len(face_objs)
    fig, ax = plt.subplots(1, num_faces + 1)
    ax[0].imshow(Image.open(original_img_path))
    for i in range(1, num_faces + 1):
        ax[i].imshow(face_objs[i-1]['face'])
    plt.show()

for i, img_path in enumerate(images_with_head):
    try:
        face_objs = DeepFace.extract_faces(img_path, detector_backend='retinaface', enforce_detection=True, expand_percentage=0.2)
    except Exception as e:
        print(f'Switching to mtcnn for {i + 1} image at {img_path}.')
        try:
            face_objs = DeepFace.extract_faces(img_path, detector_backend='mtcnn', enforce_detection=True, expand_percentage=0.2)
        except Exception as e:
            print(f'Failed to extract faces from {i + 1} image at {img_path}.')
            continue
        
    # Save faces to folder
    print(f'Extracting faces from {i + 1} image at {img_path}.')
    for i, face in enumerate(face_objs):
        face_img = face['face']
        plt.imsave(f'./extracted/faces/{os.path.splitext(os.path.basename(img_path))[0]}_{i}.png', face_img)
        #img.save(f'./extracted/faces/{os.path.splitext(os.path.basename(img_path))}_{i}.png')