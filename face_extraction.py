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
    image = Image.open(img_path)
    expand_percentage = 100
    try:
        face_objs = DeepFace.extract_faces(img_path, detector_backend='retinaface', enforce_detection=True, expand_percentage=expand_percentage)
    except Exception as e:
        print(f'Switching to mtcnn for {i + 1} image at {img_path}.')
        try:
            face_objs = DeepFace.extract_faces(img_path, detector_backend='yunet', enforce_detection=True, expand_percentage=expand_percentage)
        except Exception as e:
            print(f'Failed to extract faces from {i + 1} image at {img_path}.')
            continue
        
    # Save faces to folder
    print(f'Extracting faces from {i + 1} image at {img_path}.')
    for i, face in enumerate(face_objs):
        facial_area = face['facial_area']
        x = facial_area['x']
        y = facial_area['y']
        w = facial_area['w']
        h = facial_area['h']
        area = w * h

        half_w = w // 2
        half_h = h // 2
        centroid = (x + half_w, y + half_h)
        
        # Crop the face from the image
        scale = 2
        y_offset_scale = - 0.5
        
        threshold = 100 / scale
        if area < threshold ** 2:
            print(f'Skipping face {i + 1} from {img_path} due to small area.')
            continue
        r = max(half_w, half_h) * scale
        y_offset = max(half_w, half_h) * y_offset_scale
        face_image = image.crop((centroid[0] - r, centroid[1] - r + y_offset, centroid[0] + r, centroid[1] + r + y_offset))

        # Resize the face image
        face_image = face_image.resize((512, 512))

        # Save the face image
        face_image.save(f'./extracted/faces/{os.path.splitext(os.path.basename(img_path))[0]}_{i}.png')

        plt.imsave(f'./extracted/orthogonalized/{os.path.splitext(os.path.basename(img_path))[0]}_{i}_ortho.png', face['face'])