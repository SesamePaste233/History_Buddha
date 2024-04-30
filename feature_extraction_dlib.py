import dlib
import os
import matplotlib.pyplot as plt
import PIL.Image as Image
import cv2
import numpy as np
import json

from check_files import LoadAllFiles, Dataset

face_image_folder = './extracted/orthogonalized'

image_paths = [os.path.join(face_image_folder, f) for f in os.listdir(face_image_folder) if os.path.isfile(os.path.join(face_image_folder, f))]

orthogonalized_folder = './extracted/orthogonalized'

orthogonalized_landmarks_folder = './extracted/ortho_landmarks'

if not os.path.exists(orthogonalized_folder):
    os.makedirs(orthogonalized_folder)

if not os.path.exists(orthogonalized_landmarks_folder):
    os.makedirs(orthogonalized_landmarks_folder)

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor_68 = dlib.shape_predictor("shape_predictor_68_face_landmarks_GTX.dat")

predictor_5 = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")

def single_detection(face_dlib_img, landmark_predictor=predictor_68):
    # Use detector to find the faces
    dets = detector(face_dlib_img, 1)
    shapes = []
    for d in dets:
        shapes.append(landmark_predictor(img, d))
    return shapes

def shape2landmarks(shape):
    return np.array([(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)])

def show_landmarks(img, shape):
    plt.imshow(img)
    for i in range(shape.num_parts):
        x = shape.part(i).x
        y = shape.part(i).y
        plt.plot(x, y, 'ro')
        plt.text(x, y, str(i), color='black', fontsize=12)
    plt.show()

def show_landmarks_np(img, landmarks):
    plt.imshow(img)
    for i, (x, y) in enumerate(landmarks):
        plt.plot(x, y, 'ro')
        plt.text(x, y, str(i), color='black', fontsize=12)
    plt.show()

def orthogonalize(face_img, dlib_68_landmarks_detection):
    '''
    This function takes an image and the dlib 68 landmarks detection object as input
    and returns the orthogonalized image and the transformation matrix

    Faces may be potentially stretched and the area is not preserved.

    '''
    size = face_img.shape
    
    dlib_68_landmarks_indices = [36, 45, 48, 54] # Left eye, right eye, nose, left mouth corner, right mouth corner

    landmarks = np.array([(dlib_68_landmarks_detection.part(i).x, dlib_68_landmarks_detection.part(i).y) for i in dlib_68_landmarks_indices])

    # Define the target landmarks for the orthogonal projection
    # Here, you need to define the target landmarks manually or by some other means
    # For simplicity, I'll use some example target landmarks
    orthogonal_landmarks = np.array([
        (0.3 * size[1], 0.3 * size[0]), # Left eye
        (0.7 * size[1], 0.3 * size[0]), # Right eye
        #(0.5 * size[1], 0.5 * size[0]), # Nose
        (0.4 * size[1], 0.7 * size[0]), # Left mouth corner
        (0.6 * size[1], 0.7 * size[0])  # Right mouth corner
    ])

    # Estimate the transformation matrix using OpenCV's estimateAffinePartial2D
    transformation_matrix, _ = cv2.estimateAffinePartial2D(landmarks, orthogonal_landmarks)

    # Apply the transformation to the original image
    warped_image = cv2.warpAffine(face_img, transformation_matrix, (face_img.shape[1], face_img.shape[0]))

    # Warp the landmarks to the target landmarks
    full_landmarks = np.array([(dlib_68_landmarks_detection.part(i).x, dlib_68_landmarks_detection.part(i).y) for i in range(68)])
    warped_landmarks = cv2.transform(full_landmarks.reshape(1, -1, 2), transformation_matrix).reshape(-1, 2)

    return warped_image, warped_landmarks, transformation_matrix

def iterate(img, transform = np.eye(3)):
    shapes = single_detection(img, predictor_68)
    if len(shapes) == 0:
        print(f'No face detected. Aborting.')
        return None, None, None

    shape = shapes[0]
    #show_landmarks(img, shape)

    warped_img, warped_landmarks, warp_transform = orthogonalize(img, shape)

    # Combine the transformations
    transform = np.dot(warp_transform, transform)

    return warped_img, warped_landmarks, transform

# Load the image
for img_path in image_paths:
    file_name: str = os.path.splitext(os.path.basename(img_path))[0]

    # Remove suffix '_ortho' if it exists
    if file_name.endswith('_ortho'):
        file_name = file_name[:-6]

    img = dlib.load_rgb_image(img_path)

    iterations = 1

    # Iterate through the image
    transform = np.eye(3)

    warped_img = img

    shapes = single_detection(img, predictor_68)
    if len(shapes) == 0:
        print(f'No face detected for {img_path}.')
        continue

    # Get face chip
    shape = shapes[0]
    #face_chip = dlib.get_face_chip(img, shape, size=256, padding = 0.5)

    #warped_img = face_chip

    warped_landmarks = shape2landmarks(shape)

    #show_landmarks_np(warped_img, warped_landmarks)

    
    '''
    for i in range(iterations):
        warped_img, warped_landmarks, transform = iterate(img, transform)

        if warped_img is None:
            print(f'No face detected for {img_path}.')
            break

        # change the 2x3 transformation matrix to 3x3
        transform = np.vstack([transform, [0, 0, 1]])

        img = warped_img
    '''

    if warped_img is not None:
        #show_landmarks_np(warped_img, warped_landmarks)

        # Save orthogonalized image
        #cv2.imwrite(os.path.join(orthogonalized_folder, f'{file_name}_ortho.png'), warped_img)
        # Save wit#h plt is also possible
        #plt.imsave(os.path.join(orthogonalized_folder, f'{file_name}_ortho.png'), warped_img)

        # Save orthogonalized landmarks
        #np.save(os.path.join(orthogonalized_landmarks_folder, f'{file_name}_orthomarks.npy'), warped_landmarks)
        # Json is also possible
        with open(os.path.join(orthogonalized_landmarks_folder, f'{file_name}_orthomarks.json'), 'w') as f:
            json.dump(warped_landmarks.tolist(), f)

    