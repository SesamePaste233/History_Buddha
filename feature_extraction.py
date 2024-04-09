import dlib
import os
import matplotlib.pyplot as plt
import PIL.Image as Image

face_image_folder = './extracted/faces'

image_paths = [os.path.join(face_image_folder, f) for f in os.listdir(face_image_folder) if os.path.isfile(os.path.join(face_image_folder, f))]

# Extract feature points from the face

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the image
for img_path in image_paths[:10]:
    img = dlib.load_rgb_image(img_path)

    # Use detector to find the faces
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))

    # Get the shape
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))

        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)

        # Draw the face landmarks on the screen.
        for i in range(68):
            x = shape.part(i).x
            y = shape.part(i).y
            print(f'Point {i}: ({x}, {y})')

        # Draw the face landmarks on the screen.
        for i in range(68):
            x = shape.part(i).x
            y = shape.part(i).y
            plt.plot(x, y, 'ro')
        
        plt.imshow(img)
        plt.show()

        print(f'Landmarks saved for {img_path}.')