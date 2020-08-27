from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
# If required, create a face detection pipeline using MTCNN:

import cv2
import numpy as np
import dlib
from matplotlib.path import Path
# cap = cv2.VideoCapture(0)
# We initialise detector of dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

import mahotas

points_fidducial = []

def valid(x,y):
    if x >= 224:
        x = 223
    if y>=224:
        y = 223
    return (x,y)

def create_mask(frame):

    gray = cv2.cvtColor((frame), cv2.COLOR_BGR2GRAY)
    #
    mask = np.zeros(shape=frame.shape)
    faces = detector(gray)

    # The face landmarks code begins from here


    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        # Then we can also do cv2.rectangle function (frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        landmarks = predictor(gray, face)
        # We are then accesing the landmark points
        for n in range(0, 17):


            x = landmarks.part(n).x
            y = landmarks.part(n).y
            if x>=224:
                x = 223
            if y>=224:
                y = 223
            # cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)
            points_fidducial.append((y, x))
            #
            # points_fidducial.append((y,x))
            # if  n == 18:
            #     points_fidducial.append((landmarks.part(0).y, landmarks.part(0).x))
            # if n == 28:
            #     points_fidducial.append((landmarks.part(17).y, landmarks.part(17).x))
            # if n == 25:
            #     points_fidducial.append((landmarks.part(16).y, landmarks.part(16).x))
            # if n == 21:
            #     points_fidducial.append((landmarks.part(16).y, landmarks.part(16).x))

    # c1 = [(landmarks.part(0).y,landmarks.part(0).x),(landmarks.part(2).y,landmarks.part(1).x)]
    # c2 = [(landmarks.part(16).y, landmarks.part(0).x), (landmarks.part(7).y, landmarks.part(1).x)]
    # eye = (norm(points_fidducial[18:23] - points_fidducial[23:26]))
    # c3 = points_fidducial[0:3]
    # c3 =

    points_fidducial.append(valid(landmarks.part(26).y, landmarks.part(26).x))
    points_fidducial.append(valid(landmarks.part(25).y, landmarks.part(25).x))
    points_fidducial.append(valid(landmarks.part(24).y, landmarks.part(24).x))

    points_fidducial.append(valid(landmarks.part(19).y, landmarks.part(19).x))
    points_fidducial.append(valid(landmarks.part(18).y, landmarks.part(18).x))
    points_fidducial.append(valid(landmarks.part(17).y, landmarks.part(17).x))
    # import pdb
    # pdb.set_trace()
    # p = Path(points_fidducial)
    mahotas.polygon.fill_polygon(points_fidducial, mask)
    # import pdb
    # pdb.set_trace()
    mask = np.round((mask*255))
    # cv2.imwrite(path_mask, mask)
    return mask


    # cv2.imwrite(path,frame)
# mask
resnet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(image_size=224 , margin=0)

# img = Image.open('/vulcan/scratch/shlok/ChaLearn_liveness_challenge/OULU/Train_files_images/1_1_05_3_000069.jpg')
# # import pdb
# # pdb.set_trace()
# img_cropped =  mtcnn(img, save_path='/vulcan/scratch/shlok/ChaLearn_liveness_challenge/facenet_pytorch/temp_OULU_1.jpg')
# # img_cropped = (np.float32(img_cropped), cv2.COLOR_RGB2GRAY)
# path = '/vulcan/scratch/shlok/ChaLearn_liveness_challenge/facenet_pytorch/temp_OULU_1.jpg'
# path_mask = '/vulcan/scratch/shlok/ChaLearn_liveness_challenge/facenet_pytorch/temp_OULU_mask.jpg'
# img_cropped = cv2.imread(path)
# create_mask(img_cropped)

# Calculate embedding (unsqueeze to add batch dimension)
# img_embedding = resnet(img_cropped.unsqueeze(0))

# Or, if using for VGGFace2 classification
# resnet.classify = True
# img_probs = resnet(img_cropped.unsqueeze(0))
# print(img_probs)


