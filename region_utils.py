# region_utils.py
import insightface
import numpy as np
import cv2

# Load InsightFace model
model = insightface.app.FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=0, det_size=(640, 640))

def get_face_regions(image):
    faces = model.get(image)
    if not faces:
        return None

    face = faces[0]  # Assume one face
    landmarks = face.kps.astype(int)

    # Regions based on landmarks
    regions = {}

    # Forehead: midpoint between eyes and up
    mid_eyes = ((landmarks[0] + landmarks[1]) // 2).astype(int)
    forehead = image[mid_eyes[1]-60:mid_eyes[1]-20, mid_eyes[0]-40:mid_eyes[0]+40]
    regions['forehead'] = forehead

    # Left cheek
    left_cheek = image[landmarks[3][1]-20:landmarks[3][1]+40, landmarks[3][0]-40:landmarks[3][0]+20]
    regions['left_cheek'] = left_cheek

    # Right cheek
    right_cheek = image[landmarks[4][1]-20:landmarks[4][1]+40, landmarks[4][0]-20:landmarks[4][0]+40]
    regions['right_cheek'] = right_cheek

    # Under eyes
    under_left = image[landmarks[0][1]+10:landmarks[0][1]+30, landmarks[0][0]-20:landmarks[0][0]+20]
    under_right = image[landmarks[1][1]+10:landmarks[1][1]+30, landmarks[1][0]-20:landmarks[1][0]+20]
    regions['under_eyes'] = np.hstack((under_left, under_right))

    return regions
