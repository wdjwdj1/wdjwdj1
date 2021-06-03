
import cv2
import numpy as np

from getFeatures import getFeatures
from estimateAllTranslation import estimateAllTranslation
from applyGeometricTransformation import applyGeometricTransformation

def objectTracking(rawVideo,n_frame,target,out):
    # initilize
    frames = np.empty((n_frame,),dtype=np.ndarray)
    frames_draw = np.empty((n_frame,),dtype=np.ndarray)
    bboxs = np.empty((n_frame,),dtype=np.ndarray)
    for frame_idx in range(n_frame):
        _, frames[frame_idx] = rawVideo.read()
        frames[frame_idx] = np.asarray(frames[frame_idx], dtype=np.uint8)

    # draw rectangle roi for target objects, or use default objects initilization
    n_object=1
    bboxs[0] = np.empty((n_object, 4, 2), dtype=float)
    for i in range(n_object):
        (xmin, ymin, boxw, boxh) = target
        bboxs[0][i, :, :] = np.array([[xmin, ymin], [xmin + boxw, ymin], [xmin, ymin + boxh], [xmin + boxw, ymin + boxh]]).astype(float)

    # Start from the first frame, do optical flow for every two consecutive frames.
    startXs,startYs = getFeatures(cv2.cvtColor(frames[0],cv2.COLOR_RGB2GRAY),bboxs[0],use_shi=False)
    for i in range(1,n_frame):
        newXs, newYs = estimateAllTranslation(startXs, startYs, frames[i-1], frames[i])
        Xs, Ys ,bboxs[i] = applyGeometricTransformation(startXs, startYs, newXs, newYs, bboxs[i-1])
        
        # update coordinates
        startXs = Xs
        startYs = Ys

        # update feature points as required
        n_features_left = np.sum(Xs!=-1)

        if n_features_left == 0:
            startXs, startYs = getFeatures(cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY), bboxs[i])

        if n_features_left < 15:
            startXs,startYs = getFeatures(cv2.cvtColor(frames[i],cv2.COLOR_RGB2GRAY),bboxs[i])

        # draw bounding box and visualize feature point for each object
        frames_draw[i] = frames[i].copy()
        for j in range(n_object):
            (xmin, ymin, boxw, boxh) = cv2.boundingRect(bboxs[i][j,:,:].astype(int))

            frames_draw[i] = frames_draw[i][ymin:ymin+boxh,xmin:xmin+boxw]

            frames_draw[i] = cv2.resize(frames_draw[i], (250, 250))
            frame = np.ndarray(shape=frames_draw[i].shape, dtype="float")
            frame[:] = frames_draw[i] * (1. / 255)
            out.append(frame)


