import numpy as np
import cv2

def preprocess_frame(frame, resize_shape=(128, 120), gray=True):
    if gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, resize_shape)
    if gray:
        frame = np.expand_dims(frame, axis=-1)
    return frame.astype(np.uint8)