import pathlib
import os

import webeyetrack
import cv2

CWD = pathlib.Path(os.path.abspath(__file__)).parent

tracker = webeyetrack.WebEyeTrack(CWD.parent / 'models' / 'face_landmarker_v2_with_blendshapes.task')

# camera stream:
cap = cv2.VideoCapture(0)  # chose camera index (try 1, 2, 3)
while cap.isOpened():
    success, image = cap.read()
    print(image)
    if not success:  # no frame input
        print("Ignoring empty camera frame.")
        continue
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # frame to RGB for the face-mesh model
    image = tracker.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # frame back to BGR for OpenCV

    cv2.imshow('output window', image)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
