import webeyetrack
import cv2

tracker = webeyetrack.WebEyeTrack(
    model_path="/media/eduardo/Crucial X6/reading-analytics-group/WebEyeTrack/python/models/face_landmarker_v2_with_blendshapes.task"
)

# camera stream:
cap = cv2.VideoCapture(0)  # chose camera index (try 1, 2, 3)
while cap.isOpened():
    success, image = cap.read()
    if not success:  # no frame input
        print("Ignoring empty camera frame.")
        continue
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # frame to RGB for the face-mesh model
    results = tracker.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # frame back to BGR for OpenCV

    cv2.imshow('output window', image)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
