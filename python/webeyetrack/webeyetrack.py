import pathlib
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, Union
import time

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import cv2

# Face mesh detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_iris_connections_style())

    return annotated_image

@dataclass
class WebEyeTrackResults():
    detection_results: Any
    annotated_image: Optional[np.ndarray]
    fps: float

class WebEyeTrack():

    def __init__(self, model_path: Union[str, pathlib.Path]):
        
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        self.drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def process(self, frame: np.ndarray, draw_detection: bool = False, draw_informatics: bool = False):

        start = time.perf_counter()

        # Package the frame into a MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Detect the face
        detection_results = self.detector.detect(mp_image)

        # Draw the results
        if draw_detection:
            # for face_landmarks in detection_results.multi_face_landmarks:
            #     # Draw the face mesh annotations on the image.
            #     mp_drawing.draw_landmarks(
            #         image=frame,
            #         landmark_list=face_landmarks,
            #         connections=mp_face_mesh.FACEMESH_TESSELATION,
            #         landmark_drawing_spec=self.drawing_spec,
            #         connection_drawing_spec=self.drawing_spec
            #     )
            # STEP 5: Process the detection result. In this case, visualize it.
            annotated_image = draw_landmarks_on_image(frame, detection_results)
        else:
            annotated_image = None

        end = time.perf_counter()
        fps = 1 / (end - start)

        # Draw informatics
        if draw_informatics:
            if isinstance(annotated_image, np.ndarray):
                img = annotated_image
            else:
                img = frame
            cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        return WebEyeTrackResults(
            detection_results=detection_results,
            annotated_image=annotated_image,
            fps=fps
        )
