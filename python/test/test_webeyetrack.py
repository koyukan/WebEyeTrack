import pytest
import logging

import cv2
import webeyetrack as we
from .conftest import TEST_DIR, MODELS_DIR 

logger = logging.getLogger("webeyetrack")

@pytest.fixture
def webeyetrack():
    model_path = MODELS_DIR / 'face_landmarker_v2_with_blendshapes.task'
    return we.WebEyeTrack(model_path)

def test_facemesh(webeyetrack):

    # Load example image
    image_fp = TEST_DIR / 'webgazer_p14.png'
    image = cv2.imread(str(image_fp))

    # Process the image
    results = webeyetrack.process(image)

    # Draw the annotations
    annotated_image = we.vis.draw_landmarks_on_image(image, results.detection_results)
    annotated_image = we.vis.draw_fps(annotated_image, results.fps)

    cv2.imshow('output', annotated_image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
