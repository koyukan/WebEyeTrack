import pytest
import logging

import cv2
from webeyetrack import WebEyeTrack
from .conftest import TEST_DIR, MODELS_DIR 

logger = logging.getLogger("webeyetrack")

@pytest.fixture
def webeyetrack():
    model_path = MODELS_DIR / 'face_landmarker_v2_with_blendshapes.task'
    return WebEyeTrack(model_path)

def test_facemesh(webeyetrack):

    # Load example image
    image_fp = TEST_DIR / 'webgazer_p14.png'
    image = cv2.imread(str(image_fp))

    # Process the image
    results = webeyetrack.process(image, draw_detection=True, draw_informatics=True)
    cv2.imshow('output', results.annotated_image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
