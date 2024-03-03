import pytest
import logging

import cv2
from webeyetrack import WebEyeTrack
from .conftest import TEST_DIR

logger = logging.getLogger("webeyetrack")

@pytest.fixture
def webeyetrack():
    return WebEyeTrack()

def test_facemesh(webeyetrack):

    # Load example image
    image = TEST_DIR / 'webgazer_p14.png'

    # Process the image
    results = webeyetrack.process(image, draw_detection=True, draw_informatics=True)
    cv2.imshow(results.draw_image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
