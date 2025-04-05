from .__logger import setup
from .webeyetrack import WebEyeTrack, WebEyeTrackConfig

setup()

__all__ = [
    'WebEyeTrack',
    'WebEyeTrackConfig',
]
