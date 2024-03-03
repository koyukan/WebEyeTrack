from .__logger import setup
from . import vis

setup()

from .webeyetrack import WebEyeTrack

__all__ = [
    'WebEyeTrack',
    'vis'
]
