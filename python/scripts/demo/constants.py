import platform
import numpy as np

from PyQt5 import QtWidgets, QtGui, QtCore

# Based on platform, use different approaches for determining size
# For Windows and Linux, use the screeninfo library
# For MacOS, use the Quartz library
if platform.system() == 'Windows' or platform.system() == 'Linux':
    from screeninfo import get_monitors
    m = get_monitors()[0]
    SCREEN_HEIGHT_MM = m.height_mm
    SCREEN_WIDTH_MM = m.width_mm
    SCREEN_HEIGHT_PX = m.height
    SCREEN_WIDTH_PX = m.width
elif platform.system() == 'Darwin':
    import Quartz
    main_display_id = Quartz.CGMainDisplayID()
    width_mm, height_mm = Quartz.CGDisplayScreenSize(main_display_id)
    width_px, height_px = Quartz.CGDisplayPixelsWide(main_display_id), Quartz.CGDisplayPixelsHigh(main_display_id)
    SCREEN_HEIGHT_MM = height_mm
    SCREEN_WIDTH_MM = width_mm
    SCREEN_HEIGHT_PX = height_px
    SCREEN_WIDTH_PX = width_px

WEBCAM_WIDTH = 400

def get_screen_offset():
    screen = QtWidgets.QApplication.primaryScreen()
    screen_geometry = screen.geometry()
    available_geometry = screen.availableGeometry()

    # Calculate the offset caused by the menu bar and window decorations
    x_offset = screen_geometry.x() - available_geometry.x()
    y_offset = screen_geometry.y() - available_geometry.y()

    return x_offset, y_offset