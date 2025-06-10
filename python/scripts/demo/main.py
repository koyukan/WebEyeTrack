import sys
import cv2
import numpy as np
import imutils
import platform

from webeyetrack import WebEyeTrack
from webeyetrack.constants import GIT_ROOT
from webeyetrack.vis import draw_landmarks_on_image
from webeyetrack import vis

from constants import *
from calibration_widget import CalibrationWidget
from gaze_dot_canvas import GazeDotCanvas

from PyQt5 import QtWidgets, QtGui, QtCore

"""
Before running the script, ensure that you have installed the following:
1. Instead of vanilla OpenCV, use the headless version (if not, this will result in a conflicht with PyQt5):
   pip install opencv-python-headless
2. Install PyQt5:
   pip install PyQt5
"""

class App(QtWidgets.QMainWindow):
    gaze_dot_updated = QtCore.pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        
        # Set up window properties
        self.setWindowTitle("WebEyeTrack Demo")
        self.resize(SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX)

        # Main widget to hold the GLViewWidget and overlayed webcam
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QStackedLayout(self.central_widget)

        # Add 2D canvas
        self.canvas_2d = CalibrationWidget()
        self.layout.addWidget(self.canvas_2d)
        self.canvas_2d.hide()

        # Add gaze dot canvas
        self.gaze_dot_canvas = GazeDotCanvas()
        self.gaze_dot_canvas.setParent(self)
        self.gaze_dot_canvas.resize(self.size())
        self.gaze_dot_canvas.show()
        self.gaze_dot_updated.connect(self.gaze_dot_canvas.update_dot)

        # UI controls
        self.add_ui_controls()

        # Webcam setup
        self.cap = cv2.VideoCapture(0)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_webcam)
        self.timer.start(30)  # Update every 30ms

        # Determine the size of the webcam frame
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        intrinsics = np.array([[width, 0, width // 2], [0, height, height // 2], [0, 0, 1]])

        # # Initialize the WebEyeTrack pipeline
        # # Pipeline
        # self.pipeline = WebEyeTrack(
        #     model_asset_path=str(GIT_ROOT / 'python'/ 'weights' / 'face_landmarker_v2_with_blendshapes.task'), 
        #     frame_height=height,
        #     frame_width=width,
        #     intrinsics=intrinsics,
        #     screen_R=np.deg2rad(np.array([0, 0, 0]).astype(np.float32)),
        #     screen_t=np.array([-SCREEN_WIDTH_MM/2, 0, 0]).astype(np.float32),
        #     screen_width_mm=SCREEN_WIDTH_MM,
        #     screen_height_mm=SCREEN_HEIGHT_MM,
        #     screen_width_px=SCREEN_WIDTH_PX,
        #     screen_height_px=SCREEN_HEIGHT_PX
        # )

        # Compute the hypothetical height based on the frame's window and height to match the WEBCAM_WIDTH
        webcam_height = int(height * WEBCAM_WIDTH / width)
        
        # Webcam overlay
        self.webcam_label = QtWidgets.QLabel(self)
        self.webcam_label.setGeometry(0, 0, WEBCAM_WIDTH, webcam_height)  # Fixed position
        self.webcam_label.setStyleSheet("background-color: black;")
        self.webcam_label.setParent(self.central_widget)
        self.webcam_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)

    def toggle_canvas(self):
        if self.canvas_2d.isVisible():
            self.canvas_2d.hide()
            self.gl_widget.show()
            self.ui_container.show()
            self.webcam_label.show()
        else:
            self.gl_widget.hide()
            self.canvas_2d.show()
            self.canvas_2d.complete = False
            self.ui_container.hide()
            self.webcam_label.hide()

    def add_ui_controls(self):
        # Create a fixed-position rectangle for the UI controls
        self.ui_container = QtWidgets.QWidget(self)
        self.ui_container.setGeometry(self.central_widget.width() * 2, 10, 190, 100)  # Top-right corner
        self.ui_container.setStyleSheet("background-color: rgba(0, 0, 0, 0.7); border-radius: 10px; border: 1px solid white;")

        # Create a layout for buttons inside the container
        layout = QtWidgets.QVBoxLayout(self.ui_container)
        layout.setContentsMargins(10, 10, 10, 10)

        # Add a label
        label = QtWidgets.QLabel("Controls", self)
        label.setStyleSheet("color: white; border: 0px; font-size: 20px; padding: 0px; margin: 0px;")
        layout.addWidget(label)

        # Add the calibrate button
        calibrate_button = QtWidgets.QPushButton("Calibrate", self)
        calibrate_button.clicked.connect(self.toggle_canvas)
        calibrate_button.setStyleSheet("background-color: rgba(50, 50, 50, 1.0); color: white; border-radius: 10px; border: 0px")
        layout.addWidget(calibrate_button)

    def update_webcam(self):
        ret, frame = self.cap.read()
        if ret:

            # Define intrinsics based on the frame
            width, height = frame.shape[:2]
            intrinsics = np.array([[width, 0, width // 2], [0, height, height // 2], [0, 0, 1]])
            img = frame

            if type(img) != np.ndarray:
                img = frame

            img = imutils.resize(img, width=WEBCAM_WIDTH)  # Resize to fit QLabel
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img.shape
            bytes_per_line = ch * w
            qimg = QtGui.QImage(img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            self.webcam_label.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
