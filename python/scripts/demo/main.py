import sys
import cv2
import numpy as np
import imutils
import pathlib

import yaml
from webeyetrack import WebEyeTrack, WebEyeTrackConfig
from webeyetrack.data_protocols import TrackingStatus
from webeyetrack.constants import GIT_ROOT
from webeyetrack import vis

from constants import *
from calibration_widget import CalibrationWidget
from gaze_dot_canvas import GazeDotCanvas

from PyQt5 import QtWidgets, QtGui, QtCore

CWD = pathlib.Path(__file__).parent.resolve()

"""
Before running the script, ensure that you have installed the following:
1. Instead of vanilla OpenCV, use the headless version (if not, this will result in a conflicht with PyQt5):
   pip install opencv-python-headless
2. Install PyQt5:
   pip install PyQt5
"""

# Load local configuration
with open(CWD / 'config.yml', 'r') as f:
    config = yaml.safe_load(f)

class App(QtWidgets.QMainWindow):
    gaze_dot_updated = QtCore.pyqtSignal(float, float)

    def __init__(self):
        super().__init__()

        # Having state variables
        self.show_webcam = config['show_webcam']
        self.show_facial_landmarks = config['show_facial_landmarks']
        self.show_eye_patch = config['show_eye_patch']
        
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

        # Initialize the WebEyeTrack pipeline
        self.wet = WebEyeTrack(
            WebEyeTrackConfig(
                screen_px_dimensions=(SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX),
                screen_cm_dimensions=(SCREEN_WIDTH_MM/10, SCREEN_HEIGHT_MM/10),
                verbose=config['verbose']
            )
        )

        # Compute the hypothetical height based on the frame's window and height to match the WEBCAM_WIDTH
        webcam_height = int(height * WEBCAM_WIDTH / width)
        
        # Webcam overlay
        self.webcam_label = QtWidgets.QLabel(self)
        self.webcam_label.setGeometry(0, 0, WEBCAM_WIDTH, webcam_height)  # Fixed position
        self.webcam_label.setStyleSheet("background-color: black;")
        self.webcam_label.setParent(self.central_widget)
        self.webcam_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)

        # Eye patch overlay
        self.eye_patch_label = QtWidgets.QLabel(self)
        self.eye_patch_label.setGeometry(WEBCAM_WIDTH, 0, 512, 128)  # Fixed position
        self.eye_patch_label.setStyleSheet("background-color: black;")
        self.eye_patch_label.setParent(self.central_widget)
        self.eye_patch_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)

    def toggle_canvas(self):
        print("Toggling canvas visibility")
        # if self.canvas_2d.isVisible():
        #     self.canvas_2d.hide()
        #     self.gl_widget.show()
        #     self.ui_container.show()
        #     self.webcam_label.show()
        # else:
        #     self.gl_widget.hide()
        #     self.canvas_2d.show()
        #     self.canvas_2d.complete = False
        #     self.ui_container.hide()
        #     self.webcam_label.hide()

    def add_ui_controls(self):
        # Create a fixed-position rectangle for the UI controls
        self.ui_container = QtWidgets.QWidget(self)
        self.ui_container.setGeometry(SCREEN_WIDTH_PX - 200, 10, 190, 5*50)  # Top-right corner
        self.ui_container.setStyleSheet("background-color: rgba(0, 0, 0, 0.7); border-radius: 10px; border: 1px solid white;")

        # Create a layout for buttons inside the container
        layout = QtWidgets.QVBoxLayout(self.ui_container)
        layout.setContentsMargins(10, 10, 10, 10)

        # Add a label
        label = QtWidgets.QLabel("Controls", self)
        label.setStyleSheet("color: white; border: 0px; font-size: 20px; padding: 0px; margin: 5px;")
        layout.addWidget(label)

        # Add a checkbox for showing/hiding the webcam
        self.webcam_checkbox = QtWidgets.QCheckBox("Show Webcam", self)
        self.webcam_checkbox.setChecked(self.show_webcam)
        self.webcam_checkbox.setStyleSheet("color: white; border: 0px; font-size: 16px; padding: 5px; margin: 5px;")
        self.webcam_checkbox.stateChanged.connect(self.toggle_webcam)
        layout.addWidget(self.webcam_checkbox)

        # Add a checkbox for showing the facial landmarks
        self.landmarks_checkbox = QtWidgets.QCheckBox("Show Landmarks", self)
        self.landmarks_checkbox.setChecked(self.show_facial_landmarks)
        self.landmarks_checkbox.setStyleSheet("color: white; border: 0px; font-size: 16px; padding: 5px; margin: 5px;")
        self.landmarks_checkbox.stateChanged.connect(lambda state: setattr(self, 'show_facial_landmarks', state == QtCore.Qt.Checked))
        layout.addWidget(self.landmarks_checkbox)

        # Add a checkbox for showing the eye patch
        self.eye_patch_checkbox = QtWidgets.QCheckBox("Show Eye Patch", self)
        self.eye_patch_checkbox.setChecked(self.show_eye_patch)
        self.eye_patch_checkbox.setStyleSheet("color: white; border: 0px; font-size: 16px; padding: 5px; margin: 5px;")
        self.eye_patch_checkbox.stateChanged.connect(self.toggle_eye_patch)
        layout.addWidget(self.eye_patch_checkbox)

        # Add the calibrate button
        calibrate_button = QtWidgets.QPushButton("Calibrate", self)
        calibrate_button.clicked.connect(self.toggle_canvas)
        calibrate_button.setStyleSheet("color: white; border-radius: 10px; border: 0px; margin: 5px")
        layout.addWidget(calibrate_button)

    def toggle_webcam(self):
        self.show_webcam = not self.show_webcam
        if self.show_webcam:
            self.webcam_label.show()
        else:
            self.webcam_label.hide()

        # Update the webcam label visibility
        self.webcam_checkbox.setChecked(self.show_webcam)

    def toggle_eye_patch(self):
        self.show_eye_patch = not self.show_eye_patch
        if self.show_eye_patch:
            self.eye_patch_label.show()
        else:
            self.eye_patch_label.hide()

        # Update the eye patch label visibility
        self.eye_patch_checkbox.setChecked(self.show_eye_patch)

    def update_webcam(self):
        ret, frame = self.cap.read()
        if ret:

            # Process the frame with WebEyeTrack
            status, gaze_result, detection = self.wet.process_frame(frame)

            if self.show_facial_landmarks and detection:
                frame = vis.draw_landmarks_on_image(frame, detection)

            if status == TrackingStatus.SUCCESS:
                # Show the eye patch if requested
                if self.show_eye_patch:
                    eye_patch = gaze_result.eye_patch
                    if eye_patch is not None:
                        eye_patch = cv2.cvtColor(eye_patch, cv2.COLOR_BGR2RGB)
                        h, w, ch = eye_patch.shape
                        bytes_per_line = ch * w
                        qeye_patch = QtGui.QImage(eye_patch.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                        self.eye_patch_label.setPixmap(QtGui.QPixmap.fromImage(qeye_patch))

            # Show the frame
            if self.show_webcam:
                frame = imutils.resize(frame, width=WEBCAM_WIDTH)  # Resize to fit QLabel
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qframe = QtGui.QImage(frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                self.webcam_label.setPixmap(QtGui.QPixmap.fromImage(qframe))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
