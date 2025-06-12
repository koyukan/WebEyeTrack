import sys
import cv2
import numpy as np
import imutils
import pathlib
from collections import defaultdict

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
        self.calibration_canvas = CalibrationWidget()
        self.layout.addWidget(self.calibration_canvas)
        self.calibration_canvas.hide()
        self.calibration_canvas.calibration_status.connect(self.calib_status_update)
        self.calibration_canvas.calib_dot_updated.connect(self.calib_point_updated)

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

        # On the bottom right, show the real-time xy coordinates of the gaze dot
        self.gaze_coordinates_label = QtWidgets.QLabel(self)
        self.gaze_coordinates_label.setGeometry(SCREEN_WIDTH_PX - 180, SCREEN_HEIGHT_PX - 100, 190, 40)
        self.gaze_coordinates_label.setStyleSheet("background-color: black; color: white; font-size: 16px; padding: 5px;")
        self.gaze_coordinates_label.setParent(self.central_widget)
        self.gaze_dot_updated.connect(lambda x, y: self.gaze_coordinates_label.setText(f"Gaze: ({x:.2f}, {y:.2f})"))

        # Store the calibration points
        self.calibrating = False
        self.current_calib_point = None
        self.calib_pts = defaultdict(list)

    def init_calib(self):
        self.calibrating = False
        self.calib_pts.clear()
        self.calibration_canvas.show()
        self.calibration_canvas.complete = False
        self.calibration_canvas.start_calibration()
        self.ui_container.hide()
        if self.show_webcam:
            self.webcam_label.hide()
        if self.show_eye_patch:
            self.eye_patch_label.hide()

    def calib_status_update(self, status):
        self.calibrating = status

    def end_calib(self):
        self.calibrating = False
        self.calibration_canvas.hide()
        self.ui_container.show()
        if self.show_webcam:
            self.webcam_label.show()
        if self.show_eye_patch:
            self.eye_patch_label.show()

        # Obtain the calibration points
        # print(self.calib_pts)

        # Extract the frames and normalize the points
        # frames = []
        calib_gaze_results = []
        calib_norm_pogs = []
        pred_norm_pogs = []
        for point, gaze_results in self.calib_pts.items():
            # Before adding the frames, we get many points for each
            # calibration point, so we should try to find the best
            # one. We want to find the center of the points (remove outliers)
            # and pick the centermost point.
            if gaze_results:
                pogs = [gaze_result.norm_pog for gaze_result in gaze_results]
                pogs = np.array(pogs)
                
                # Compute the mean and standard deviation
                mean_pog = np.mean(pogs, axis=0)
                std_pog = np.std(pogs, axis=0)

                print(f"Calibration point {point}: mean={mean_pog}, std={std_pog}")
                
                # Get the point closest to the mean
                distances = np.linalg.norm(pogs - mean_pog, axis=1)
                closest_index = np.argmin(distances)
                best_pog = pogs[closest_index]
                calib_gaze_results.append(gaze_results[closest_index])
                calib_norm_pogs.append(point)
                pred_norm_pogs.append(best_pog)


        # If we have enough points, we can proceed with the calibration
        if len(calib_gaze_results) < 3:
            print("Not enough calibration points collected. Please try again.")
            return
        
        # print(self.calib_pts)
        print(calib_norm_pogs)
        print(pred_norm_pogs)

        returned_calib = self.wet.adapt_from_gaze_results(
            calib_gaze_results,
            np.stack(calib_norm_pogs),
            affine_transform=True,
            steps_inner=10,
            inner_lr=1e-4,
            adaptive_lr=True
        )
        print(f"Calibration completed successfully: {returned_calib}.")

        screen_height_px, screen_width_px = SCREEN_HEIGHT_PX//2, SCREEN_WIDTH_PX//2
        screen_img = np.zeros((screen_height_px, screen_width_px, 3), dtype=np.uint8)
        for i, pog in enumerate(calib_norm_pogs):

            # gt_pt = row['mouseClickX'], row['mouseClickY']
            gt_tb_pt = pog
            pred_pt = returned_calib[i]

            # Draw the points as circles
            x, y = (gt_tb_pt[0] + 0.5), (gt_tb_pt[1] + 0.5)
            cv2.circle(screen_img, (int(x * screen_width_px), int(y * screen_height_px)), 10, (0, 0, 255), -1)
            x2, y2 = (pred_pt[0] + 0.5), (pred_pt[1] + 0.5)
            cv2.circle(screen_img, (int(x2 * screen_width_px), int(y2 * screen_height_px)), 10, (255, 255, 0), -1)

            # Draw a line between the original calibration point and the resulting calibration point
            cv2.line(screen_img, (int(x * screen_width_px), int(y * screen_height_px)),
                        (int(x2 * screen_width_px), int(y2 * screen_height_px)), (255, 0, 255), 1)

        # Save the image
        calib_img_path = CWD / 'calibration_result.png'
        cv2.imwrite(str(calib_img_path), screen_img)

    def calib_point_updated(self, x, y):
        print(f"Calibration point updated: ({x}, {y})")
        self.current_calib_point = (x, y)

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
        calibrate_button.clicked.connect(self.init_calib)
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

                # Update the gaze dot position
                gaze_x, gaze_y = gaze_result.norm_pog
                self.gaze_dot_updated.emit(gaze_x, gaze_y)

                # If calibrating, store the calibration point
                if self.calibrating and self.current_calib_point is not None:
                    self.calib_pts[self.current_calib_point].append(gaze_result)

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
