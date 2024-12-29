import sys
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph.opengl as gl
import numpy as np
import cv2
import imutils
import platform

from webeyetrack.constants import GIT_ROOT
from webeyetrack.datasets.utils import draw_landmarks_on_image
from webeyetrack import vis
from webeyetrack.pipelines.flge import FLGE

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

# Pipeline
EYE_TRACKING_APPROACH = "model-based"
pipeline = FLGE(str(GIT_ROOT / 'python'/ 'weights' / 'face_landmarker_v2_with_blendshapes.task'), EYE_TRACKING_APPROACH)

WEBCAM_WIDTH = 320
WEBCAM_HEIGHT = 240
SCALE = 0.0075 # Scale factor of mm to GL units

R = np.array([
    [-1, 0, 0],
    [0, 0, 1],  # New Y-axis points where Z-axis was
    [0, 1, 0]  # New Z-axis points where Y-axis was
])

class PointCloudApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Main widget to hold the GLViewWidget and overlayed webcam
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QStackedLayout(self.central_widget)

        # Set up the GL view widget
        self.gl_widget = gl.GLViewWidget()
        self.layout.addWidget(self.gl_widget)

        # Add grid to the view
        grid = gl.GLGridItem()
        self.gl_widget.addItem(grid)

        # Generate random 3D points
        num_points = 1000
        pos = np.random.rand(num_points, 3) * 10 - 5  # Random points in a 10x10x10 cube

        # Create a GL scatter plot
        scatter_plot = gl.GLScatterPlotItem(pos=pos, color=(1, 0, 0, 1), size=5)
        self.gl_widget.addItem(scatter_plot)
        
        # Add static elements
        self.add_screen_rect() # Add first
        self.add_xyz_axes()
        self.add_camera_frustum()

        # Webcam overlay
        self.webcam_label = QtWidgets.QLabel(self)
        self.webcam_label.setGeometry(0, 0, WEBCAM_WIDTH, WEBCAM_HEIGHT)  # Fixed position
        self.webcam_label.setStyleSheet("background-color: black;")
        self.webcam_label.setParent(self.central_widget)
        self.webcam_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)

        # Webcam setup
        self.cap = cv2.VideoCapture(0)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_webcam)
        self.timer.start(30)  # Update every 30ms

        # Set up window properties
        self.setWindowTitle("3D Point Cloud Viewer with Webcam Overlay")
        self.resize(SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX)

    # def add_screen_rect(self):

    #     # Define rectangle corner points
    #     rectangle_points = np.array([
    #         [-SCREEN_WIDTH_MM / 2, 0, 0],
    #         [SCREEN_WIDTH_MM / 2, 0, 0],
    #         [SCREEN_WIDTH_MM / 2, -SCREEN_HEIGHT_MM, 0],
    #         [-SCREEN_WIDTH_MM / 2, -SCREEN_HEIGHT_MM, 0]
    #     ]) * SCALE

    #     # Define lines for the rectangle
    #     rectangle_lines = np.array([
    #         [0, 1],
    #         [1, 2],
    #         [2, 3],
    #         [3, 0]
    #     ])

    #     # Create the rectangle lines in the GLViewWidget
    #     for line in rectangle_lines:
    #         line_points = rectangle_points[line]
    #         line_points = np.dot(line_points, R)  # Rotate the points
    #         plot_line = gl.GLLinePlotItem(pos=line_points, color=(0, 0, 1, 1), width=2, antialias=True)
    #         self.gl_widget.addItem(plot_line)

    def add_screen_rect(self):

        # Define rectangle corner points
        # rectangle_points = np.array([
        #     [-SCREEN_WIDTH_MM / 2, -SCREEN_HEIGHT_MM / 2, 0],
        #     [SCREEN_WIDTH_MM / 2, -SCREEN_HEIGHT_MM / 2, 0],
        #     [SCREEN_WIDTH_MM / 2, SCREEN_HEIGHT_MM / 2, 0],
        #     [-SCREEN_WIDTH_MM / 2, SCREEN_HEIGHT_MM / 2, 0]
        # ]) * SCALE

        rectangle_points = np.array([
            [-SCREEN_WIDTH_MM / 2, 0, 0],
            [SCREEN_WIDTH_MM / 2, 0, 0],
            [SCREEN_WIDTH_MM / 2, -SCREEN_HEIGHT_MM, 0],
            [-SCREEN_WIDTH_MM / 2, -SCREEN_HEIGHT_MM, 0]
        ]) * SCALE

        rectangle_points = np.dot(rectangle_points, R)  # Rotate the points

        # Define faces (triangles) for the rectangle
        rectangle_faces = np.array([
            [0, 1, 2],  # First triangle
            [0, 2, 3]   # Second triangle
        ])

        # Create the mesh item
        rectangle_mesh = gl.GLMeshItem(
            vertexes=rectangle_points,  # Points of the rectangle
            faces=rectangle_faces,      # Faces defined by the points
            faceColors=[(0, 0, 0.3, 1)] * 4,  # Semi-transparent blue
            drawEdges=True,             # Optionally draw edges
            edgeColor=(0, 0, 1, 1)      # Edge color
        )

        # Add the rectangle to the GLViewWidget
        self.gl_widget.addItem(rectangle_mesh)


    def add_xyz_axes(self):
        # Axis length
        axis_length = 1.0

        # X-axis
        x_points = np.array([[0, 0, 0], [axis_length, 0, 0]])
        x_points = np.dot(x_points, R)  # Rotate the points
        x_line = gl.GLLinePlotItem(pos=x_points, color=(1, 0, 0, 1), width=2, antialias=True)
        self.gl_widget.addItem(x_line)

        # Y-axis
        y_points = np.array([[0, 0, 0], [0, axis_length, 0]])
        y_points = np.dot(y_points, R)  # Rotate the points
        y_line = gl.GLLinePlotItem(pos=y_points, color=(0, 1, 0, 1), width=2, antialias=True)
        self.gl_widget.addItem(y_line)

        # Z-axis
        z_points = np.array([[0, 0, 0], [0, 0, axis_length]])
        z_points = np.dot(z_points, R)  # Rotate the points
        z_line = gl.GLLinePlotItem(pos=z_points, color=(0, 0, 1, 1), width=2, antialias=True)
        self.gl_widget.addItem(z_line)

    def add_camera_frustum(self):
        # Frustum parameters
        frustrum_scale = 1.5
        origin = np.array([0, 0, 0])
        near_plane_dist = 0.5
        far_plane_dist = 0.75
        frustum_width = 1.0 / frustrum_scale
        frustum_height = 0.75 / frustrum_scale

        # Define points: camera origin and 4 points at the far plane
        points = np.array([
            origin,
            [frustum_width, frustum_height, far_plane_dist],   # Top-right
            [-frustum_width, frustum_height, far_plane_dist],  # Top-left
            [-frustum_width, -frustum_height, far_plane_dist], # Bottom-left
            [frustum_width, -frustum_height, far_plane_dist]   # Bottom-right
        ])

        # Define lines to form the frustum
        lines = np.array([
            [0, 1],  # Origin to top-right
            [0, 2],  # Origin to top-left
            [0, 3],  # Origin to bottom-left
            [0, 4],  # Origin to bottom-right
            [1, 2],  # Top edge
            [2, 3],  # Left edge
            [3, 4],  # Bottom edge
            [4, 1]   # Right edge
        ])

        # Create the frustum lines in the GLViewWidget
        for line in lines:
            line_points = points[line]
            line_points = np.dot(line_points, R)  # Rotate the points
            plot_line = gl.GLLinePlotItem(pos=line_points, color=(1, 0, 0, 1), width=2, antialias=True)
            self.gl_widget.addItem(plot_line)

    def update_webcam(self):
        ret, frame = self.cap.read()
        if ret:

            # Define intrinsics based on the frame
            width, height = frame.shape[:2]
            intrinsics = np.array([[width, 0, width // 2], [0, height, height // 2], [0, 0, 1]])

            result = pipeline.process_frame(
                frame, 
                intrinsics, 
                smooth=True,
                # screen_R=np.deg2rad(np.array([0, -180, 0]).astype(np.float32)),
                # screen_t=np.array([0.5*SCREEN_WIDTH_MM, 0, 0]).astype(np.float32),
                screen_R=np.deg2rad(np.array([0, 0, 0]).astype(np.float32)),
                # screen_t=np.array([SCREEN_WIDTH_MM/2, 0, 0]).astype(np.float32),
                screen_t=np.array([0, 0, 0]).astype(np.float32),
                screen_width_mm=SCREEN_WIDTH_MM,
                screen_height_mm=SCREEN_HEIGHT_MM,
                screen_width_px=SCREEN_WIDTH_PX,
                screen_height_px=SCREEN_HEIGHT_PX
            )

            img = frame
            if result:
                if EYE_TRACKING_APPROACH == "model-based":
                    img = vis.model_based_gaze_render(frame, result)
                elif EYE_TRACKING_APPROACH == "landmark2d":
                    img = vis.landmark_gaze_render(frame, result)
                elif EYE_TRACKING_APPROACH == 'blendshape':
                    img = vis.blendshape_gaze_render(frame, result)

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
    window = PointCloudApp()
    window.show()
    sys.exit(app.exec_())
