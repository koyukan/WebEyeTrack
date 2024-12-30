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
SCALE = 0.005 # Scale factor of mm to GL units

R = np.array([
    [-1, 0, 0],
    [0, 0, 1],  # New Y-axis points where Z-axis was
    [0, 1, 0]  # New Z-axis points where Y-axis was
])

def get_screen_offset():
    screen = QtWidgets.QApplication.primaryScreen()
    screen_geometry = screen.geometry()
    available_geometry = screen.availableGeometry()

    # Calculate the offset caused by the menu bar and window decorations
    x_offset = screen_geometry.x() - available_geometry.x()
    y_offset = screen_geometry.y() - available_geometry.y()

    return x_offset, y_offset

class Canvas2DWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: white;")
        self.resize(SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX)
        self.circle_radius = 20
        self.current_position = [0.5, 0.5]  # Normalized screen position (center)
        self.src_color = QtGui.QColor(255, 0, 0)
        self.current_color = QtGui.QColor(255, 0, 0)
        self.target_color = QtGui.QColor(255, 255, 255)
        # self.calibration_steps = 9
        self.current_step = 0

        self.calib_positions = np.array([ # 9 points
            [0.5, 0.5],
            [0.1, 0.1],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.9, 0.9],
            [0.5, 0.1],
            [0.5, 0.9],
            [0.1, 0.5],
            [0.9, 0.5]
        ])

        self.show_instructions = True
        self.complete = False
        self.instruction_timer = QtCore.QTimer(self)
        self.instruction_timer.setSingleShot(True)
        self.instruction_timer.timeout.connect(self.start_calibration)
        self.instruction_timer.start(5*1000)  # 10 seconds

        self.color_transition_timer = QtCore.QTimer(self)
        self.color_transition_timer.timeout.connect(self.update_circle_color)

        self.gradient_step = 0
        self.gradient_duration = 2000  # 5 seconds in milliseconds
        self.gradient_total_steps = self.gradient_duration // 50

        # Add a return button
        self.return_button = QtWidgets.QPushButton("Return", self)
        # self.return_button.setGeometry(10, 10, 100, 30)
        # Place the button in the center
        self.return_button.move(self.width() // 2 - 25, self.height() // 2 + 30)
        self.return_button.setStyleSheet("""
            background-color: black;
            color: white;
            border-radius: 10px;
            border: 3px solid white;
            padding: 10px;
        """)
        self.return_button.clicked.connect(self.on_return_clicked)
        self.return_button.hide()  # Initially hide the button

    def on_return_clicked(self):
        self.parent().parent().toggle_canvas()  # Assuming parent manages the toggle to GLViewWidget

    def start_calibration(self):
        self.show_instructions = False
        self.current_step = 0
        self.start_color_transition()

    def start_color_transition(self):
        self.move_timer = QtCore.QTimer(self)
        self.move_timer.setSingleShot(True)
        self.move_timer.timeout.connect(self.move_circle)

        self.gradient_step = 0
        self.color_transition_timer.start(50)

    def update_circle_color(self):
        ratio = self.gradient_step / self.gradient_total_steps
        new_r = int(self.src_color.red() * (1 - ratio) + self.target_color.red() * ratio)
        new_g = int(self.src_color.green() * (1 - ratio) + self.target_color.green() * ratio)
        new_b = int(self.src_color.blue() * (1 - ratio) + self.target_color.blue() * ratio)

        self.current_color = QtGui.QColor(new_r, new_g, new_b)
        self.update()

        self.gradient_step += 1
        if self.gradient_step > self.gradient_total_steps:
            self.color_transition_timer.stop()
            self.move_timer.start(0)

    def move_circle(self):
        self.move_timer.stop()
        self.current_step += 1
        if self.current_step >= len(self.calib_positions):
            self.show_instructions = True
            self.complete = True
            self.update()
            self.return_button.show()
            return

        self.current_position = self.calib_positions[self.current_step]
        self.current_color = QtGui.QColor(255, 0, 0)  # Reset to red
        self.start_color_transition()
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Get the screen offset
        x_offset, y_offset = get_screen_offset()
        painter.translate(x_offset, y_offset)  # Shift the canvas up by the offset

        # DEBUG
        # Draw gridlines for debugging purposes
        painter.setPen(QtGui.QPen(QtGui.QColor(200, 200, 200), 1))
        for x in range(0, self.width(), self.width() // 10):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), self.height() // 10):
            painter.drawLine(0, y, self.width(), y)

        # Draw X and Y axes at the origin (center of the widget)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 2))
        painter.drawLine(self.width() // 2, 0, self.width() // 2, self.height())  # Y-axis
        painter.drawLine(0, self.height() // 2, self.width(), self.height() // 2)  # X-axis

        if self.show_instructions:
            # Draw instructions
            if self.complete:
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
                painter.setFont(QtGui.QFont("Arial", 16))
                instructions = "Calibration complete"
                text_rect = self.rect()
                painter.drawText(text_rect, QtCore.Qt.AlignCenter, instructions)

            else:
                painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
                painter.setFont(QtGui.QFont("Arial", 16))
                instructions = "Look at the dots until they turn white."
                text_rect = self.rect()
                painter.drawText(text_rect, QtCore.Qt.AlignCenter, instructions)
        else:
            # Draw the circle
            painter.setBrush(QtGui.QBrush(self.current_color))
            circle_center_x = int(self.width() * self.current_position[0])
            circle_center_y = int(self.height() * self.current_position[1])
            painter.drawEllipse(circle_center_x - self.circle_radius, 
                                circle_center_y - self.circle_radius, 
                                self.circle_radius * 2, 
                                self.circle_radius * 2)

class GazeDotCanvas(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground)
        self.setStyleSheet("background: transparent;")
        self.dot_position = [0.5, 0.5]  # Normalized position (center)
        self.dot_radius = 15
        self.dot_color = QtGui.QColor(0, 255, 0)

    def update_dot(self, x, y):
        """Update the dot's position and trigger a repaint."""
        self.dot_position = [x, y]
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Get the screen offset
        x_offset, y_offset = get_screen_offset()
        painter.translate(x_offset, y_offset)  # Shift the canvas up by the offset

        painter.setBrush(QtGui.QBrush(self.dot_color))
        circle_center_x = int(self.width() * self.dot_position[0])
        circle_center_y = int(self.height() * self.dot_position[1])
        painter.drawEllipse(circle_center_x - self.dot_radius,
                            circle_center_y - self.dot_radius,
                            self.dot_radius * 2,
                            self.dot_radius * 2)

class PointCloudApp(QtWidgets.QMainWindow):
    gaze_dot_updated = QtCore.pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        
        # Set up window properties
        self.setWindowTitle("3D Point Cloud Viewer with Webcam Overlay")
        self.resize(SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX)

        # Main widget to hold the GLViewWidget and overlayed webcam
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QStackedLayout(self.central_widget)

        # Set up the GL view widget
        self.gl_widget = gl.GLViewWidget()
        self.layout.addWidget(self.gl_widget)

        # Add 2D canvas
        self.canvas_2d = Canvas2DWidget()
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

        # Add grid to the view
        grid = gl.GLGridItem()
        self.gl_widget.addItem(grid)

        # Initialize an empty point cloud
        self.point_cloud_item = gl.GLScatterPlotItem()
        self.gl_widget.addItem(self.point_cloud_item)
        
        # Add static elements
        self.add_screen_rect() # Add first
        self.add_xyz_axes()
        self.add_camera_frustum()
        self.add_gaze_elements()

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

    def toggle_canvas(self):
        if self.canvas_2d.isVisible():
            self.canvas_2d.hide()
            self.gl_widget.show()
            self.ui_container.show()
            self.webcam_label.show()
        else:
            self.gl_widget.hide()
            self.canvas_2d.show()
            self.ui_container.hide()
            self.webcam_label.hide()
            # self.ui_container.raise_()

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

    def add_screen_rect(self):

        # Define rectangle corner points
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
        frustrum_scale = 1.0
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

    def add_gaze_elements(self):
        # Create left and right eyeballs
        self.left_eyeball = gl.GLScatterPlotItem(size=10, color=(1, 1, 1, 1))
        self.right_eyeball = gl.GLScatterPlotItem(size=10, color=(1, 1, 1, 1))
        self.gl_widget.addItem(self.left_eyeball)
        self.gl_widget.addItem(self.right_eyeball)

        # Create left and right points-of-gaze
        self.left_pog = gl.GLScatterPlotItem(size=10, color=(0, 1, 0, 1))
        self.right_pog = gl.GLScatterPlotItem(size=10, color=(0, 0, 1, 1))
        self.gl_widget.addItem(self.left_pog)
        self.gl_widget.addItem(self.right_pog)

        # Create left and right gaze vectors
        self.left_gaze_vector = gl.GLLinePlotItem(color=(0, 1, 0, 1), width=2)
        self.right_gaze_vector = gl.GLLinePlotItem(color=(0, 0, 1, 1), width=2)
        self.gl_widget.addItem(self.left_gaze_vector)
        self.gl_widget.addItem(self.right_gaze_vector)

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

                # Update the pointclouds based on the face points
                # Get 3D landmark positions for the Face Mesh
                points = result.tf_facial_landmarks[:, :3] * SCALE
                points = np.dot(points, R)
                color = (1, 0, 0, 1)
                self.point_cloud_item.setData(pos=points, color=color, size=5)

                for side in ['left', 'right']:
                    e = result.left if side == 'left' else result.right
                    eyeball = self.left_eyeball if side == 'left' else self.right_eyeball
                    gaze_vector = self.left_gaze_vector if side == 'left' else self.right_gaze_vector
                    pog = self.left_pog if side == 'left' else self.right_pog

                    # Eyeball position
                    origin = np.dot(e.origin, R)
                    direction = np.dot(e.direction, R)
                    eyeball.setData(pos=origin * SCALE, size=20)

                    # Gaze vector (line from origin to scaled direction)
                    points = np.array([
                        origin,
                        origin + direction * np.array([1, 1, 1]) * 1e3
                    ]) * SCALE
                    gaze_vector.setData(pos=points)

                    # Point-of-gaze
                    pog_mm_c_3d = np.array([[e.pog_mm_c[0], e.pog_mm_c[1], 0]])
                    pog_mm_c = np.dot(pog_mm_c_3d, R)
                    pog.setData(pos=pog_mm_c * SCALE, size=20)

                # Update the 2D PoG
                # gaze_x, gaze_y = np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)
                gaze_x, gaze_y = result.pog_norm[0], result.pog_norm[1]
                self.gaze_dot_updated.emit(gaze_x, gaze_y)

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
