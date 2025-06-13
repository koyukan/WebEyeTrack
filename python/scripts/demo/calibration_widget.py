from PyQt5 import QtWidgets, QtGui, QtCore

from constants import *

class CalibrationWidget(QtWidgets.QWidget):
    calibration_status = QtCore.pyqtSignal(bool)
    calib_dot_updated = QtCore.pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: white;")
        self.resize(SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX)
        self.circle_radius = 20
        self.src_color = QtGui.QColor(255, 0, 0)
        self.current_color = QtGui.QColor(255, 0, 0)
        self.target_color = QtGui.QColor(255, 255, 255)
        # self.calibration_steps = 9
        self.current_step = 0

        self.calib_positions = np.array([ # 4 points
            [-0.4, -0.4],
            [0.4, -0.4],
            [-0.4, 0.4],
            [0.4, 0.4],
        ])
        self.current_position = self.calib_positions[0]

        self.show_instructions = True
        self.complete = False
        self.instruction_timer = QtCore.QTimer(self)
        self.instruction_timer.setSingleShot(True)
        self.instruction_timer.timeout.connect(self.perform_calibration)

        self.color_transition_timer = QtCore.QTimer(self)
        self.color_transition_timer.timeout.connect(self.update_circle_color)

        self.gradient_step = 0
        self.gradient_duration = 2000  # 5 seconds in milliseconds
        self.gradient_total_steps = self.gradient_duration // 50

        # Add a layout to manage the widget's contents
        self.layout = QtWidgets.QVBoxLayout(self)

        # Add a return button
        self.return_button = QtWidgets.QPushButton("Return")
        self.layout.addWidget(self.return_button, alignment=QtCore.Qt.AlignCenter)
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
        self.parent().parent().end_calib()  # Assuming parent manages the toggle to GLViewWidget

    def start_calibration(self):
        self.show_instructions = True
        self.complete = False
        self.instruction_timer.start(5*1000)  # 10 seconds

    def perform_calibration(self):
        self.show_instructions = False
        self.current_step = 0
        self.current_position = self.calib_positions[self.current_step]
        self.calibration_status.emit(True)
        self.calib_dot_updated.emit(self.current_position[0], self.current_position[1])
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
            self.calibration_status.emit(False)
            self.update()
            self.return_button.show()
            return

        self.current_position = self.calib_positions[self.current_step]
        self.current_color = QtGui.QColor(255, 0, 0)  # Reset to red
        self.start_color_transition()
        self.update()
        self.calib_dot_updated.emit(self.current_position[0], self.current_position[1])

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
                painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0)))
                painter.setFont(QtGui.QFont("Arial", 16))
                painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 200)))
                instructions = "Calibration complete"
                text_rect = self.rect()
                painter.drawText(text_rect, QtCore.Qt.AlignCenter, instructions)

            else:
                painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0)))
                painter.setFont(QtGui.QFont("Arial", 16))
                instructions = "Look at the dots until they turn white."
                painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 200)))
                text_rect = self.rect()
                painter.drawText(text_rect, QtCore.Qt.AlignCenter, instructions)
        else:
            # Draw the circle
            painter.setBrush(QtGui.QBrush(self.current_color))
            circle_center_x = int(self.width() * (self.current_position[0] + 0.5))
            circle_center_y = int(self.height() * (self.current_position[1] + 0.5))
            painter.drawEllipse(circle_center_x - self.circle_radius, 
                                circle_center_y - self.circle_radius, 
                                self.circle_radius * 2, 
                                self.circle_radius * 2)