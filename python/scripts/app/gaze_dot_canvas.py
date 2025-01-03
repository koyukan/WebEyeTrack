from PyQt5 import QtWidgets, QtGui, QtCore

from constants import get_screen_offset

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