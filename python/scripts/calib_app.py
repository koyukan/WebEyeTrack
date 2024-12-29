import sys
from PyQt5 import QtWidgets
import pyqtgraph.opengl as gl
import numpy as np

class PointCloudApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the GL view widget
        self.gl_widget = gl.GLViewWidget()
        self.setCentralWidget(self.gl_widget)

        # Add grid to the view
        grid = gl.GLGridItem()
        self.gl_widget.addItem(grid)

        # Generate random 3D points
        num_points = 1000
        pos = np.random.rand(num_points, 3) * 10 - 5  # Random points in a 10x10x10 cube

        # Create a GL scatter plot
        scatter_plot = gl.GLScatterPlotItem(pos=pos, color=(1, 0, 0, 1), size=5)
        self.gl_widget.addItem(scatter_plot)

        # Set up window properties
        self.setWindowTitle("3D Point Cloud Viewer")
        self.resize(800, 600)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = PointCloudApp()
    window.show()
    sys.exit(app.exec_())
