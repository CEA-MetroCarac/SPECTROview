from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication, QDialog, QVBoxLayout, QLineEdit
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtGui import QResizeEvent
from PySide6.QtCore import QTimer

class Graph(QDialog):
    plotSelected = Signal(str)  # Signal emitted when a plot is selected

    num_plots = 0  # Class variable to track the count of created windows

    def __init__(self):
        super(Graph, self).__init__()
        self.setWindowTitle("Graph Plot")
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.ax = self.figure.add_subplot(111)

        # Update window title based on the count of created windows
        Graph.num_plots += 1
        self.setWindowTitle(f"Graph Plot {Graph.num_plots}")

        # Connect plotSelected signal to update_title slot
        self.plotSelected.connect(self.update_title)

        # Connect resizeEvent to apply tight_layout and redraw the plot
        self.resizeEvent = self.customResizeEvent

    def customResizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        self.tight_layout_and_redraw()

    def tight_layout_and_redraw(self):
        self.figure.tight_layout()
        self.canvas.draw()

    def plot(self, x_data, y_data, title="", xlabel="", ylabel=""):
        self.ax.clear()
        self.ax.plot(x_data, y_data)
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.tight_layout_and_redraw()

    def update_title(self, new_title):
        self.setWindowTitle(new_title)

    def mousePressEvent(self, event):
        self.plotSelected.emit(self.windowTitle())

    def update_plot_title(self, new_title):
        self.setWindowTitle(new_title)

    def set_title(self, title):
        self.ax.set_title(title)
        self.tight_layout_and_redraw()

    def set_xlabel(self, xlabel):
        self.ax.set_xlabel(xlabel)
        self.tight_layout_and_redraw()

    def set_ylabel(self, ylabel):
        self.ax.set_ylabel(ylabel)
        self.tight_layout_and_redraw()

    def set_xlimits(self, xmin, xmax):
        self.ax.set_xlim(xmin, xmax)
        self.tight_layout_and_redraw()

    def set_ylimits(self, ymin, ymax):
        self.ax.set_ylim(ymin, ymax)
        self.tight_layout_and_redraw()

    def set_legend(self, legend_labels):
        self.ax.legend(legend_labels)
        self.tight_layout_and_redraw()
