from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

class HistogramWidget(QWidget):
    rangeChanged = Signal(float, float)  # emits (vmin, vmax)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        
        self.figure = plt.Figure(figsize=(4,1))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        
        # initial data
        self.data = np.array([])
        self.vmin = None
        self.vmax = None
        
        # handles
        self.min_line = None
        self.max_line = None
        self.dragging = None  # 'min' or 'max'
        
        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)
    
    def set_data(self, data, vmin=None, vmax=None):
        """Update histogram data and redraw."""
        self.data = np.array(data)
        self.ax.clear()
        self.ax.hist(self.data, bins=100, color='blue', edgecolor='black')
        
        # Set bracket positions
        self.vmin = vmin if vmin is not None else np.min(data)
        self.vmax = vmax if vmax is not None else np.max(data)
        
        # Add draggable lines
        self.min_line = self.ax.axvline(self.vmin, color='red', lw=2, picker=5)
        self.max_line = self.ax.axvline(self.vmax, color='red', lw=2, picker=5)
        
        self.ax.set_yticks([])
        self.ax.set_xlim(np.min(data), np.max(data))
        self.canvas.draw_idle()
    
    def on_press(self, event):
        """Detect if user clicked near a line."""
        if event.inaxes != self.ax:
            return
        # threshold in data coordinates
        threshold = (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 0.02
        if abs(event.xdata - self.vmin) < threshold:
            self.dragging = 'min'
        elif abs(event.xdata - self.vmax) < threshold:
            self.dragging = 'max'
    
    def on_motion(self, event):
        """Drag the selected line."""
        if self.dragging is None or event.inaxes != self.ax:
            return
        x = event.xdata
        if self.dragging == 'min':
            self.vmin = min(x, self.vmax)
            self.min_line.set_xdata([self.vmin, self.vmin])
        elif self.dragging == 'max':
            self.vmax = max(x, self.vmin)
            self.max_line.set_xdata([self.vmax, self.vmax])
        self.canvas.draw_idle()
        self.rangeChanged.emit(self.vmin, self.vmax)
    
    def on_release(self, event):
        self.dragging = None
