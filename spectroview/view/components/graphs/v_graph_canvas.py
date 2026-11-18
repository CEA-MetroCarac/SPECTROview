"""Debounced FigureCanvas and fast-pan NavigationToolbar for VGraph."""

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from PySide6.QtCore import QTimer


class _DebouncedFigureCanvas(FigureCanvas):
    """FigureCanvasQTAgg whose resize-triggered redraws are debounced.

    The stock canvas re-renders synchronously on *every* resizeEvent; with
    ``layout="compressed"`` each render re-runs constrained layout
    (hundreds of ms), so rapid interactive resizes queue an Agg redraw storm.
    This canvas updates figure dimensions in inches immediately (so transforms
    stay valid) while coalescing redraws into a single trailing pass once
    resizing settles.
    """

    RESIZE_DEBOUNCE_MS = 60

    def __init__(self, figure):
        super().__init__(figure)
        # Parented to the canvas: dies with it (no stray C++ timer).
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._flush_pending_resize)

    def resizeEvent(self, event):
        w = event.size().width()
        h = event.size().height()
        if w <= 0 or h <= 0:
            return
        dpi = self.figure.dpi
        self.figure.set_size_inches(w / dpi, h / dpi, forward=False)
        self._resize_timer.start(self.RESIZE_DEBOUNCE_MS)

    def _flush_pending_resize(self):
        self.draw_idle()


class _FastNavigationToolbar(NavigationToolbar2QT):
    """Hidden matplotlib navigation helper with a fast interactive pan.

    The stock pan redraws the whole figure -- including a constrained-layout
    pass -- on every mouse-move. During a drag the layout cannot change
    (only axis limits do), so the layout engine is suspended between press
    and release, and drag redraws are throttled to ~``PAN_FRAME_S``. The
    layout engine is restored and a final full redraw issued on release.
    """

    PAN_FRAME_S = 0.03

    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        self._suspended_layout_engine = None
        self._last_pan_draw = 0.0

    def _suspend_layout(self):
        fig = self.canvas.figure
        if self._suspended_layout_engine is None:
            engine = fig.get_layout_engine()
            if engine is not None:
                self._suspended_layout_engine = engine
                fig.set_layout_engine(None)

    def _restore_layout(self):
        if self._suspended_layout_engine is not None:
            self.canvas.figure.set_layout_engine(self._suspended_layout_engine)
            self._suspended_layout_engine = None

    def press_pan(self, event):
        super().press_pan(event)
        if self._pan_info is not None:
            self._suspend_layout()
            self._last_pan_draw = 0.0

    def drag_pan(self, event):
        import time
        if self._pan_info is None:
            return
        for ax in self._pan_info.axes:
            ax.drag_pan(self._pan_info.button, event.key, event.x, event.y)
        now = time.monotonic()
        if now - self._last_pan_draw >= self.PAN_FRAME_S:
            self._last_pan_draw = now
            self.canvas.draw_idle()

    def release_pan(self, event):
        try:
            super().release_pan(event)  # applies the final limits + draw_idle
        finally:
            self._restore_layout()
            self.canvas.draw_idle()
