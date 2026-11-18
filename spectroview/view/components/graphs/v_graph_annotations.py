"""Annotation rendering and interactive canvas drag/resize/edit mixin for VGraph."""

import matplotlib.patches as patches
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog

from spectroview.view.components.customize_graph.customize_annotation_dialogs import (
    EditArrowDialog,
    EditBoxDialog,
    EditCalloutDialog,
    EditLineDialog,
    EditSpanDialog,
    EditTextDialog,
)


class GraphAnnotationMixin:
    """Mixin handling annotation artist rendering, hit-testing, dragging, resizing, and editing."""

    # Pixel proximity for grabbing a resize border/endpoint instead of
    # moving the whole annotation.
    _HANDLE_TOL_PX = 8

    _HANDLE_CURSORS = {
        'resize-x1': Qt.CursorShape.SizeHorCursor, 'resize-x2': Qt.CursorShape.SizeHorCursor,
        'resize-l': Qt.CursorShape.SizeHorCursor, 'resize-r': Qt.CursorShape.SizeHorCursor,
        'resize-y1': Qt.CursorShape.SizeVerCursor, 'resize-y2': Qt.CursorShape.SizeVerCursor,
        'resize-t': Qt.CursorShape.SizeVerCursor, 'resize-b': Qt.CursorShape.SizeVerCursor,
        'resize-tl': Qt.CursorShape.SizeFDiagCursor, 'resize-br': Qt.CursorShape.SizeFDiagCursor,
        'resize-tr': Qt.CursorShape.SizeBDiagCursor, 'resize-bl': Qt.CursorShape.SizeBDiagCursor,
    }

    # Annotation type -> its modal edit dialog (double-click on canvas).
    _EDIT_DIALOGS = {
        'vline': EditLineDialog, 'hline': EditLineDialog,
        'text': EditTextDialog, 'arrow': EditArrowDialog,
        'vspan': EditSpanDialog, 'hspan': EditSpanDialog,
        'box': EditBoxDialog, 'callout': EditCalloutDialog,
    }

    def _render_annotations(self):
        """Render all annotations (lines and text) on the plot."""
        if not self.annotations:
            return

        for ann in self.annotations:
            ann_type = ann.get('type')
            try:
                if ann_type == 'vline':
                    self._render_vline(ann)
                elif ann_type == 'hline':
                    self._render_hline(ann)
                elif ann_type == 'text':
                    self._render_text(ann)
                elif ann_type == 'arrow':
                    self._render_arrow(ann)
                elif ann_type in ('vspan', 'hspan'):
                    self._render_span(ann)
                elif ann_type == 'box':
                    self._render_box(ann)
                elif ann_type == 'callout':
                    self._render_callout(ann)
            except Exception as e:
                print(f"[WARNING] Failed to render annotation {ann.get('id')}: {e}")

    def _render_vline(self, ann: dict):
        """Render vertical line annotation."""
        x_pos = ann.get('x', 0)
        color = ann.get('color', 'red')
        linestyle = ann.get('linestyle', '--')
        linewidth = ann.get('linewidth', 1.5)
        label = ann.get('label', None)

        line = self.ax.axvline(
            x=x_pos,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
            zorder=100,  # Render on top of other elements
            picker=5  # Enable picking with 5pt tolerance
        )

        # Attach metadata for drag handling
        line._annotation_data = ann
        line._is_dragging = False
        return line

    def _render_hline(self, ann: dict):
        """Render horizontal line annotation."""
        y_pos = ann.get('y', 0)
        color = ann.get('color', 'blue')
        linestyle = ann.get('linestyle', '--')
        linewidth = ann.get('linewidth', 1.5)
        label = ann.get('label', None)

        line = self.ax.axhline(
            y=y_pos,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
            zorder=100,  # Render on top of other elements
            picker=5  # Enable picking with 5pt tolerance
        )

        # Attach metadata for drag handling
        line._annotation_data = ann
        line._is_dragging = False
        return line

    def _render_text(self, ann: dict):
        """Render text annotation."""
        x_pos = ann.get('x', 0)
        y_pos = ann.get('y', 0)
        text = ann.get('text', '')
        fontsize = ann.get('fontsize', 12)
        color = ann.get('color', 'black')
        ha = ann.get('ha', 'center')
        va = ann.get('va', 'center')

        # Get bbox from annotation, use default if not specified
        bbox_props = ann.get('bbox')

        text_obj = self.ax.text(
            x_pos,
            y_pos,
            text,
            fontsize=fontsize,
            color=color,
            ha=ha,
            va=va,
            bbox=bbox_props,
            zorder=101,  # Render on top of lines
            picker=True  # Enable picking for drag functionality
        )

        # Attach metadata for drag handling
        text_obj._annotation_data = ann
        text_obj._is_dragging = False
        return text_obj

    def _render_arrow(self, ann: dict):
        """Render arrow annotation.

        Uses a FancyArrowPatch (added via add_patch) rather than
        ax.annotate('', ...) -- Patch.contains() picks along the whole
        drawn arrow shape, whereas an empty-string Annotation's pick region
        is effectively just its (near zero-size) text anchor point, which
        would make double-click-to-edit and drag only work when clicking
        almost exactly on the arrow tip.
        """
        x1, y1 = ann.get('x1', 0), ann.get('y1', 0)
        x2, y2 = ann.get('x2', 0), ann.get('y2', 0)
        color = ann.get('color', 'black')
        linewidth = ann.get('linewidth', 1.5)
        linestyle = ann.get('linestyle', '-')

        arrow = patches.FancyArrowPatch(
            (x1, y1), (x2, y2),
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            arrowstyle='-|>',
            mutation_scale=15,
            zorder=100,
            picker=True,
        )
        self.ax.add_patch(arrow)

        arrow._annotation_data = ann
        arrow._is_dragging = False
        return arrow

    def _render_span(self, ann: dict):
        """Render a vertical or horizontal shaded span (axvspan/axhspan).

        Both return a plain Rectangle patch under the hood (confirmed:
        axvspan's y-extent is a blended-transform 0-1 axes fraction, x is
        data coords, and vice-versa for axhspan) -- that's what makes
        dragging just a set_x()/set_y() call, no custom polygon math.
        """
        ann_type = ann.get('type')
        color = ann.get('color', 'orange')
        alpha = ann.get('alpha', 0.3)

        if ann_type == 'vspan':
            x1, x2 = ann.get('x1', 0), ann.get('x2', 1)
            span = self.ax.axvspan(x1, x2, color=color, alpha=alpha, zorder=50, picker=True)
        else:
            y1, y2 = ann.get('y1', 0), ann.get('y2', 1)
            span = self.ax.axhspan(y1, y2, color=color, alpha=alpha, zorder=50, picker=True)

        span._annotation_data = ann
        span._is_dragging = False
        return span

    def _render_box(self, ann: dict):
        """Render a rectangle/box annotation."""
        x, y = ann.get('x', 0), ann.get('y', 0)
        width = ann.get('width', 1)
        height = ann.get('height', 1)
        facecolor = ann.get('facecolor', 'yellow')
        edgecolor = ann.get('edgecolor', 'black')
        linewidth = ann.get('linewidth', 1.5)
        alpha = ann.get('alpha', 0.3)

        box = patches.Rectangle(
            (x, y), width, height,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
            zorder=99,
            picker=True,
        )
        self.ax.add_patch(box)

        box._annotation_data = ann
        box._is_dragging = False
        return box

    def _render_callout(self, ann: dict):
        """Render a callout: text with an arrow pointing at a data point."""
        x, y = ann.get('x', 0), ann.get('y', 0)
        tx, ty = ann.get('tx', x), ann.get('ty', y)
        text = ann.get('text', '')
        fontsize = ann.get('fontsize', 11)
        color = ann.get('color', 'black')
        arrowcolor = ann.get('arrowcolor', 'black')

        callout = self.ax.annotate(
            text, xy=(x, y), xytext=(tx, ty),
            fontsize=fontsize, color=color,
            arrowprops=dict(arrowstyle='->', color=arrowcolor),
            zorder=101,
            picker=True,
        )

        callout._annotation_data = ann
        callout._is_dragging = False
        return callout

    def _ax_data_coords(self, event):
        """(x, y) of a mouse event in self.ax's data coordinates, computed
        from the raw pixel position -- not event.xdata/ydata.

        event.xdata/ydata come from event.inaxes, which resolves to the
        topmost overlapping Axes: a twinx() secondary axis covers self.ax
        entirely, so relying on them broke annotation picking/dragging on
        any graph with a secondary axis. Transforming through
        self.ax.transData directly sidesteps the ambiguity.
        """
        return self.ax.transData.inverted().transform((event.x, event.y))

    def _annotation_handle_at(self, artist, ann_data, event):
        """Which draggable handle of `artist` the mouse is over, or 'move'.

        Spans/boxes expose their borders as resize handles, arrows their two
        endpoints, callouts their pointed-at anchor; anything else is a
        whole-annotation move. Measured in pixels so grabbing feels the same
        at any data scale (via self.ax.transData, not event.xdata -- see
        _ax_data_coords for the twin-axis reason)."""
        t = self.ax.transData.transform
        tol = self._HANDLE_TOL_PX
        ann_type = ann_data['type']

        if ann_type == 'vspan':
            px1, px2 = t((ann_data['x1'], 0))[0], t((ann_data['x2'], 0))[0]
            if abs(event.x - px1) <= tol:
                return 'resize-x1'
            if abs(event.x - px2) <= tol:
                return 'resize-x2'
        elif ann_type == 'hspan':
            py1, py2 = t((0, ann_data['y1']))[1], t((0, ann_data['y2']))[1]
            if abs(event.y - py1) <= tol:
                return 'resize-y1'
            if abs(event.y - py2) <= tol:
                return 'resize-y2'
        elif ann_type == 'box':
            left, bottom = t((ann_data['x'], ann_data['y']))
            right, top = t((ann_data['x'] + ann_data['width'],
                            ann_data['y'] + ann_data['height']))
            horiz = 'l' if abs(event.x - left) <= tol else ('r' if abs(event.x - right) <= tol else '')
            vert = 't' if abs(event.y - top) <= tol else ('b' if abs(event.y - bottom) <= tol else '')
            within_x = min(left, right) - tol <= event.x <= max(left, right) + tol
            within_y = min(bottom, top) - tol <= event.y <= max(bottom, top) + tol
            if vert and horiz and within_x and within_y:
                return f'resize-{vert}{horiz}'
            if horiz and within_y:
                return f'resize-{horiz}'
            if vert and within_x:
                return f'resize-{vert}'
        elif ann_type == 'arrow':
            p1, p2 = t((ann_data['x1'], ann_data['y1'])), t((ann_data['x2'], ann_data['y2']))
            if abs(event.x - p1[0]) <= tol and abs(event.y - p1[1]) <= tol:
                return 'p1'
            if abs(event.x - p2[0]) <= tol and abs(event.y - p2[1]) <= tol:
                return 'p2'
        elif ann_type == 'callout':
            anchor = t((ann_data['x'], ann_data['y']))
            if abs(event.x - anchor[0]) <= tol and abs(event.y - anchor[1]) <= tol:
                return 'anchor'
        return 'move'

    def _update_hover_cursor(self, event):
        """Show a resize/move cursor while hovering an annotation handle so
        the grab affordance is discoverable. Skipped while a matplotlib
        pan/zoom mode owns the cursor."""
        if not self.annotations or str(getattr(self.toolbar, 'mode', '')):
            return
        shape = None
        for artist in self.ax.findobj():
            data = getattr(artist, '_annotation_data', None)
            if data is not None and artist.contains(event)[0]:
                handle = self._annotation_handle_at(artist, data, event)
                shape = self._HANDLE_CURSORS.get(handle, Qt.CursorShape.SizeAllCursor)
                break
        # Only touch the cursor when the state changes -- avoids fighting
        # other widgets for the cursor on every idle mouse move.
        if shape != getattr(self, '_hover_cursor_shape', None):
            self._hover_cursor_shape = shape
            self.canvas.setCursor(shape if shape is not None else Qt.CursorShape.ArrowCursor)

    def _on_annotation_click(self, event):
        """Handle a press on the canvas: a single click on an annotation
        starts a potential drag/resize, a double-click opens its edit dialog.

        Hit-testing is done manually against self.ax's own annotation
        artists (self.ax.findobj() + artist.contains(event)) rather than
        relying on matplotlib's pick_event -- Figure.pick() only delivers a
        pick event to an artist when event.inaxes equals that artist's own
        Axes, which (see _ax_data_coords) is unreliable whenever a
        secondary Y-axis overlaps self.ax. artist.contains(event) itself
        only needs the event's pixel position, which is always correct.
        """
        if event.x is None or event.y is None:
            return

        # Cancel any pending/in-progress drag -- a new press always
        # supersedes it, whether or not this press turns out to be on an
        # annotation.
        if hasattr(self, '_drag_candidate'):
            del self._drag_candidate
        if hasattr(self, '_dragged_annotation'):
            self._dragged_annotation._is_dragging = False
            del self._dragged_annotation

        for ann in self.ax.findobj():
            if hasattr(ann, '_annotation_data'):
                contains, _ = ann.contains(event)
                if contains:
                    if event.dblclick:
                        self._edit_annotation_direct(ann._annotation_data)
                    else:
                        # Record as drag candidate -- actual drag starts on
                        # mouse move past a small threshold (see
                        # _on_annotation_drag). The handle under the press
                        # decides move-vs-resize for the whole gesture.
                        self._drag_candidate = ann
                        self._drag_handle = self._annotation_handle_at(ann, ann._annotation_data, event)
                        self._drag_start_x, self._drag_start_y = self._ax_data_coords(event)
                    return

    def _edit_annotation_direct(self, annotation):
        """Open the edit dialog for an annotation (double-click on canvas).
        The dialog previews live against this graph and restores on Cancel."""
        dialog_cls = self._EDIT_DIALOGS.get(annotation['type'])
        if dialog_cls is None:
            return
        dialog = dialog_cls(annotation, self, None)
        if dialog.exec() == QDialog.Accepted:
            annotation.update(dialog.get_properties())
            if annotation['type'] == 'vline':
                annotation['label'] = f"V-Line at x={annotation['x']:.2f}"
            elif annotation['type'] == 'hline':
                annotation['label'] = f"H-Line at y={annotation['y']:.2f}"
            if self.df is not None:
                self.plot(self.df)

    def _update_coordinates(self, event):
        """Update built-in live coordinate display label under the graph canvas.

        Uses toolbar._mouse_event_to_message when available so that artist cursor
        data (such as the Z value on 2D heatmaps and wafer maps) is formatted and
        preserved along with the spatial (X, Y) coordinates.
        """
        if hasattr(self, "toolbar") and hasattr(self.toolbar, "_mouse_event_to_message"):
            try:
                msg = self.toolbar._mouse_event_to_message(event)
                self.toolbar.set_message(msg)
                return
            except Exception:
                pass
        if hasattr(self, "lbl_coordinates"):
            inaxes = getattr(event, "inaxes", None)
            xdata = getattr(event, "xdata", None)
            ydata = getattr(event, "ydata", None)
            if inaxes and xdata is not None and ydata is not None:
                try:
                    coord_str = inaxes.format_coord(xdata, ydata)
                    clean_str = " ".join(coord_str.split())
                    self.lbl_coordinates.setText(clean_str)
                    if hasattr(self.toolbar, "locLabel"):
                        self.toolbar.locLabel.setText(clean_str)
                except Exception:
                    self.lbl_coordinates.setText("")
            else:
                self.lbl_coordinates.setText("")

    def _on_figure_leave(self, event):
        """Clear live coordinate display when cursor leaves the graph canvas."""
        if hasattr(self, "lbl_coordinates"):
            self.lbl_coordinates.setText("")
        if hasattr(self.toolbar, "locLabel"):
            self.toolbar.locLabel.setText("")

    def _on_annotation_drag(self, event):
        """Handle annotation drag (mouse move while dragging).

        Uses _ax_data_coords(event) throughout instead of event.xdata/
        event.ydata -- see that method's docstring for why those are
        unreliable whenever a secondary Y-axis overlaps self.ax.
        """
        self._update_coordinates(event)
        if event.x is None or event.y is None:
            return
        xdata, ydata = self._ax_data_coords(event)

        # Idle hover (no active/pending drag): show the grab affordance.
        if not hasattr(self, '_drag_candidate') and not hasattr(self, '_dragged_annotation'):
            self._update_hover_cursor(event)
            return

        # Promote drag candidate to actual drag once mouse moves
        if hasattr(self, '_drag_candidate') and not hasattr(self, '_dragged_annotation'):
            dx = abs(xdata - (self._drag_start_x or 0))
            dy = abs(ydata - (self._drag_start_y or 0))
            # Use a small threshold to distinguish click from drag
            x_range = abs(self.ax.get_xlim()[1] - self.ax.get_xlim()[0])
            y_range = abs(self.ax.get_ylim()[1] - self.ax.get_ylim()[0])
            if dx > x_range * 0.005 or dy > y_range * 0.005:
                self._dragged_annotation = self._drag_candidate
                self._dragged_annotation._is_dragging = True
                # Snapshot for delta-based dragging of multi-point shapes
                # below (vline/hline/text jump straight to the mouse instead).
                self._drag_start_coords = dict(self._dragged_annotation._annotation_data)
                del self._drag_candidate

        if not hasattr(self, '_dragged_annotation'):
            return

        ann = self._dragged_annotation
        if not getattr(ann, '_is_dragging', False):
            return

        ann_data = ann._annotation_data
        ann_type = ann_data['type']

        # A border/endpoint grab resizes; anything else moves the whole shape.
        if getattr(self, '_drag_handle', 'move') != 'move':
            self._resize_dragged_annotation(ann, ann_data, xdata, ydata)
            self.canvas.draw_idle()
            return

        # Update visual position based on annotation type
        if ann_type == 'vline':
            ann.set_xdata([xdata, xdata])
            ann_data['x'] = xdata
        elif ann_type == 'hline':
            ann.set_ydata([ydata, ydata])
            ann_data['y'] = ydata
        elif ann_type == 'text':
            ann.set_position((xdata, ydata))
            ann_data['x'] = xdata
            ann_data['y'] = ydata
        elif ann_type == 'arrow':
            start = self._drag_start_coords
            dx = xdata - self._drag_start_x
            dy = ydata - self._drag_start_y
            ann_data['x1'] = start['x1'] + dx
            ann_data['y1'] = start['y1'] + dy
            ann_data['x2'] = start['x2'] + dx
            ann_data['y2'] = start['y2'] + dy
            ann.set_positions((ann_data['x1'], ann_data['y1']), (ann_data['x2'], ann_data['y2']))
        elif ann_type == 'vspan':
            start = self._drag_start_coords
            dx = xdata - self._drag_start_x
            ann_data['x1'] = start['x1'] + dx
            ann_data['x2'] = start['x2'] + dx
            ann.set_x(ann_data['x1'])
        elif ann_type == 'hspan':
            start = self._drag_start_coords
            dy = ydata - self._drag_start_y
            ann_data['y1'] = start['y1'] + dy
            ann_data['y2'] = start['y2'] + dy
            ann.set_y(ann_data['y1'])
        elif ann_type == 'box':
            start = self._drag_start_coords
            dx = xdata - self._drag_start_x
            dy = ydata - self._drag_start_y
            ann_data['x'] = start['x'] + dx
            ann_data['y'] = start['y'] + dy
            ann.set_xy((ann_data['x'], ann_data['y']))
        elif ann_type == 'callout':
            # Moves only the text position -- the arrow's pointed-at data
            # point (xy) is left fixed, matching set_position()'s own
            # behavior (it only ever touches xytext, never xy).
            ann.set_position((xdata, ydata))
            ann_data['tx'] = xdata
            ann_data['ty'] = ydata

        self.canvas.draw_idle()

    def _resize_dragged_annotation(self, ann, ann_data, xdata, ydata):
        """Move the grabbed border/endpoint (self._drag_handle) to the mouse,
        updating both the artist and ann_data. Span/box edges are clamped to
        a small minimum extent so a shape can't collapse to zero and become
        un-grabbable."""
        handle = self._drag_handle
        ann_type = ann_data['type']
        min_dx = abs(self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 0.01
        min_dy = abs(self.ax.get_ylim()[1] - self.ax.get_ylim()[0]) * 0.01

        if ann_type == 'vspan':
            x1, x2 = ann_data['x1'], ann_data['x2']
            if handle == 'resize-x1':
                x1 = min(xdata, x2 - min_dx)
            else:
                x2 = max(xdata, x1 + min_dx)
            ann_data['x1'], ann_data['x2'] = x1, x2
            ann.set_x(min(x1, x2))
            ann.set_width(abs(x2 - x1))
        elif ann_type == 'hspan':
            y1, y2 = ann_data['y1'], ann_data['y2']
            if handle == 'resize-y1':
                y1 = min(ydata, y2 - min_dy)
            else:
                y2 = max(ydata, y1 + min_dy)
            ann_data['y1'], ann_data['y2'] = y1, y2
            ann.set_y(min(y1, y2))
            ann.set_height(abs(y2 - y1))
        elif ann_type == 'box':
            suffix = handle.split('-', 1)[1]  # e.g. 'tl', 'r', 'b'
            x, y = ann_data['x'], ann_data['y']
            right, top = x + ann_data['width'], y + ann_data['height']
            if 'l' in suffix:
                x = min(xdata, right - min_dx)
            if 'r' in suffix:
                right = max(xdata, x + min_dx)
            if 'b' in suffix:
                y = min(ydata, top - min_dy)
            if 't' in suffix:
                top = max(ydata, y + min_dy)
            ann_data['x'], ann_data['y'] = x, y
            ann_data['width'], ann_data['height'] = right - x, top - y
            ann.set_bounds(x, y, right - x, top - y)
        elif ann_type == 'arrow':
            if handle == 'p1':
                ann_data['x1'], ann_data['y1'] = xdata, ydata
            else:
                ann_data['x2'], ann_data['y2'] = xdata, ydata
            ann.set_positions((ann_data['x1'], ann_data['y1']),
                              (ann_data['x2'], ann_data['y2']))
        elif ann_type == 'callout':
            # Move the pointed-at anchor; the text (xytext) stays put.
            ann_data['x'], ann_data['y'] = xdata, ydata
            ann.xy = (xdata, ydata)

    def _on_annotation_release(self, event):
        """Handle mouse release (finish dragging)."""
        # Clean up drag candidate if no drag occurred
        if hasattr(self, '_drag_candidate'):
            del self._drag_candidate

        if not hasattr(self, '_dragged_annotation'):
            return

        ann = self._dragged_annotation
        ann._is_dragging = False

        # Emit signal to update model
        ann_data = ann._annotation_data
        if ann_data['type'] == 'vline':
            self.annotation_position_changed.emit(
                self.graph_id, ann_data['id'], ann_data['x'], 0
            )
        elif ann_data['type'] == 'hline':
            self.annotation_position_changed.emit(
                self.graph_id, ann_data['id'], 0, ann_data['y']
            )
        elif ann_data['type'] == 'text':
            self.annotation_position_changed.emit(
                self.graph_id, ann_data['id'], ann_data['x'], ann_data['y']
            )
        elif ann_data['type'] == 'arrow':
            self.annotation_position_changed.emit(
                self.graph_id, ann_data['id'], ann_data['x1'], ann_data['y1']
            )
        elif ann_data['type'] == 'vspan':
            self.annotation_position_changed.emit(
                self.graph_id, ann_data['id'], ann_data['x1'], 0
            )
        elif ann_data['type'] == 'hspan':
            self.annotation_position_changed.emit(
                self.graph_id, ann_data['id'], 0, ann_data['y1']
            )
        elif ann_data['type'] == 'box':
            self.annotation_position_changed.emit(
                self.graph_id, ann_data['id'], ann_data['x'], ann_data['y']
            )
        elif ann_data['type'] == 'callout':
            self.annotation_position_changed.emit(
                self.graph_id, ann_data['id'], ann_data['tx'], ann_data['ty']
            )

        del self._dragged_annotation
        if hasattr(self, '_drag_start_coords'):
            del self._drag_start_coords
        self._drag_handle = 'move'
        # Force the next hover move to re-evaluate the cursor.
        self._hover_cursor_shape = None
        self.canvas.draw_idle()
