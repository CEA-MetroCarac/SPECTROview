"""Whole-workspace graph snapshot undo/redo mixin for VMWorkspaceGraphs."""

from __future__ import annotations

from typing import Dict, List


class GraphUndoMixin:
    """Snapshot-based undo/redo history and batching for VMWorkspaceGraphs."""

    def _init_undo_state(self, max_undo_depth: int = 50) -> None:
        """Initialize undo/redo stacks and batching counters."""
        self._undo_stack: List[Dict[int, dict]] = []
        self._redo_stack: List[Dict[int, dict]] = []
        self._max_undo_depth = max_undo_depth
        # begin/end_undo_batch() collapse several graph calls from one user
        # action into a single undo step; depth-counted so nested batches compose.
        self._undo_batch_depth = 0
        self._undo_batch_pending = False

    def begin_undo_batch(self) -> None:
        """Mark the start of a sequence of mutations representing ONE user action.

        Call ``end_undo_batch()`` (via try/finally) when done. Only the first
        actual mutation inside the outermost begin/end pair records an undo
        snapshot; further mutations before the matching end collapse into that
        same step.
        """
        self._undo_batch_depth += 1
        if self._undo_batch_depth == 1:
            self._undo_batch_pending = True

    def end_undo_batch(self) -> None:
        """Close the innermost undo batch scope."""
        self._undo_batch_depth = max(0, self._undo_batch_depth - 1)
        if self._undo_batch_depth == 0:
            self._undo_batch_pending = False

    def _record_undo_point_if_needed(self) -> None:
        """Capture an undo snapshot before a graph-set mutation when required."""
        if self._undo_batch_depth == 0:
            self._push_undo_snapshot()
        elif self._undo_batch_pending:
            self._push_undo_snapshot()
            self._undo_batch_pending = False

    def _capture_snapshot(self) -> Dict[int, dict]:
        """Deep-copied snapshot of every graph's current saved state."""
        return {graph_id: graph.save() for graph_id, graph in self.graphs.items()}

    def _push_undo_snapshot(self) -> None:
        self._undo_stack.append(self._capture_snapshot())
        if len(self._undo_stack) > self._max_undo_depth:
            self._undo_stack.pop(0)
        self._redo_stack.clear()  # a new action invalidates prior redo history
        self.undo_state_changed.emit()

    def _restore_snapshot(self, snapshot: Dict[int, dict]) -> None:
        """Replace self.graphs wholesale with ``snapshot``'s contents."""
        self._populate_graphs_from_dict(snapshot)
        self._emit_graphs_list()

    def _reset_undo_history(self) -> None:
        """Clear undo/redo history when the workspace graph set is replaced wholesale."""
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._undo_batch_depth = 0
        self._undo_batch_pending = False
        self.undo_state_changed.emit()

    @property
    def can_undo(self) -> bool:
        """Return True if at least one undo snapshot is available."""
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        """Return True if at least one redo snapshot is available."""
        return bool(self._redo_stack)

    def undo(self) -> bool:
        """Revert to the workspace state before the last undo-tracked action."""
        if not self._undo_stack:
            return False
        self._redo_stack.append(self._capture_snapshot())
        snapshot = self._undo_stack.pop()
        self._restore_snapshot(snapshot)
        self.undo_state_changed.emit()
        return True

    def redo(self) -> bool:
        """Re-apply the last undone action."""
        if not self._redo_stack:
            return False
        self._undo_stack.append(self._capture_snapshot())
        snapshot = self._redo_stack.pop()
        self._restore_snapshot(snapshot)
        self.undo_state_changed.emit()
        return True
