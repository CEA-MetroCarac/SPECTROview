"""Domain-oriented facade over the currently running desktop application.

This is the single live-application boundary used by the embedded MCP server
and by graph commands emitted from the existing AI Chat.  It intentionally
speaks in datasets, spectra, fits, maps and graphs; Qt widget mechanics are
contained inside this adapter and never leak into MCP tool schemas.
"""

from __future__ import annotations

import copy
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from spectroview.ai_agent.agent.commands import CreatePlot, DeletePlots, UpdatePlot
from spectroview.api.io import export_results as export_results_file
from spectroview.application.dispatch import DirectDispatcher, Dispatcher
from spectroview.application.errors import ApplicationAPIError
from spectroview.model.graph_control import GraphValidationError, normalize_graph_patch


WORKSPACES = ("spectra", "maps", "graphs")
DATASET_KINDS = ("spectra", "map", "dataframe")


def _json_value(value: Any) -> Any:
    """Convert NumPy/pandas values and non-finite floats to JSON-safe data."""
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_value(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return [_json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class SpectroviewApplicationAPI:
    """Stable operations over one running SPECTROview ``Main`` controller.

    Parameters
    ----------
    application:
        The desktop controller. It is deliberately accepted structurally so
        tests can provide small fakes instead of constructing pixels/widgets.
    dispatcher:
        Marshals calls onto the controller's owning thread. Production passes
        :class:`QtMainThreadDispatcher`; tests may use ``DirectDispatcher``.
    """

    def __init__(self, application: Any, dispatcher: Optional[Dispatcher] = None) -> None:
        self._application = application
        self._dispatcher: Dispatcher = dispatcher or DirectDispatcher()
        self._ready = True

    # ------------------------------------------------------------------
    # Boundary helpers
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._ready = False
        close = getattr(self._dispatcher, "close", None)
        if close is not None:
            close()

    def _call(self, callback):
        if not self._ready:
            raise ApplicationAPIError(
                "APPLICATION_NOT_READY", "The running application is not available."
            )
        return self._dispatcher.call(callback)

    @property
    def _spectra_vm(self):
        return self._application.v_spectra_workspace.vm

    @property
    def _maps_vm(self):
        return self._application.v_maps_workspace.vm

    @property
    def _graphs_workspace(self):
        return self._application.v_graphs_workspace

    @property
    def _graphs_vm(self):
        return self._graphs_workspace.vm

    @staticmethod
    def _dataset_id(kind: str, name: str) -> str:
        return f"{kind}:{name}"

    def _split_dataset_id(self, dataset_id: str) -> Tuple[str, str]:
        if not dataset_id or ":" not in dataset_id:
            raise ApplicationAPIError(
                "INVALID_DATASET_ID",
                "dataset_id must use '<kind>:<name>', for example 'spectra:sample'.",
            )
        kind, name = dataset_id.split(":", 1)
        if kind not in DATASET_KINDS or not name:
            raise ApplicationAPIError(
                "INVALID_DATASET_ID",
                "Dataset kind must be spectra, map, or dataframe and the name cannot be empty.",
                {"dataset_id": dataset_id},
            )
        return kind, name

    @staticmethod
    def _require_workspace(workspace: str) -> str:
        value = str(workspace).strip().lower()
        if value not in WORKSPACES:
            raise ApplicationAPIError(
                "INVALID_WORKSPACE", "workspace must be spectra, maps, or graphs."
            )
        return value

    def _workspace_vm(self, workspace: str):
        workspace = self._require_workspace(workspace)
        if workspace == "spectra":
            return self._spectra_vm
        if workspace == "maps":
            return self._maps_vm
        raise ApplicationAPIError(
            "UNSUPPORTED_WORKSPACE", "This operation applies only to spectra or maps."
        )

    # ------------------------------------------------------------------
    # Existing AppContext surface (used by the graph/DataFrame MCP tools)
    # ------------------------------------------------------------------

    def load_dataframes(self, file_paths: List[str]) -> List[str]:
        def mutate():
            self._graphs_vm.load_dataframes(file_paths)
            return list(self._graphs_vm.dataframes.keys())
        return self._call(mutate)

    def show_graph(self, graph_id: Optional[Union[str, int]] = None) -> None:
        def call():
            self._application.show()
            self._application.raise_()
            self._application.activateWindow()
            self._application.tabWidget.setCurrentWidget(self._graphs_workspace)
            if graph_id is not None:
                try:
                    gid = int(graph_id)
                    entry = self._graphs_workspace.graph_widgets.get(gid)
                    if entry and len(entry) >= 3 and entry[2]:
                        self._graphs_workspace.mdi_area.setActiveSubWindow(entry[2])
                except Exception:
                    pass
        self._call(call)


    def list_dataframes(self) -> List[str]:
        return self._call(lambda: list(self._graphs_vm.dataframes))

    def active_dataframe_name(self) -> str:
        return self._call(lambda: self._graphs_vm.selected_df_name or "")

    def get_dataframe(self, name: str = "") -> Optional[pd.DataFrame]:
        def read():
            key = name or self._graphs_vm.selected_df_name or ""
            dataframe = self._graphs_vm.get_dataframe(key)
            return dataframe.copy(deep=False) if dataframe is not None else None

        return self._call(read)

    def list_graphs(self) -> Dict[int, Dict[str, Any]]:
        return self._call(
            lambda: {gid: graph.save() for gid, graph in self._graphs_vm.graphs.items()}
        )

    def submit(self, command: Any) -> Any:
        """Execute a validated agent command through this same facade."""
        if isinstance(command, CreatePlot):
            return self.create_graph(command.config)
        if isinstance(command, UpdatePlot):
            return self.update_graph(command.graph_id, command.properties)
        if isinstance(command, DeletePlots):
            return self.delete_graphs(command.delete_all, command.graph_ids)
        raise ApplicationAPIError(
            "UNSUPPORTED_COMMAND", f"Unsupported application command: {type(command).__name__}."
        )

    # ------------------------------------------------------------------
    # Application/workspace state
    # ------------------------------------------------------------------

    def get_active_workspace(self) -> Dict[str, Any]:
        def read():
            tabs = self._application.tabWidget
            index = tabs.currentIndex()
            name = tabs.tabText(index).strip().lower() if index >= 0 else ""
            return {"name": name, "index": index}

        return self._call(read)

    def get_current_selection(self) -> Dict[str, Any]:
        def read():
            workspace = self.get_active_workspace()["name"]
            if workspace == "spectra":
                return {
                    "workspace": workspace,
                    "spectrum_names": list(self._spectra_vm.selected_fnames),
                }
            if workspace == "maps":
                return {
                    "workspace": workspace,
                    "map_name": self._maps_vm.current_map_name,
                    "spectrum_names": list(self._maps_vm.selected_fnames),
                }
            active_graph = self._active_graph_id_unchecked()
            return {
                "workspace": "graphs",
                "dataframe_name": self._graphs_vm.selected_df_name,
                "graph_id": active_graph,
            }

        return self._call(read)

    def get_application_state(self) -> Dict[str, Any]:
        def read():
            active = self.get_active_workspace()["name"]
            runtime = getattr(self._application, "_mcp_runtime", None)
            return {
                "ready": self._ready,
                "active_workspace": active,
                "selection": self.get_current_selection(),
                "counts": {
                    "spectra_datasets": len(self._spectra_vm.store.map_names),
                    "map_datasets": len(self._maps_vm.store.map_names),
                    "dataframes": len(self._graphs_vm.dataframes),
                    "graphs": len(self._graphs_vm.graphs),
                },
                "fitting": {
                    "spectra": bool(self._spectra_vm._is_fitting),
                    "maps": bool(self._maps_vm._is_fitting),
                },
                "mcp": {
                    "running": bool(runtime is not None and runtime.running),
                    "endpoint": runtime.endpoint if runtime is not None else None,
                },
            }

        return self._call(read)

    # ------------------------------------------------------------------
    # Datasets and spectra
    # ------------------------------------------------------------------

    def list_datasets(self, workspace: Optional[str] = None) -> List[Dict[str, Any]]:
        def read():
            target = self._require_workspace(workspace) if workspace else None
            result: List[Dict[str, Any]] = []
            if target in (None, "spectra"):
                for name in self._spectra_vm.store.map_names:
                    md = self._spectra_vm.store.get_map_data(name)
                    result.append({
                        "dataset_id": self._dataset_id("spectra", name),
                        "name": name,
                        "kind": "spectrum" if md.n_spectra == 1 else "spectra",
                        "workspace": "spectra",
                        "spectrum_count": md.n_spectra,
                        "point_count": len(md.x_axis),
                    })
            if target in (None, "maps"):
                for name in self._maps_vm.store.map_names:
                    md = self._maps_vm.store.get_map_data(name)
                    result.append({
                        "dataset_id": self._dataset_id("map", name),
                        "name": name,
                        "kind": "map",
                        "workspace": "maps",
                        "spectrum_count": md.n_spectra,
                        "point_count": len(md.x_axis),
                    })
            if target in (None, "graphs"):
                for name, dataframe in self._graphs_vm.dataframes.items():
                    result.append({
                        "dataset_id": self._dataset_id("dataframe", name),
                        "name": name,
                        "kind": "dataframe",
                        "workspace": "graphs",
                        "row_count": len(dataframe),
                        "columns": [str(c) for c in dataframe.columns],
                    })
            return result

        return self._call(read)

    def get_dataset_info(self, dataset_id: str) -> Dict[str, Any]:
        def read():
            kind, name = self._split_dataset_id(dataset_id)
            if kind == "dataframe":
                dataframe = self._graphs_vm.get_dataframe(name)
                if dataframe is None:
                    self._dataset_not_found(dataset_id)
                return {
                    "dataset_id": dataset_id,
                    "name": name,
                    "kind": kind,
                    "row_count": len(dataframe),
                    "column_count": len(dataframe.columns),
                    "columns": [
                        {"name": str(column), "dtype": str(dataframe[column].dtype)}
                        for column in dataframe.columns
                    ],
                    "source": self._graphs_vm.dataframe_sources.get(name),
                }

            vm = self._spectra_vm if kind == "spectra" else self._maps_vm
            md = vm.store.get_map_data(name)
            if md is None:
                self._dataset_not_found(dataset_id)
            x = md.x_axis
            subtracted = md.is_baseline_subtracted
            subtracted_count = (
                int(np.count_nonzero(subtracted))
                if isinstance(subtracted, np.ndarray)
                else (md.n_spectra if subtracted else 0)
            )
            return _json_value({
                "dataset_id": dataset_id,
                "name": name,
                "kind": kind,
                "spectrum_count": md.n_spectra,
                "point_count": len(x),
                "x_range": [float(np.min(x)), float(np.max(x))] if len(x) else None,
                "processed": md.x is not None or md.Y is not None,
                "crop_range": [md.range_min, md.range_max],
                "normalization_factor": md.intensity_norm_factor,
                "baseline_configured": md.baseline_config is not None,
                "baseline_subtracted_count": subtracted_count,
                "fit_configured": bool(md.fit_model and md.fit_model.get("peak_models")),
                "fit_result_count": int(np.count_nonzero(md.fit_success))
                if md.fit_success is not None else 0,
                "spectrum_names": list(md.fnames[:100]),
                "spectrum_names_truncated": len(md.fnames) > 100,
                "metadata": copy.deepcopy(md.map_metadata),
            })

        return self._call(read)

    @staticmethod
    def _dataset_not_found(dataset_id: str) -> None:
        raise ApplicationAPIError(
            "DATASET_NOT_FOUND", f"Dataset {dataset_id!r} is not loaded."
        )

    def get_spectrum(
        self,
        dataset_id: str,
        spectrum: Optional[Union[int, str]] = None,
        processed: bool = True,
        max_points: int = 1000,
    ) -> Dict[str, Any]:
        def read():
            kind, name = self._split_dataset_id(dataset_id)
            if kind == "dataframe":
                raise ApplicationAPIError(
                    "INVALID_DATASET_KIND", "get_spectrum requires a spectra or map dataset."
                )
            if not 2 <= int(max_points) <= 10000:
                raise ApplicationAPIError(
                    "INVALID_MAX_POINTS", "max_points must be between 2 and 10000."
                )
            vm = self._spectra_vm if kind == "spectra" else self._maps_vm
            md = vm.store.get_map_data(name)
            if md is None:
                self._dataset_not_found(dataset_id)

            if spectrum is None:
                row = 0
            elif isinstance(spectrum, int):
                row = spectrum
            else:
                try:
                    row = md.fnames.index(str(spectrum))
                except ValueError as exc:
                    raise ApplicationAPIError(
                        "SPECTRUM_NOT_FOUND",
                        f"Spectrum {spectrum!r} is not in {dataset_id!r}.",
                    ) from exc
            if row < 0 or row >= md.n_spectra:
                raise ApplicationAPIError(
                    "SPECTRUM_NOT_FOUND",
                    f"Spectrum index {row} is outside 0..{md.n_spectra - 1}.",
                )

            x = md.x_axis if processed else md.x0
            matrix = md.y_matrix if processed else md.Y0
            y = matrix[row]
            total = len(x)
            if total > max_points:
                indices = np.linspace(0, total - 1, int(max_points), dtype=int)
                x_out, y_out = x[indices], y[indices]
            else:
                x_out, y_out = x, y
            return _json_value({
                "dataset_id": dataset_id,
                "spectrum_index": row,
                "spectrum_name": md.fnames[row],
                "processed": bool(processed),
                "total_points": total,
                "returned_points": len(x_out),
                "downsampled": len(x_out) != total,
                "coordinates": md.coords[row].tolist() if len(md.coords) > row else None,
                "x": x_out.tolist(),
                "y": y_out.tolist(),
            })

        return self._call(read)

    def get_current_spectrum(
        self, processed: bool = True, max_points: int = 1000
    ) -> Dict[str, Any]:
        def read():
            workspace = self.get_active_workspace()["name"]
            if workspace == "spectra":
                selected = self._spectra_vm.selected_fnames
                if not selected:
                    raise ApplicationAPIError(
                        "NO_ACTIVE_SPECTRUM", "No spectrum is selected in Spectra."
                    )
                return self.get_spectrum(
                    self._dataset_id("spectra", selected[0]), 0, processed, max_points
                )
            if workspace == "maps":
                map_name = self._maps_vm.current_map_name
                selected = self._maps_vm.selected_fnames
                if not map_name or not selected:
                    raise ApplicationAPIError(
                        "NO_ACTIVE_SPECTRUM", "No map spectrum is selected."
                    )
                return self.get_spectrum(
                    self._dataset_id("map", map_name), selected[0], processed, max_points
                )
            raise ApplicationAPIError(
                "NO_ACTIVE_SPECTRUM", "The active Graphs workspace has no current spectrum."
            )

        return self._call(read)

    # ------------------------------------------------------------------
    # Processing and fitting (reuses ViewModel orchestration/fit engine)
    # ------------------------------------------------------------------

    def _processing_targets(self, workspace: str, apply_all: bool) -> List[str]:
        vm = self._workspace_vm(workspace)
        if workspace == "maps":
            if apply_all:
                return list(vm.store.map_names)
            return [vm.current_map_name] if vm.current_map_name else []
        return list(vm._get_active_spectra() if apply_all else vm._get_selected_spectra())

    def crop_spectrum(
        self, workspace: str, xmin: float, xmax: float, apply_all: bool = False
    ) -> Dict[str, Any]:
        def mutate():
            target_workspace = self._require_workspace(workspace)
            if target_workspace == "graphs":
                raise ApplicationAPIError(
                    "UNSUPPORTED_WORKSPACE", "Cropping applies only to spectra or maps."
                )
            if not math.isfinite(xmin) or not math.isfinite(xmax) or xmin == xmax:
                raise ApplicationAPIError(
                    "INVALID_RANGE", "xmin and xmax must be finite, different numbers."
                )
            targets = self._processing_targets(target_workspace, apply_all)
            if not targets:
                raise ApplicationAPIError(
                    "NO_ACTIVE_DATASET", f"No target is selected in {target_workspace}."
                )
            vm = self._workspace_vm(target_workspace)
            if vm._is_fitting:
                raise ApplicationAPIError(
                    "FIT_IN_PROGRESS", "Spectral data cannot be cropped while fitting is active."
                )
            vm.apply_spectral_range(float(xmin), float(xmax), bool(apply_all))
            return {"workspace": target_workspace, "targets": targets,
                    "range": [min(xmin, xmax), max(xmin, xmax)]}

        return self._call(mutate)

    def normalize_spectrum(
        self, workspace: str, factor: float, apply_all: bool = False
    ) -> Dict[str, Any]:
        def mutate():
            target_workspace = self._require_workspace(workspace)
            if target_workspace == "graphs":
                raise ApplicationAPIError(
                    "UNSUPPORTED_WORKSPACE", "Normalization applies only to spectra or maps."
                )
            if not math.isfinite(factor) or factor == 0:
                raise ApplicationAPIError(
                    "INVALID_NORMALIZATION", "factor must be a finite non-zero number."
                )
            targets = (
                [self._maps_vm.current_map_name]
                if target_workspace == "maps" and self._maps_vm.current_map_name
                else self._processing_targets(target_workspace, apply_all)
            )
            if not targets:
                raise ApplicationAPIError(
                    "NO_ACTIVE_DATASET", f"No target is selected in {target_workspace}."
                )
            vm = self._workspace_vm(target_workspace)
            if vm._is_fitting:
                raise ApplicationAPIError(
                    "FIT_IN_PROGRESS", "Spectral data cannot be normalized while fitting is active."
                )
            vm.apply_y_normalization(
                float(factor), bool(apply_all)
            )
            return {"workspace": target_workspace, "targets": targets, "factor": factor}

        return self._call(mutate)

    def subtract_baseline(
        self, workspace: str, apply_all: bool = False
    ) -> Dict[str, Any]:
        def mutate():
            target_workspace = self._require_workspace(workspace)
            if target_workspace == "graphs":
                raise ApplicationAPIError(
                    "UNSUPPORTED_WORKSPACE", "Baseline correction applies only to spectra or maps."
                )
            targets = self._processing_targets(target_workspace, apply_all)
            if not targets:
                raise ApplicationAPIError(
                    "NO_ACTIVE_DATASET", f"No target is selected in {target_workspace}."
                )
            vm = self._workspace_vm(target_workspace)
            if vm._is_fitting:
                raise ApplicationAPIError(
                    "FIT_IN_PROGRESS", "A baseline cannot be subtracted while fitting is active."
                )
            missing = [name for name in targets
                       if (vm.store.get_map_data(name) is None
                           or vm.store.get_map_data(name).baseline_config is None)]
            if missing:
                raise ApplicationAPIError(
                    "BASELINE_NOT_CONFIGURED",
                    "A baseline must be configured before it can be subtracted.",
                    {"datasets": missing},
                )
            vm.subtract_baseline(bool(apply_all))
            return {"workspace": target_workspace, "targets": targets}

        return self._call(mutate)

    def get_fit_configuration(self, dataset_id: str = "") -> Dict[str, Any]:
        def read():
            resolved = dataset_id or self._current_fit_dataset_id()
            kind, name = self._split_dataset_id(resolved)
            if kind not in ("spectra", "map"):
                raise ApplicationAPIError(
                    "INVALID_DATASET_KIND", "Fit configuration requires spectra or map data."
                )
            vm = self._spectra_vm if kind == "spectra" else self._maps_vm
            md = vm.store.get_map_data(name)
            if md is None:
                self._dataset_not_found(resolved)
            if not md.fit_model:
                raise ApplicationAPIError(
                    "FIT_NOT_CONFIGURED", f"No fit model is configured for {resolved!r}."
                )
            return _json_value({
                "dataset_id": resolved,
                "fitting": bool(vm._is_fitting),
                "configuration": copy.deepcopy(md.fit_model),
            })

        return self._call(read)

    def _current_fit_dataset_id(self) -> str:
        workspace = self.get_active_workspace()["name"]
        if workspace == "spectra" and self._spectra_vm.selected_fnames:
            return self._dataset_id("spectra", self._spectra_vm.selected_fnames[0])
        if workspace == "maps" and self._maps_vm.current_map_name:
            return self._dataset_id("map", self._maps_vm.current_map_name)
        raise ApplicationAPIError(
            "NO_ACTIVE_DATASET", "No spectra or map dataset is active."
        )

    def fit_spectrum(self, workspace: str, apply_all: bool = False) -> Dict[str, Any]:
        def mutate():
            target_workspace = self._require_workspace(workspace)
            if target_workspace == "graphs":
                raise ApplicationAPIError(
                    "UNSUPPORTED_WORKSPACE", "Fitting applies only to spectra or maps."
                )
            vm = self._workspace_vm(target_workspace)
            if vm._is_fitting:
                raise ApplicationAPIError(
                    "FIT_IN_PROGRESS", f"A fit is already running in {target_workspace}."
                )
            targets = self._processing_targets(target_workspace, apply_all)
            if not targets:
                raise ApplicationAPIError(
                    "NO_ACTIVE_DATASET", f"No target is selected in {target_workspace}."
                )
            configured = [name for name in targets
                          if (vm.store.get_map_data(name) is not None
                              and vm.store.get_map_data(name).fit_model)]
            if not configured:
                raise ApplicationAPIError(
                    "FIT_NOT_CONFIGURED", "No target dataset has a configured fit model."
                )
            vm.fit(bool(apply_all))
            if not vm._is_fitting:
                raise ApplicationAPIError(
                    "FIT_START_FAILED", "The fitting engine did not start."
                )
            return {"workspace": target_workspace, "status": "started",
                    "targets": configured}

        return self._call(mutate)

    def get_fit_results(
        self, workspace: str, collect: bool = False, limit: int = 100
    ) -> Dict[str, Any]:
        def read():
            target_workspace = self._require_workspace(workspace)
            if target_workspace == "graphs":
                raise ApplicationAPIError(
                    "UNSUPPORTED_WORKSPACE", "Fit results apply only to spectra or maps."
                )
            if not 1 <= int(limit) <= 1000:
                raise ApplicationAPIError("INVALID_LIMIT", "limit must be between 1 and 1000.")
            vm = self._workspace_vm(target_workspace)
            if collect and vm._is_fitting:
                raise ApplicationAPIError(
                    "FIT_IN_PROGRESS", "Wait for fitting to finish before collecting results."
                )
            if collect:
                vm.collect_fit_results()
            dataframe = vm.df_fit_results
            if dataframe is None or dataframe.empty:
                raise ApplicationAPIError(
                    "FIT_RESULTS_NOT_AVAILABLE", "No collected fit results are available."
                )
            preview = dataframe.head(int(limit))
            return _json_value({
                "workspace": target_workspace,
                "total_rows": len(dataframe),
                "returned_rows": len(preview),
                "columns": [str(c) for c in dataframe.columns],
                "rows": preview.to_dict(orient="records"),
            })

        return self._call(read)

    # ------------------------------------------------------------------
    # Graphs and maps
    # ------------------------------------------------------------------

    def _active_graph_id_unchecked(self) -> Optional[int]:
        getter = getattr(self._graphs_workspace, "_get_active_graph_id", None)
        return getter() if getter is not None else None

    def list_graph_configurations(self) -> List[Dict[str, Any]]:
        def read():
            active = self._active_graph_id_unchecked()
            return [_json_value({"graph_id": gid, "active": gid == active, **graph.save()})
                    for gid, graph in sorted(self._graphs_vm.graphs.items())]

        return self._call(read)

    def get_active_graph(self) -> Dict[str, Any]:
        def read():
            graph_id = self._active_graph_id_unchecked()
            if graph_id is None:
                raise ApplicationAPIError("NO_ACTIVE_GRAPH", "No graph is active.")
            graph = self._graphs_vm.get_graph(graph_id)
            if graph is None:
                raise ApplicationAPIError("NO_ACTIVE_GRAPH", "The active graph no longer exists.")
            return _json_value({"graph_id": graph_id, **graph.save()})

        return self._call(read)

    def create_graph(self, configuration: Dict[str, Any]) -> Dict[str, Any]:
        def mutate():
            config = copy.deepcopy(configuration)
            dataframe_name = config.get("df_name") or self._graphs_vm.selected_df_name
            if not dataframe_name or self._graphs_vm.get_dataframe(dataframe_name) is None:
                raise ApplicationAPIError(
                    "DATAFRAME_NOT_FOUND", f"DataFrame {dataframe_name!r} is not loaded."
                )
            config["df_name"] = dataframe_name
            try:
                config = normalize_graph_patch(
                    config, dataframe=self._graphs_vm.get_dataframe(dataframe_name)
                )
            except GraphValidationError as exc:
                raise ApplicationAPIError("INVALID_GRAPH_CONFIGURATION", str(exc)) from exc
            before = set(self._graphs_vm.graphs)
            success = self._graphs_workspace.create_plot_from_config(dataframe_name, config)
            created = sorted(set(self._graphs_vm.graphs) - before)
            if not success or not created:
                raise ApplicationAPIError("GRAPH_CREATE_FAILED", "The graph could not be created.")
            gid = created[-1]
            self._application.tabWidget.setCurrentWidget(self._graphs_workspace)
            image_path = None
            entry = self._graphs_workspace.graph_widgets.get(gid)
            if entry and entry[0]:
                widget = entry[0]
                fig = getattr(widget, "figure", None)
                if fig is None and hasattr(widget, "plot_widget"):
                    fig = getattr(widget.plot_widget, "figure", None)
                if fig is not None:
                    import tempfile, time
                    from pathlib import Path
                    temp_dir = Path(tempfile.gettempdir()) / "spectroview_plots"
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    dest = temp_dir / f"spectroview_graph_{gid}_{int(time.time())}.png"
                    try:
                        canvas = getattr(widget, "canvas", None)
                        if canvas is not None and hasattr(canvas, "draw"):
                            try:
                                canvas.draw()
                            except Exception:
                                pass
                        fig.savefig(str(dest), format="png", dpi=150, bbox_inches="tight")
                        if dest.is_file() and dest.stat().st_size > 0:
                            image_path = str(dest)
                    except Exception:
                        pass
            return {"graph_id": gid, "image_path": image_path, "configuration": config}

        return self._call(mutate)

    def update_graph(self, graph_id: Union[str, int], properties: Dict[str, Any]) -> Dict[str, Any]:
        def mutate():
            if not properties:
                raise ApplicationAPIError(
                    "INVALID_GRAPH_CONFIGURATION", "At least one graph property is required."
                )
            updated = self._graphs_workspace.update_graphs_from_config(graph_id, properties)
            if not updated:
                raise ApplicationAPIError(
                    "GRAPH_UPDATE_FAILED", "No matching graph was updated."
                )
            self._application.tabWidget.setCurrentWidget(self._graphs_workspace)
            return {"graph_ids": updated, "properties": copy.deepcopy(properties)}

        return self._call(mutate)

    def delete_graphs(
        self, delete_all: bool = False, graph_ids: Optional[Iterable[int]] = None
    ) -> Dict[str, Any]:
        def mutate():
            requested = [int(gid) for gid in (graph_ids or [])]
            open_ids = list(self._graphs_workspace.graph_widgets)
            if delete_all:
                targets = open_ids if not requested else [gid for gid in open_ids if gid not in requested]
            else:
                targets = [gid for gid in requested if gid in open_ids]
            if not targets:
                raise ApplicationAPIError("NO_ACTIVE_GRAPH", "No matching graph is open.")
            for gid in targets:
                self._graphs_workspace.graph_widgets[gid][2].close()
            return {"deleted_graph_ids": targets}

        return self._call(mutate)

    def get_active_map(self) -> Dict[str, Any]:
        def read():
            name = self._maps_vm.current_map_name
            if not name:
                raise ApplicationAPIError("NO_ACTIVE_MAP", "No map is active.")
            info = self.get_dataset_info(self._dataset_id("map", name))
            info["selected_spectra"] = list(self._maps_vm.selected_fnames)
            info["map_type"] = self._maps_vm.map_type
            return info

        return self._call(read)

    # ------------------------------------------------------------------
    # Explicit filesystem write
    # ------------------------------------------------------------------

    def export_results(
        self, workspace: str, output_path: str, overwrite: bool = False
    ) -> Dict[str, Any]:
        def write():
            target_workspace = self._require_workspace(workspace)
            if target_workspace == "graphs":
                raise ApplicationAPIError(
                    "UNSUPPORTED_WORKSPACE", "Only spectra or maps fit results can be exported."
                )
            vm = self._workspace_vm(target_workspace)
            dataframe = vm.df_fit_results
            if dataframe is None or dataframe.empty:
                raise ApplicationAPIError(
                    "FIT_RESULTS_NOT_AVAILABLE", "Collect fit results before exporting them."
                )
            path = Path(output_path).expanduser().resolve()
            if path.suffix.lower() not in (".csv", ".xlsx", ".xls"):
                raise ApplicationAPIError(
                    "INVALID_EXPORT_PATH", "Export path must end in .csv, .xlsx, or .xls."
                )
            if not path.parent.exists():
                raise ApplicationAPIError(
                    "INVALID_EXPORT_PATH", f"Parent folder does not exist: {path.parent}"
                )
            if path.exists() and not overwrite:
                raise ApplicationAPIError(
                    "OVERWRITE_REQUIRED",
                    f"The destination already exists: {path}. Set overwrite=true explicitly.",
                )
            try:
                exported = export_results_file(dataframe, path)
            except Exception as exc:
                raise ApplicationAPIError("EXPORT_FAILED", str(exc)) from exc
            return {"path": str(exported), "rows": len(dataframe),
                    "bytes": exported.stat().st_size}

        return self._call(write)
