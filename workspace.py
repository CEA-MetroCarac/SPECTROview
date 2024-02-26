# workspace.py
import os
import json
import pandas as pd
from PySide6.QtWidgets import QFileDialog, QMessageBox
from PySide6.QtCore import QSettings, QFileInfo


class SaveLoadWorkspace:
    def __init__(self, ui, callbacks_df, callbacks_plot):
        self.ui = ui
        self.callbacks_df = callbacks_df
        self.callbacks_plot = callbacks_plot

    def save_workspace(self, fname=None):
        if fname is None:
            # Initialize the last used directory from QSettings
            last_dir = self.callbacks_df.settings.value("last_directory", "/")

            default_filename = "ICAR_number.json"
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            fname, _ = QFileDialog.getSaveFileName(self.ui.tabWidget,
                                                   "Save Workspace",
                                                   os.path.join(last_dir,
                                                                default_filename),
                                                   # Correct this line
                                                   "Workspace file ("
                                                   "*.json);;All Files (*)",
                                                   options=options)
        if fname:
            dataframes_json = {key: df.to_json(orient='split') for key, df in
                               self.callbacks_df.original_dfs.items()}

            # Create a dictionary to store workspace data
            # Convert plot_specs to JSON-compatible dictionary
            plot_specs_json = {}
            for plot_id, spec in self.callbacks_plot.plot_specs.items():
                spec_copy = spec.copy()  # Create a copy to avoid modifying
                # the original
                spec_copy['associated_df'] = spec_copy['associated_df'].to_dict(
                    orient='split')
                plot_specs_json[str(plot_id)] = spec_copy

            workspace = {
                "df_filepaths": self.callbacks_df.file_paths,
                "df_filters": self.callbacks_df.df_filters,
                "original_dfs": dataframes_json,
                "plot_specs": plot_specs_json,
            }

            with open(fname, "w") as file:
                json.dump(workspace, file)

    def load_workspace(self, fname=None):
        if fname is None:
            # Initialize the last used directory from QSettings
            last_dir = self.callbacks_df.settings.value("last_directory", "/")

            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            fname, _ = QFileDialog.getOpenFileName(self.ui.tabWidget,
                                                   "Open Workspace", last_dir,
                                                   "Workspace Files ("
                                                   "*.json);;All Files (*)",
                                                   options=options)
            if not fname:
                return
        self.callbacks_plot.clear_workspace()  # CLEAR WORKSPACES

        if fname:
            # Update the last used directory in QSettings
            last_dir = QFileInfo(fname).absolutePath()
            self.callbacks_df.settings.setValue("last_directory", last_dir)

            with open(fname, "r") as file:
                wsp = json.load(file)

            # self.callbacks_df.action_open_df(file_paths=wsp["df_filepaths"],
            #                                  original_dfs=None)
            original_dfs = {key: pd.read_json(value, orient='split') for
                            key, value in wsp["original_dfs"].items()}
            self.callbacks_df.action_open_df(file_paths=None,
                                             original_dfs=original_dfs)

            self.callbacks_df.df_filters = wsp["df_filters"]
            self.callbacks_df.update_filter_list()

            # Convert plot_specs back from dictionaries
            plot_specs_json = wsp["plot_specs"]
            plot_specs = {}
            for plot_id, spec in plot_specs_json.items():
                # Create a copy to avoid modifying the original
                spec_copy = spec.copy()
                spec_copy['associated_df'] = pd.DataFrame(
                    spec_copy['associated_df']['data'],
                    columns=spec_copy['associated_df']['columns'])
                plot_specs[int(plot_id)] = spec_copy

            self.callbacks_plot.plot_specs = self.reset_plot_ids(plot_specs)
            self.callbacks_plot.reload_workspace()

    def load_recipe(self, fname=None):
        if fname is None:
            # Initialize the last used directory from QSettings
            last_dir = self.callbacks_df.settings.value("last_directory", "/")

            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            fname, _ = QFileDialog.getOpenFileName(self.ui.tabWidget,
                                                   "Open Workspace", last_dir,
                                                   "Workspace Files ("
                                                   "*.json);;All Files (*)",
                                                   options=options)
            if not fname:
                return

        if fname:
            # Update the last used directory in QSettings
            last_dir = QFileInfo(fname).absolutePath()
            self.callbacks_df.settings.setValue("last_directory", last_dir)

            with open(fname, "r") as file:
                wsp = json.load(file)

            self.callbacks_df.df_filters = wsp["df_filters"]
            self.callbacks_df.update_filter_list()

            # Convert plot_specs back from dictionaries
            plot_specs_json = wsp["plot_specs"]
            plot_specs = {}
            for plot_id, spec in plot_specs_json.items():
                # Create a copy to avoid modifying the original
                spec_copy = spec.copy()
                spec_copy['associated_df'] = pd.DataFrame(
                    spec_copy['associated_df']['data'],
                    columns=spec_copy['associated_df']['columns'])
                plot_specs[int(plot_id)] = spec_copy

            self.callbacks_plot.plot_specs = self.reset_plot_ids(plot_specs)
            self.callbacks_plot.reload_plot_recipe()

    def reset_plot_ids(self, original_plot_specs):
        """ use when remove a plot """
        # Create a new dictionary to store the updated plot_specs
        updated_plot_specs = {}
        new_plot_id = self.callbacks_plot.plot_counter + 1

        # Iterate through the original plot_specs
        for plot_id, spec in original_plot_specs.items():
            # Add the spec to the updated_plot_specs with the new plot_id
            updated_plot_specs[new_plot_id] = spec
            new_plot_id += 1  # Increment the new plot_id

        return updated_plot_specs
