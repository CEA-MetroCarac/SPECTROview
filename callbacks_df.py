# callbacks_dataframe.py module.
import pandas as pd
import os
from pathlib import Path
import copy
import io
from PySide6.QtWidgets import QFileDialog, QDialog, QTableWidget, \
    QTableWidgetItem, QVBoxLayout, QMessageBox, QListWidgetItem, \
    QApplication, QCheckBox
from PySide6.QtCore import Qt, QSettings, QFileInfo


class CallbacksDf:
    def __init__(self, ui):
        self.ui = ui  # connect to main.py

        QSettings.setDefaultFormat(QSettings.IniFormat)
        self.settings = QSettings("CEA-Leti", "DaProViz")

        self.original_dfs = {}  # store all loaded df in a dict
        self.working_dfs = {}

        self.selected_df = None  # to always save the original df
        self.filtered_df = None  # df with or without filtering
        self.selected_df_name = None
        self.selected_x_column = None
        self.selected_y_column = None
        self.selected_hue_column = None

        self.df_filters = []  # store added filters of each df in a LIST

        self.file_paths = []

        self.last_dir = None

    def action_open_df(self, file_paths=None, original_dfs=None):
        """ Load dataframe(s) to work with """
        # Initialize self.original_dfs if it's None
        if self.original_dfs is None:
            self.original_dfs = {}

        if original_dfs:
            # Load "original_dfs" from a saved workspace
            self.original_dfs = original_dfs
        else:
            if file_paths is None:
                # Initialize the last used directory from QSettings
                last_dir = self.settings.value("last_directory", "/")

                # Open the QFileDialog with the last used directory
                options = QFileDialog.Options()
                options |= QFileDialog.ReadOnly
                file_paths, _ = QFileDialog.getOpenFileNames(
                    self.ui.tabWidget, "Open File(s)", last_dir,
                    "All Files (*)", options=options)

            # Load dataframes from Excel files
            if file_paths:
                # Update the last used directory in QSettings
                last_dir = QFileInfo(file_paths[0]).absolutePath()
                self.settings.setValue("last_directory", last_dir)

                self.file_paths += file_paths
                for file_path in file_paths:
                    file_path = Path(file_path)
                    # Extract fname without spaces and use it as the df_name
                    fname = file_path.stem
                    ext = file_path.suffix.lower()

                    if ext == '.xlsx':
                        excel_file = pd.ExcelFile(file_path)
                        sheet_names = excel_file.sheet_names

                        for sheet_name in sheet_names:
                            # Remove spaces within sheet_names
                            sheet_name_cleaned = sheet_name.replace(" ", "")
                            df_name = f"{fname}_{sheet_name_cleaned}"
                            self.original_dfs[df_name] = pd.read_excel(
                                excel_file, sheet_name=sheet_name)

                    elif ext == '.csv':
                        skip_rows = 0
                        with open(file_path, 'r') as file:
                            for line in file:
                                if line.startswith("#Site"):
                                    break
                                skip_rows += 1

                        # Read CSV files and skip identified rows
                        df = pd.read_csv(file_path, skiprows=range(skip_rows),
                                         header=None)
                        # Remove semicolons at the end of each row
                        df.replace(';$', '', regex=True, inplace=True)
                        # Create a new DataFrame with delimiter ";"
                        new_df = pd.read_csv(
                            io.StringIO(df.to_csv(index=False, header=False)),
                            delimiter=';')
                        # Add a new column labeled "#Wafer" with CSV filename
                        new_df['wafer'] = file_path.name
                        columns_order = ['wafer'] + [col for col in
                                                     new_df.columns if
                                                     col != 'wafer']
                        new_df = new_df[columns_order]

                        self.original_dfs[fname] = new_df

        # Make a deep copy of the original dfs dictionary
        self.working_dfs = copy.deepcopy(self.original_dfs)
        self.update_listbox_dfs()  # to show loaded dfs in listbox

        # select the first loaded df by default
        first_item = self.ui.listbox_dfs.item(0)
        if first_item:
            self.ui.listbox_dfs.setCurrentItem(first_item)
            self.select_df(first_item)

    def concat_dfs(self):
        """To merge dfs of all wafers into one df"""
        merged_df = []
        # Concatenate all dataframes in the original_dfs dictionary
        all_dfs = list(self.original_dfs.values())
        merged_df = pd.concat(all_dfs, ignore_index=True, sort=False)
        merged_df = merged_df.fillna('')  # Fill empty with NaN

        # Add the merged dataframe to the original_dfs dictionary
        self.original_dfs['Merged_df'] = merged_df

        # Update the working_dfs dictionary
        self.working_dfs = copy.deepcopy(self.original_dfs)
        self.update_listbox_dfs()

    def remove_df(self):
        selected_item = self.ui.listbox_dfs.currentItem()
        if selected_item:
            selected_df_name = selected_item.text()
            if selected_df_name in self.original_dfs:
                del self.original_dfs[selected_df_name]
                del self.working_dfs[selected_df_name]
                self.update_listbox_dfs()

    def add_filter(self, ):
        """ add filter expression and apprend it to the filter dictionary """
        filter_expression = self.ui.ent_filter_query.text().strip()
        if filter_expression:
            filter_data = {"expression": filter_expression, "state": False}
            self.df_filters.append(filter_data)

        # Add the filter expression to QListWidget as a checkbox item
        item = QListWidgetItem()
        checkbox = QCheckBox(filter_expression)
        item.setSizeHint(checkbox.sizeHint())
        self.ui.filter_list.addItem(item)
        self.ui.filter_list.setItemWidget(item, checkbox)
        self.ui.ent_filter_query.clear()
        print(self.df_filters)

    def collect_selected_filters(self):
        """Collect selected filters from the UI"""
        selected_filters = []
        for i in range(self.ui.filter_list.count()):
            item = self.ui.filter_list.item(i)
            checkbox = self.ui.filter_list.itemWidget(item)
            filter_expression = checkbox.text()
            state = checkbox.isChecked()
            selected_filters.append(
                {"expression": filter_expression, "state": state})
        return selected_filters

    def apply_selected_filters(self, filters=None):
        if self.selected_df is not None:
            self.selected_df = self.original_dfs[self.selected_df_name]
            if filters:
                self.df_filters = filters
            else:
                selected_filters = self.collect_selected_filters()
                self.df_filters = selected_filters

            for filter_data in self.df_filters:
                filter_expr = filter_data["expression"]
                is_checked = filter_data["state"]

                if is_checked:
                    try:
                        filter_expr = filter_expr.encode('ascii',
                                                         'ignore').decode(
                            'ascii')
                        self.selected_df = self.selected_df.query(filter_expr)
                        self.working_dfs[
                            self.selected_df_name] = self.selected_df
                    except Exception as e:
                        QMessageBox.critical(self.ui, "Error",
                                             f"Filter error: {str(e)}")

    def update_filter_list(self):
        self.ui.filter_list.clear()
        for filter_data in self.df_filters:
            filter_expression = filter_data["expression"]
            item = QListWidgetItem()
            checkbox = QCheckBox(filter_expression)
            item.setSizeHint(checkbox.sizeHint())
            self.ui.filter_list.addItem(item)
            self.ui.filter_list.setItemWidget(item, checkbox)
            checkbox.setChecked(filter_data["state"])

    def remove_selected_filters(self):
        selected_items = [item for item in self.ui.filter_list.selectedItems()]

        for item in selected_items:
            checkbox = self.ui.filter_list.itemWidget(item)
            filter_expression = checkbox.text()

            # Remove filter data from df_filters list
            for filter_data in self.df_filters[:]:
                if filter_data.get("expression") == filter_expression:
                    self.df_filters.remove(filter_data)

            # Remove the item from the QListWidget
            self.ui.filter_list.takeItem(self.ui.filter_list.row(item))

    def update_listbox_dfs(self):
        self.ui.listbox_dfs.clear()  # Clear the listbox
        df_names = list(self.working_dfs.keys())  # insert to listbox
        for df_name in df_names:
            item = QListWidgetItem(df_name)
            self.ui.listbox_dfs.addItem(item)

    def select_df(self, selected_item):
        # Get selected df_name
        selected_df_name = selected_item.text()
        self.selected_df_name = selected_df_name
        # find df in the dfs list
        self.selected_df = self.working_dfs[selected_df_name]
        self.update_comboboxes()

    def update_comboboxes(self):
        selected_df_name = self.selected_df_name
        if selected_df_name in self.working_dfs:
            try:
                selected_df = self.working_dfs[selected_df_name]
                columns = selected_df.columns.tolist()
                # Clear the comboboxes
                self.ui.combo_xaxis.clear()
                self.ui.combo_yaxis.clear()
                self.ui.combo_hue.clear()
                self.ui.combo_hue_2.clear()

                self.ui.combo_xaxis.addItem("Select X values")
                self.ui.combo_yaxis.addItem("Select Y values")
                self.ui.combo_hue.addItem("Select hue values")
                self.ui.combo_hue_2.addItem("Select hue values")

                # Populate the comboboxes with columns
                for column in columns:
                    self.ui.combo_xaxis.addItem(column)
                    self.ui.combo_yaxis.addItem(column)
                    self.ui.combo_hue.addItem(column)
                    self.ui.combo_hue_2.addItem(column)
            except:  # the case of mapping data
                pass

    def set_selected_x_column(self, index):
        self.selected_x_column = self.ui.combo_xaxis.itemText(index)

    def set_selected_y_column(self, index):
        self.selected_y_column = self.ui.combo_yaxis.itemText(index)

    def set_selected_hue_column(self, index):
        self.selected_hue_column = self.ui.combo_hue.itemText(index)

    def save_df(self):
        selected_df_name = self.selected_df_name
        if selected_df_name in self.original_dfs:
            default_filename = f"{selected_df_name}.xlsx"
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            save_path, _ = QFileDialog.getSaveFileName(
                self.ui.tabWidget,
                "Save DataFrame",
                default_filename,
                "Excel Files (*.xlsx);;All Files (*)",
                options=options
            )
            if save_path:
                try:
                    selected_df = self.original_dfs[selected_df_name]
                    selected_df.to_excel(save_path, index=False)
                    QMessageBox.information(
                        self.ui.tabWidget, "Success",
                        f"{selected_df_name} saved to {save_path}")
                except Exception as e:
                    QMessageBox.critical(
                        self.ui.tabWidget, "Error",
                        f"Error saving {selected_df_name}: {str(e)}")

    def save_all_df_handler(self):
        # Switch between 2 save fnc with the Ctrl key
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            self.save_all_df()
        else:
            self.save_all_df_in_one()

    def save_all_df(self):
        # Initialize the last used directory from QSettings
        last_dir = self.settings.value("last_directory", "/")
        save_folder = QFileDialog.getExistingDirectory(
            self.ui.tabWidget, "Select Folder to Save DataFrames", last_dir)
        if save_folder:
            try:
                for df_name, df in self.original_dfs.items():
                    default_filename = f"{df_name}.xlsx"
                    save_path = os.path.join(save_folder, default_filename)
                    df.to_excel(save_path, index=False)
                QMessageBox.information(
                    self.ui.tabWidget, "Success",
                    "All DataFrames saved to the selected folder.")
            except Exception as e:
                QMessageBox.critical(
                    self.ui.tabWidget, "Error",
                    f"Error saving DataFrames: {str(e)}")

    def save_all_df_in_one(self):
        # Initialize the last used directory from QSettings
        last_dir = self.settings.value("last_directory", "/")
        save_path, _ = QFileDialog.getSaveFileName(
            self.ui.tabWidget, "Save all DataFrames in Excel file",
            last_dir, "Excel Files (*.xlsx)")
        if save_path:
            try:
                with pd.ExcelWriter(save_path, engine="xlsxwriter") as writer:
                    for df_name, df in self.original_dfs.items():
                        df.to_excel(writer, sheet_name=df_name, index=False)

                QMessageBox.information(
                    self.ui.tabWidget, "Success",
                    "All dataFrames are saved in one Excel file.")
            except Exception as e:
                QMessageBox.critical(
                    self.ui.tabWidget, "Error",
                    f"Error saving DataFrames: {str(e)}")

    def view_df(self):
        view_df = self.selected_df
        # Create a QDialog to contain the table
        df_viewer = QDialog(self.ui.tabWidget)
        df_viewer.setWindowTitle("DataFrame Viewer")
        # Create a QTableWidget and populate it with data from the DataFrame
        table_widget = QTableWidget(df_viewer)
        table_widget.setColumnCount(view_df.shape[1])
        table_widget.setRowCount(view_df.shape[0])
        table_widget.setHorizontalHeaderLabels(view_df.columns)
        for row in range(view_df.shape[0]):
            for col in range(view_df.shape[1]):
                item = QTableWidgetItem(str(view_df.iat[row, col]))
                table_widget.setItem(row, col, item)
        # Set the resizing mode for the QTableWidget to make it resizable
        table_widget.setSizeAdjustPolicy(QTableWidget.AdjustToContents)
        # Use a QVBoxLayout to arrange the table within a scroll area
        layout = QVBoxLayout(df_viewer)
        layout.addWidget(table_widget)
        df_viewer.exec_()
