import os
import tempfile
import shutil

import pandas as pd
 
from PySide6.QtWidgets import QMessageBox,  QFileDialog
from PySide6.QtCore import QFileInfo


class FileConverter():
    def __init__(self, ui, settings):
        super().__init__()
        self.ui = ui
        self.settings = settings
        self.selected_file_paths = []
        
        self.ui.btn_browse.clicked.connect(lambda: self.browse_files())
        self.ui.btn_convert.clicked.connect(lambda: self.convert())

        
    def browse_files(self, file_paths=None):
        """
        Browse and select files to convert. Populates listbox with file names.
        """
        # Ensure file_paths is None or a list
        if file_paths is not None and not isinstance(file_paths, list):
            print(f"Warning: Invalid file_paths type: {type(file_paths)}. Expected list or None.")
            return

        self.selected_file_paths.clear()
        self.ui.listbox.clear()

        # Show dialog if no file_paths provided
        if file_paths is None:
            last_dir = self.settings.value("last_directory", "/")
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            file_paths, _ = QFileDialog.getOpenFileNames(
                self.ui.tabWidget,
                "Open 2D Map File(s) to Convert",
                last_dir,
                "Text Files (*.txt)",
                options=options
            )

        # If files were selected
        if file_paths:
            last_dir = QFileInfo(file_paths[0]).absolutePath()
            self.settings.setValue("last_directory", last_dir)
            
            for file_path in file_paths:
                if os.path.isfile(file_path):
                    abs_path = os.path.abspath(file_path)
                    self.selected_file_paths.append(abs_path)
                    self.ui.listbox.addItem(os.path.basename(abs_path))


    def convert(self, output_dir=None):
        """
        Convert selected files and add output file names to listbox.
        """
        if not self.selected_file_paths:
            QMessageBox.warning(self.ui.tabWidget, "No Files", "Please select files first.")
            return

        for input_path in self.selected_file_paths:
            selected_file_name = os.path.basename(input_path)
            if selected_file_name.endswith('_converted.txt'):
                continue

            name, _ = os.path.splitext(selected_file_name)
            if output_dir is None:
                output_dir = os.path.dirname(input_path)
            output_path = os.path.join(output_dir, f"{name}_converted.txt")

            with tempfile.TemporaryDirectory() as dirtemp:
                temp_input_path = os.path.join(dirtemp, selected_file_name)
                shutil.copy(input_path, temp_input_path)

                # Clean tab formatting
                with open(temp_input_path, 'r+', encoding='utf-8') as fid:
                    data = fid.read().replace("\t\t", "\t")
                    fid.seek(0)
                    fid.write(data)
                    fid.truncate()

                try:
                    self.convert_action(temp_input_path, output_path)
                    self.ui.listbox.addItem(os.path.basename(output_path))
                except Exception as e:
                    QMessageBox.critical(self.ui.tabWidget, "Conversion Error", f"Failed to convert {selected_file_name}\n{str(e)}")

    def convert_action(self, input_path, output_path):
        """
        Converts the file by reshaping intensity values into a matrix
        and saving the output as a reformatted text file.
        """
        try:
            dfr = pd.read_csv(input_path, delimiter="\t")
            assert {'#X', '#Y', '#Wave', '#Intensity'}.issubset(dfr.columns)
        except Exception as e:
            raise ValueError(f"Invalid file format: {str(e)}")

        # Count number of points per spectrum
        dfr_spectrum = dfr.groupby(['#Y', '#X']).size().reset_index(name='size')
        wavenumber_range = dfr_spectrum['size'][1]
        total_spectra_number = len(dfr_spectrum)

        dfr_wavenumbers = dfr['#Wave'][:wavenumber_range]
        dfr_intensity = dfr['#Intensity']

        # Build intensity matrix
        intensity_data = {
            str(dfr_wavenumbers.iloc[i]): [
                dfr_intensity[i + j * wavenumber_range] for j in range(total_spectra_number)
            ]
            for i in range(wavenumber_range)
        }

        # Create intensity DataFrame and concat with coordinates
        dfr_intensity_matrix = pd.DataFrame(intensity_data)
        dfr_spectrum = pd.concat([dfr_spectrum[['#Y', '#X']].reset_index(drop=True), dfr_intensity_matrix], axis=1)

        # Cleanup
        dfr_spectrum.rename(columns={'#X': '', '#Y': ''}, inplace=True)

        # Export
        dfr_spectrum.to_csv(output_path, sep='\t', encoding='utf-8', index=False)
