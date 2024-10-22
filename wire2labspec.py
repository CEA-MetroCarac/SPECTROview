import os
import platform
import sys
import tkinter as tk
from tkinter import filedialog
import shutil
import tempfile
from PIL import Image, ImageTk
import pyperclip
import pandas as pd


class Wire2LabspecGUI:
    """
    TODO
    """

    def __init__(self, master):
        """
        TODO

        Parameters
        ----------
        master
        """
        self.selected_file_paths = []

        self.master = master
        self.master.title("Labspec6 Converter (v2)")
        self.master.geometry("400x320")
        self.master.resizable(width=True, height=True)
        if platform.system() == 'Windows':
            # .ico format not supported on Linux
            self.master.iconbitmap(resource_path("icon.ico"))

        # Create Frame for logo & text
        frame1 = tk.Frame(self.master)
        frame1.pack(side=tk.TOP, padx=10, pady=10)

        # create left-side frame inside the frame1 for logo
        left_frame = tk.Frame(frame1)
        left_frame.pack(side=tk.LEFT)

        logo = Image.open(resource_path("logo.png"))
        logo = ImageTk.PhotoImage(logo)
        logo_label = tk.Label(left_frame, image=logo)
        logo_label.image = logo
        logo_label.pack()

        # create right-side frame inside the frame1 for description text
        right_frame = tk.Frame(frame1)
        right_frame.pack(side=tk.RIGHT)

        description = tk.Label(right_frame,
                               text="This tool is used to convert "
                                    "hyperspectral data recorded by Raman "
                                    "InVia to the Labspec6 format (HORIBA)",
                               wraplengt=250)
        description.pack()

        # Create Frame for buttons
        button_frame = tk.Frame(self.master)
        button_frame.pack(side=tk.TOP)
        self.browse_button = tk.Button(button_frame, text="Browse",
                                       command=self.browse_files)
        self.browse_button.pack(side=tk.LEFT, padx=20)
        self.convert_button = tk.Button(button_frame, text="Convert",
                                        command=self.convert_files)
        self.convert_button.pack(side=tk.LEFT, padx=20)

        # Create a LabelFrame for the Listbox
        listbox_frame = tk.LabelFrame(self.master, text="File list:")
        listbox_frame.pack(fill="both", expand="yes", padx=15, pady=5)

        self.listbox = tk.Listbox(listbox_frame)
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create the menu bar
        self.menu_bar = tk.Menu(self.master)
        self.master.config(menu=self.menu_bar)

        # Create the "Help" menu
        help_menu = tk.Menu(self.menu_bar, tearoff=False)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Change log", command=self.show_changelog)

    def show_about(self):
        """ Display version and Developers information """
        # Create the about dialog
        about_dialog = tk.Toplevel(self.master)
        about_dialog.title("About")
        about_dialog.geometry("200x220")
        about_dialog.resizable(width=False, height=False)

        # Create the text label
        about_text = "Wire2Labspec\nVersion 2\n\nCreated by: \n" \
                     " Van-Hoan Le \n(van-hoan.le@cea.fr)\n\n " \
                     "Patrick Quemere \n(patrick.quemere@cea.fr)"

        about_label = tk.Label(about_dialog, text=about_text)
        about_label.pack(padx=10, pady=10)

        # Create the OK button
        ok_button = tk.Button(about_dialog, text="OK",
                              command=about_dialog.destroy)
        ok_button.pack(pady=5)

    def show_changelog(self):
        """ Show changes, modifications in new versions... """
        # Create the about dialog
        about_dialog = tk.Toplevel(self.master)
        about_dialog.title("About")
        about_dialog.geometry("450x150")
        about_dialog.resizable(width=False, height=False)

        # Create the text label
        about_text = " Version v2\n\n- GUI windows are now resizable, " \
                     "long file names are thus more readable. \n- Multiple " \
                     "files conversion\n- Automatically copy the working " \
                     "folder to clipboard"
        about_label = tk.Label(about_dialog, text=about_text)
        about_label.pack(padx=10, pady=10)

        # Create the OK button
        ok_button = tk.Button(about_dialog, text="Okay",
                              command=about_dialog.destroy)
        ok_button.pack(pady=5)

    def browse_files(self, file_paths=None):
        """
        The function of the "Browse" button:
        - Select one or mutiples files to convert
        - The absolute path of the selected file is also collected and thus,
        the name of all other text files in the working folder is collected
        and showed in the listbox.

        Parameters
        ----------
        file_paths: list of str, optional
            List of files paths to work with
        """
        self.selected_file_paths.clear()  # Clear the selected file list
        self.listbox.delete(0, tk.END)  # Clear listbox

        if file_paths is None:
            file_paths = filedialog.askopenfilenames()

        for file_path in file_paths:
            # Collect file_path and file_name of all selected files
            selected_file_path = os.path.abspath(file_path)
            selected_file_name = os.path.basename(file_path)

            self.selected_file_paths.append(selected_file_path)  # add to list
            self.listbox.insert(tk.END,
                                selected_file_name)  # show filenames in listbox

    def convert_files(self, output_dir=None):
        """
        Convert Wire formatted files to Labspec formatted files

        Parameters
        ----------
        output_dir: str, optional
            Output dirname to save the converted files
        """
        for selected_file_path in self.selected_file_paths:
            selected_file_name = os.path.basename(selected_file_path)
            if not selected_file_name.endswith('_converted.txt'):
                input_path = selected_file_path
                if output_dir is None:
                    output_dir = os.path.dirname(input_path)
                name = os.path.splitext(selected_file_name)[0]
                output_path = os.path.join(output_dir, name + '_converted.txt')

                with tempfile.TemporaryDirectory() as dirtemp:
                    temp_input_path = os.path.join(dirtemp, selected_file_name)

                    # Create a copy (tempory file) of the input file
                    shutil.copy(input_path, temp_input_path)

                    # replace double tabs with single tabs
                    with open(temp_input_path, 'r+') as fid:
                        input_file_data = fid.read().replace("\t\t", "\t")
                        # overwrite the modified data to the tempory file
                        # move the file pointer to the beginning of the file
                        fid.seek(0)
                        # overwrite the file with the modified data
                        fid.write(input_file_data)
                        # truncate the remaining data in case the new data is
                        # shorter than the old data
                        fid.truncate()

                    # Call the labspec_converter function to convert the
                    # modified input_file
                    self.labspec_converter(temp_input_path, output_path)

                # Add the converted file to the listbox
                self.listbox.insert(tk.END, os.path.basename(output_path))

                # copy working folder dir to clipboard
                pyperclip.copy(output_dir)

    def labspec_converter(self, input_path, output_path):
        """
        Read the original hyperspectral data (.txt file, format WIRE of
        Renishaw), then processed it as a data frame in order to
        convert/re-arrange to the Labspec6 format.

        Parameters
        ----------
        input_path: str
            Absolute path of the file to be converted
        output_path: str
            Absolute path of the output file after converting
        """
        # Read hyperspectral data
        dfr = pd.read_csv(input_path, delimiter="\t", header=0)

        # Switch the column X and Y
        # Count the number of the wavelength values within a spectrum
        dfr_spectrum = dfr.groupby(['#Y', '#X'], axis=0, as_index=False).size()
        wavenumber_range = dfr_spectrum['size'][1]

        # Get all wavenumber values from column #Wave
        # (i.e., from 0 to the [1] first value of the "size" column of
        # dfr_spectrum)
        dfr_wavenumbers = dfr['#Wave'][0:dfr_spectrum['size'][1]]

        #  Get associated Intensity values
        dfr_intensity = dfr['#Intensity']
        # print(dfr_intensity)

        for wavenumber in dfr_wavenumbers:
            dfr_spectrum[
                wavenumber] = "0"  # Create columns and filled all with 0
        # total spectra within the map
        total_spectra_number = dfr_spectrum.shape[0]

        for i in range(0, wavenumber_range):
            # Creat an empty list for each wavenumber column
            intensity_at_each_wavenumber = []

            for j in range(0, total_spectra_number):
                position_ij = i + j * wavenumber_range
                intensity_at_each_wavenumber.append(dfr_intensity[position_ij])

            # Attribute the values
            dfr_spectrum[dfr_wavenumbers[i]] = intensity_at_each_wavenumber

        # Delete the "size" column. Number 1 means delete only this column
        dfr_spectrum = dfr_spectrum.drop('size', axis=1)
        dfr_spectrum.rename(columns={'#X': '', '#Y': ''}, inplace=True)

        # Output
        dfr_spectrum.to_csv(output_path, sep='\t', encoding='utf-8',
                            header=True, index=False)


def resource_path(relative_path):
    """
    TODO

    Parameters
    ----------
    relative_path

    Returns
    -------

    """
    base_path = getattr(sys, '_MEIPASS',
                        os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


def launcher():
    """ Launch the application """
    # Create the Tkinter window
    master = tk.Tk()

    Wire2LabspecGUI(master)
    # Start the Tkinter event loop
    master.mainloop()


if __name__ == '__main__':
    launcher()
