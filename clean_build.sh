#!/bin/bash

# Configuration
VENV_NAME=".venv_clean"
PROJECT_DIR="$(pwd)"

echo "----------------------------------------------------------------"
echo "Starting Clean Build Process for SPECTROview"
echo "----------------------------------------------------------------"

# 1. Clean up existing build artifacts and venv
echo "Cleaning up previous builds and environment..."
rm -rf build dist
if [ -d "$VENV_NAME" ]; then
    echo "Removing existing virtual environment '$VENV_NAME'..."
    rm -rf "$VENV_NAME"
fi

# 2. Create new virtual environment
echo "Creating new virtual environment..."
python3 -m venv "$VENV_NAME"
if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment."
    exit 1
fi

# 3. Activate virtual environment
source "$VENV_NAME/bin/activate"
echo "Virtual environment activated: $(which python)"

# 4. Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# 5. Install dependencies from pyproject.toml
# This installs the package in editable mode, which pulls in dependencies defined in pyproject.toml
echo "Installing project dependencies..."
pip install .

if [ $? -ne 0 ]; then
    echo "Error: Failed to install project dependencies."
    deactivate
    exit 1
fi

# 6. Install PyInstaller explicitly (not in runtime deps usually)
echo "Installing PyInstaller..."
pip install pyinstaller

# 7. Build with PyInstaller
echo "----------------------------------------------------------------"
echo "Building application with PyInstaller..."
echo "----------------------------------------------------------------"

pyinstaller main.spec --noconfirm --clean

if [ $? -eq 0 ]; then
    echo "----------------------------------------------------------------"
    echo "Build Successful!"
    echo "App bundle is located in: $PROJECT_DIR/dist/"
    echo "----------------------------------------------------------------"
else
    echo "----------------------------------------------------------------"
    echo "Build Failed!"
    echo "----------------------------------------------------------------"
    deactivate
    exit 1
fi

# Deactivate venv
deactivate
echo "Done."
