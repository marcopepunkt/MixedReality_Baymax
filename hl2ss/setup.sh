#!/bin/bash

# Name of the conda environment
ENV_NAME="hl2ss"

# Check if the environment already exists
if conda env list | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists."
else
    echo "Creating environment '$ENV_NAME'..."
    conda env create -f environment.yaml
fi

# Activate the environment
echo "Activating environment '$ENV_NAME'..."
conda activate "$ENV_NAME"

# Function to check if a Python package is installed
check_installed() {
    PACKAGE=$1
    python -c "import $PACKAGE" &> /dev/null
    return $?
}

# Install or upgrade openmim and other packages if not already installed
if check_installed openmim; then
    echo "openmim is already installed."
else
    echo "Installing/upgrading openmim..."
    pip install -U openmim
fi

if check_installed mmengine; then
    echo "mmengine is already installed."
else
    echo "Installing mmengine..."
    mim install mmengine
fi

# Check if the version of mmcv is >= 2.0.0
if python -c "import mmcv; assert mmcv.__version__ >= '2.0.0,<2.2.0'" &> /dev/null; then
    echo "mmcv>=2.0.0 is already installed."
else
    echo "Installing/upgrading mmcv to version >= 2.0.0,<2.2.0..."
    mim install "mmcv>=2.0.0,<2.2.0"
fi

if check_installed mmdet; then
    echo "mmdet is already installed."
else 
    echo "Installing/upgrading mmdet..."
    mim install mmdet
fi

echo "Environment setup complete!"
