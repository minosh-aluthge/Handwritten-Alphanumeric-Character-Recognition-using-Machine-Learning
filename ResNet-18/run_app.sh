#!/bin/bash

# This line helps the script find and use the 'conda' command.
# Make sure the path to conda.sh is correct for your system.
# It's usually in ~/anaconda3/ or ~/miniconda3/
source ~/anaconda3/etc/profile.d/conda.sh

echo "Activating Conda environment: rapids-env..."
conda activate rapids-env

echo "Changing to project directory..."
cd "/home/ubuntu/Desktop/AIML project/ResNet-18"

echo "Running Python application: app.py..."
python app.py

echo "Script finished."
