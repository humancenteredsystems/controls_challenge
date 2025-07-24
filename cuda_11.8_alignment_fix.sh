#!/bin/bash
# CUDA 11.8 Alignment Fix for 'pointcloud' conda environment
# This script aligns all CUDA components to version 11.8 to resolve hanging issues

echo "=== CUDA 11.8 Alignment Fix ==="
echo "Current environment: $(conda info --envs | grep '*')"

echo -e "\n1. Remove conflicting CUDA 12.x packages..."
pip uninstall -y nvidia-cuda-cupti-cu12
pip uninstall -y nvidia-cuda-nvrtc-cu12  
pip uninstall -y nvidia-cuda-runtime-cu12

echo -e "\n2. Downgrade conda CUDA to 11.8..."
conda install -y cuda-version=11.8 -c conda-forge
conda install -y cuda-nvrtc=11.8 -c conda-forge

echo -e "\n3. Verify current package versions..."
echo "Conda CUDA packages:"
conda list | grep cuda

echo -e "\nPyTorch CUDA version:"
python -c "import torch; print('PyTorch CUDA:', torch.version.cuda)"

echo -e "\nRemaining pip CUDA packages:"
pip list | grep nvidia-cuda

echo -e "\n4. Test ONNX Runtime provider availability..."
python -c "import onnxruntime as ort; print('Available providers:', ort.get_available_providers())"

echo -e "\n=== Fix completed! ==="
echo "Next step: Test CUDA initialization with timeout"