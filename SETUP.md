# Installation & Setup Guide

## 📋 System Requirements

- **Python**: 3.8 or higher
- **OS**: Windows, macOS, or Linux
- **RAM**: 8GB minimum (16GB recommended for deep learning models)
- **GPU** (Optional): NVIDIA GPU with CUDA 11.0+ for faster training

## 🔧 Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/AlphaNum-Character-Recognition.git
cd AlphaNum-Character-Recognition
```

### 2. Create Virtual Environment (Recommended)

#### Using venv:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

#### Using conda:
```bash
conda create -n alphanumrecognition python=3.9
conda activate alphanumrecognition
```

### 3. Install Dependencies

#### For All Models:
```bash
pip install -r requirements.txt
```

#### For Specific Models:

**LeNet-5 & MobileNetV2:**
```bash
pip install tensorflow>=2.10.0 numpy pandas scikit-learn matplotlib seaborn jupyter
```

**ResNet-18 (PyTorch):**
```bash
pip install torch torchvision torchaudio
pip install Flask Pillow numpy
```

**KNN & SVM:**
```bash
pip install scikit-learn opencv-python numpy pandas matplotlib jupyter
```

**Vision Transformer:**
```bash
pip install tensorflow keras-cv numpy pandas matplotlib jupyter
```

**For HOG Features (KNN & SVM):**
```bash
pip install scikit-image opencv-python
```

### 4. Verify Installation

Run this to verify everything is working:

```python
python -c "
import tensorflow as tf
import torch
import sklearn
import cv2
print('✓ TensorFlow:', tf.__version__)
print('✓ PyTorch:', torch.__version__)
print('✓ Scikit-learn:', sklearn.__version__)
print('✓ OpenCV:', cv2.__version__)
"
```

## 📦 Complete Requirements File

Create `requirements.txt`:

```
# Deep Learning Frameworks
tensorflow>=2.10.0
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0

# Image Processing
opencv-python>=4.8.0
scikit-image>=0.21.0
Pillow>=10.0.0

# Machine Learning
scikit-learn>=1.3.0
numpy>=1.24.0
scipy>=1.10.0

# Data Processing & Analysis
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Web Framework
Flask>=3.0.0
Werkzeug>=3.0.0

# Jupyter & Notebooks
jupyter>=1.0.0
jupyterlab>=3.6.0
ipython>=8.0.0

# Utilities
joblib>=1.3.0
tqdm>=4.65.0
PyYAML>=6.0

# GPU Support (Optional)
# tensorflow[and-cuda]>=2.10.0
# For PyTorch GPU: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 Launching Applications

### Jupyter Notebook

```bash
jupyter notebook
```

Then open a web browser to `http://localhost:8888`

### Web Applications

#### MobileNetV2 App:
```bash
cd "moble net"
python app.py
# Visit http://localhost:5000
```

#### ResNet-18 App:
```bash
cd ResNet-18
python app.py
# Visit http://localhost:5000
```

Or use the provided script:
```bash
cd ResNet-18
bash run_app.sh
```

## ⚙️ GPU Setup (Optional)

### For TensorFlow with GPU:

```bash
# Install NVIDIA GPU support
pip install tensorflow[and-cuda]

# Or with specific CUDA version
pip install tensorflow tensorflow-gpu==2.10.0
```

### For PyTorch with GPU:

```bash
# Find your CUDA version: nvidia-smi
# Visit https://pytorch.org/get-started/locally/ for your specific setup

# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🐛 Troubleshooting

### Issue: `No module named 'tensorflow'`
**Solution:**
```bash
pip install --upgrade tensorflow
```

### Issue: `CUDA out of memory`
**Solution:** Reduce batch size in training notebooks:
```python
BATCH_SIZE = 16  # Reduce from 32 or 64
```

### Issue: `ImportError: cannot import name 'hog'`
**Solution:**
```bash
pip install scikit-image
```

### Issue: Port 5000 already in use
**Solution:**
```bash
python app.py --port 5001  # Use different port
```

### Issue: Dataset not found
**Solution:** Ensure dataset folders are in the correct location:
```
AlphaNum/
├── train/
├── validation/
└── test/
```

## 📊 Dataset Setup

### Directory Structure:
```
AlphaNum/
├── train/
│   ├── 48/  (digit '0')
│   ├── 49/  (digit '1')
│   ├── ...
│   ├── 65/  (letter 'A')
│   ├── ...
│   └── 122/ (letter 'z')
├── validation/
│   └── [same structure as train]
└── test/
    └── [same structure as train]
```

### ASCII Value Mapping:
- 48-57: Digits 0-9
- 65-90: Letters A-Z
- 97-122: Letters a-z
- 999: Special null class

## 🔄 Data Preprocessing

Pre-computed HOG features are available in `hog_features/` folder:
- `train_hog_features_48x48.npy`
- `validation_hog_features_48x48.npy`
- `test_hog_features_48x48.npy`

These allow fast training without recomputing features.

## ✅ Verification Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All requirements installed
- [ ] Dataset in correct location
- [ ] Can import TensorFlow: `python -c "import tensorflow"`
- [ ] Can import PyTorch: `python -c "import torch"`
- [ ] Jupyter launches: `jupyter notebook`
- [ ] Can access web apps at http://localhost:5000

## 📞 Getting Help

If you encounter issues:
1. Check the Troubleshooting section above
2. Visit [Issues](https://github.com/yourusername/repo/issues)
3. See [MODELS.md](MODELS.md) for model-specific setup
4. Consult official documentation:
   - [TensorFlow](https://www.tensorflow.org/install)
   - [PyTorch](https://pytorch.org/get-started/locally/)
   - [Scikit-learn](https://scikit-learn.org/stable/install.html)

---

**Last Updated**: October 2024
