# AlphaNumeric Character Recognition - Multi-Model Comparison

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org)

A comprehensive machine learning project comparing multiple deep learning and machine learning models for **alphanumeric character recognition** (0-9, A-Z, a-z and special characters). This project includes implementations of CNN, KNN with HOG features, SVM, Vision Transformers, and other state-of-the-art architectures.

## 🎯 Project Overview

This project demonstrates a thorough comparison of different machine learning approaches to solve the same problem: recognizing handwritten and printed alphanumeric characters. Each model includes:
- Training and validation pipelines
- Detailed performance metrics (accuracy, precision, recall, F1-score)
- Image prediction utilities with web interfaces
- Confusion matrix analysis
- Model comparison and benchmarking

## 📊 Models Included

| Model | Architecture | Framework | Best Accuracy | Status |
|-------|-------------|-----------|----------------|--------|
| **LeNet-5** | CNN (Classic) | TensorFlow/Keras | 85-88% | ✅ Complete |
| **KNN + HOG** | K-Nearest Neighbors | Scikit-learn | 80-85% | ✅ Complete |
| **SVM + HOG** | Support Vector Machine | Scikit-learn | 82-87% | ✅ Complete |
| **MobileNetV2** | Lightweight CNN | TensorFlow/Keras | 88-92% | ✅ Complete |
| **ResNet-18** | Deep CNN | PyTorch | 90-95% | ✅ Complete |
| **Vision Transformer** | Transformer-based | TensorFlow/Keras | 92-97% | ✅ Complete |

## 📁 Project Structure

```
AIML project/
├── README.md                          # This file
├── MODELS.md                          # Detailed model documentation
├── SETUP.md                           # Installation and setup guide
├── RESULTS.md                         # Experimental results and comparisons
│
├── AlphaNum/                          # Dataset directory (48x48 images)
│   ├── train/                         # Training images (0-9, A-Z, a-z, special chars)
│   ├── validation/                    # Validation images
│   └── test/                          # Test images
│
├── AlphaNum2/, AlphaNum3/             # Alternative dataset versions
│
├── LeNet-5/                           # LeNet-5 CNN Model
│   ├── LeNet-5.ipynb                 # Training notebook
│   ├── LeNet-5_Image_Predictor.ipynb # Prediction notebook
│   ├── lenet5_tuned_best.h5          # Saved model weights
│   └── IMPROVED_MODEL_GUIDE.md       # Model tuning guide
│
├── KNN hog/                           # KNN with HOG Features
│   ├── KNN with HOG.ipynb            # Training and evaluation
│   └── KNN_HOG_Image_Predictor.ipynb # Image prediction
│
├── SVM/                               # SVM with HOG Features
│   ├── svm hog.ipynb                 # Training and evaluation
│   ├── SVM_HOG_Image_Predictor.ipynb # Image prediction
│   └── generate_hog_features.ipynb   # HOG feature extraction
│
├── moble net/                         # MobileNetV2 Model
│   ├── MobileNetV2.ipynb             # Training notebook
│   ├── MobileNetV2_Image_Predictor.ipynb # Prediction
│   ├── app.py                        # Flask web application
│   ├── requirements.txt              # Dependencies
│   └── templates/                    # HTML templates
│
├── ResNet-18/                         # ResNet-18 Model (PyTorch)
│   ├── ResNet-18.ipynb               # Training notebook
│   ├── ResNet-18_Image_Predictor.ipynb # Prediction
│   ├── app.py                        # Flask web application
│   ├── requirements.txt              # Dependencies
│   ├── run_app.sh                    # Script to run web app
│   └── templates/                    # HTML templates
│
├── ViT/                               # Vision Transformer
│   ├── ViT_AlphaNum_Classification.ipynb # Training
│   ├── ViT_Image_Predictor.ipynb     # Prediction
│   └── ViT_Image_Size_Comparison.ipynb # Resolution analysis
│
├── hog_features/                      # Pre-computed HOG features
│   └── *.npy files                   # Numpy arrays for quick loading
│
├── prediction_results/                # Output directory for predictions
│   └── batch_predictions/
│
└── results/                           # Model comparison results
    ├── KNN/, LeNet-5/, MobileNetV2/  # Per-model results
    ├── ResNet-18/, SVMhog/, ViT/     # Confusion matrices & metrics
    └── misclassification_comparison.csv # Comparison data
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Option 1: Using Jupyter Notebooks (Recommended for Beginners)

1. **Clone or download this repository**
   ```bash
   git clone https://github.com/yourusername/AlphaNum-Character-Recognition.git
   cd AlphaNum-Character-Recognition
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter and explore notebooks**
   ```bash
   jupyter notebook
   ```

4. **Start with any model training notebook:**
   - `LeNet-5/LeNet-5.ipynb` - Simple CNN
   - `KNN hog/KNN with HOG.ipynb` - KNN approach
   - `ResNet-18/ResNet-18.ipynb` - State-of-the-art CNN

### Option 2: Run Web Applications

#### MobileNetV2 Web App
```bash
cd moble\ net/
pip install -r requirements.txt
python app.py
# Open browser: http://localhost:5000
```

#### ResNet-18 Web App
```bash
cd ResNet-18/
pip install -r requirements.txt
python app.py
# Open browser: http://localhost:5000
```

## 📊 Performance Comparison

See [RESULTS.md](RESULTS.md) for detailed performance metrics, confusion matrices, and visual comparisons.

### Quick Summary:
- **Fastest Model**: KNN + HOG (~5ms per prediction)
- **Most Accurate**: Vision Transformer (97% accuracy)
- **Best Balance**: ResNet-18 (95% accuracy, good speed)
- **Lightest**: MobileNetV2 (92% accuracy, smallest footprint)

## 🔧 Configuration

### Image Resolution
Most models support multiple input resolutions:
- **48x48**: Fast, good for real-time applications
- **64x64**: Standard resolution, balanced accuracy
- **128x128**: High accuracy but slower

To change resolution, modify:
```python
IMG_SIZE = 48  # Change this value in preprocessing cells
```

### Model Parameters
Edit hyperparameters in training notebooks:
- Learning rate
- Batch size
- Number of epochs
- Regularization (dropout, L2)

See [MODELS.md](MODELS.md) for detailed parameter explanations.

## 📈 Key Features

✅ **Multiple Model Implementations**
- Classical ML: KNN, SVM
- Deep Learning: LeNet-5, MobileNetV2, ResNet-18, ViT

✅ **Comprehensive Evaluation**
- Accuracy, Precision, Recall, F1-score
- Confusion matrices
- Per-class performance analysis
- Misclassification patterns

✅ **Interactive Prediction**
- Jupyter notebooks for batch predictions
- Web interfaces for single image prediction
- Drawing interfaces for real-time testing

✅ **Dataset Management**
- Multiple dataset versions
- Pre-computed HOG features for fast training
- Flexible image resolutions

✅ **Reproducible Results**
- Clear documentation
- Hyperparameter specifications
- Training logs and metrics

## 🎓 Learning Resources

This project is ideal for:
- Understanding different ML approaches to the same problem
- Comparing classical ML vs deep learning
- Learning model implementation and evaluation
- Getting hands-on experience with TensorFlow and PyTorch

## 📝 Notebooks Guide

### Training Notebooks
- `*/[Model]*.ipynb` - Data loading, model training, evaluation
- Shows entire pipeline from data to model deployment
- Includes visualization of results

### Prediction Notebooks
- `*_Image_Predictor.ipynb` - Load trained model and predict on new images
- Batch processing capabilities
- Performance analysis on test sets

### Analysis Notebooks
- `*_Comparison.ipynb` - Compare models or resolutions
- Detailed performance metrics
- Visual comparisons

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional model architectures
- Data augmentation techniques
- Performance optimization
- Better web interfaces
- Additional languages/character sets

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ✨ Citation

If you use this project in your research or work, please cite:
```
@misc{alphanumrecognition2024,
  title={AlphaNumeric Character Recognition - Multi-Model Comparison},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/AlphaNum-Character-Recognition}}
}
```

## 📞 Contact & Support

- **Issues**: Report bugs and feature requests on GitHub Issues
- **Discussions**: Join discussions for questions and ideas
- **Email**: your.email@example.com

## 🙏 Acknowledgments

- Dataset created and curated for alphanumeric recognition
- Model architectures based on academic research
- Built with TensorFlow, PyTorch, and Scikit-learn communities

---

**Last Updated**: October 2024
**Maintained By**: [Your Name/Team]
