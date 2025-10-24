# Detailed Model Documentation

## 📚 Overview

This document provides detailed information about each model implementation, including architecture, performance, and usage guidelines.

---

## 1. LeNet-5 CNN

### 🏗️ Architecture
- **Type**: Convolutional Neural Network (Classical)
- **Framework**: TensorFlow/Keras
- **Input Size**: 48x48 grayscale images
- **Layers**:
  - Conv2D (32 filters, 3x3) + ReLU
  - MaxPooling2D (2x2)
  - Conv2D (64 filters, 3x3) + ReLU
  - MaxPooling2D (2x2)
  - Flatten
  - Dense (128 units) + ReLU + Dropout(0.2)
  - Dense (62 units, softmax)

### 📊 Performance
- **Best Accuracy**: 85-88%
- **Training Time**: ~15-30 minutes on CPU
- **Model Size**: ~1 MB
- **Inference Time**: ~5-10ms per image

### 🎯 Strengths
- Fast to train
- Lightweight model
- Good baseline for comparison
- Easy to understand and modify

### ⚠️ Limitations
- Lower accuracy than deeper networks
- May underfit on complex patterns
- Limited capacity

### 📖 Usage
```python
# Load and predict
from tensorflow.keras.models import load_model
model = load_model('lenet5_tuned_best.h5')
predictions = model.predict(preprocessed_images)
```

### 📝 Notebooks
- `LeNet-5/LeNet-5.ipynb` - Training
- `LeNet-5/LeNet-5_Image_Predictor.ipynb` - Prediction
- `LeNet-5/IMPROVED_MODEL_GUIDE.md` - Tuning guide

---

## 2. KNN with HOG Features

### 🏗️ Approach
- **Type**: K-Nearest Neighbors Classifier
- **Framework**: Scikit-learn
- **Feature Extraction**: Histogram of Oriented Gradients (HOG)

### 🎛️ Parameters
- **n_neighbors**: 5
- **HOG Parameters**:
  - orientations: 9
  - pixels_per_cell: (8, 8)
  - cells_per_block: (2, 2)
  - Feature dimension: ~324

### 📊 Performance
- **Best Accuracy**: 80-85%
- **Training Time**: ~5-10 seconds
- **Model Size**: ~50-100 MB (includes training data)
- **Inference Time**: ~5-15ms per image (fast!)

### 🎯 Strengths
- Extremely fast inference
- No training required (lazy learning)
- Good interpretability
- Works well with hand-crafted features

### ⚠️ Limitations
- Stores entire training set in memory
- Does not learn features (uses HOG)
- Accuracy plateau below deep learning models
- Large model size in memory

### 🔧 Configuration
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
# Adjust n_neighbors for accuracy/speed tradeoff
# Larger k = smoother decisions, may underfit
# Smaller k = more flexible, may overfit
```

### 📝 Notebooks
- `KNN hog/KNN with HOG.ipynb` - Training
- `KNN hog/KNN_HOG_Image_Predictor.ipynb` - Prediction

---

## 3. SVM with HOG Features

### 🏗️ Approach
- **Type**: Support Vector Machine Classifier
- **Framework**: Scikit-learn
- **Feature Extraction**: HOG (same as KNN)

### 🎛️ Parameters
- **Kernel**: RBF (Radial Basis Function)
- **C**: 1.0 (regularization parameter)
- **gamma**: 'scale' (kernel coefficient)

### 📊 Performance
- **Best Accuracy**: 82-87%
- **Training Time**: ~2-5 minutes
- **Model Size**: ~5-20 MB
- **Inference Time**: ~2-5ms per image (very fast!)

### 🎯 Strengths
- Better accuracy than KNN
- Smaller model than KNN
- Very fast inference
- Good generalization

### ⚠️ Limitations
- Training is slower than KNN
- Hyperparameter tuning important
- Less interpretable than KNN
- Still uses hand-crafted features

### 🔧 Configuration
```python
from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
# Adjust C for regularization (lower = more regularization)
# Try different kernels: 'linear', 'poly', 'rbf', 'sigmoid'
```

### 📝 Notebooks
- `SVM/svm hog.ipynb` - Training
- `SVM/SVM_HOG_Image_Predictor.ipynb` - Prediction
- `SVM/generate_hog_features.ipynb` - Feature extraction

---

## 4. MobileNetV2

### 🏗️ Architecture
- **Type**: Efficient CNN with Depthwise Separable Convolutions
- **Framework**: TensorFlow/Keras
- **Input Size**: 224x224 (internally resized from 48x48)
- **Base**: Pre-trained on ImageNet, fine-tuned for alphanumeric

### 📊 Performance
- **Best Accuracy**: 88-92%
- **Training Time**: ~30-60 minutes
- **Model Size**: ~9 MB
- **Inference Time**: ~15-20ms per image

### 🎯 Strengths
- Lightweight architecture designed for mobile
- Good accuracy-to-speed ratio
- Pre-trained weights available
- Very efficient inference

### ⚠️ Limitations
- Lower accuracy than ResNet
- May require more careful tuning
- Cannot handle very low-resolution images

### 🔧 Configuration
```python
from tensorflow.keras.applications import MobileNetV2
base_model = MobileNetV2(input_shape=(224, 224, 3), 
                         include_top=False, 
                         weights='imagenet')
# Freeze base layers for transfer learning
base_model.trainable = False
```

### 📝 Notebooks
- `moble net/MobileNetV2.ipynb` - Training
- `moble net/MobileNetV2_Image_Predictor.ipynb` - Prediction

### 🌐 Web App
```bash
cd moble\ net/
python app.py  # http://localhost:5000
```

---

## 5. ResNet-18

### 🏗️ Architecture
- **Type**: Deep Residual Network
- **Framework**: PyTorch
- **Input Size**: 224x224 (resized from 48x48)
- **Layers**: 18 convolutional layers + residual connections

### 📊 Performance
- **Best Accuracy**: 90-95%
- **Training Time**: ~45-90 minutes
- **Model Size**: ~40-50 MB
- **Inference Time**: ~20-30ms per image

### 🎯 Strengths
- Excellent accuracy
- Good balance between accuracy and speed
- Residual connections enable very deep networks
- PyTorch offers flexibility

### ⚠️ Limitations
- Larger model size
- Slower than MobileNetV2
- Requires more GPU memory
- PyTorch learning curve

### 🔧 Configuration
```python
import torch
from torchvision import models
model = models.resnet18(pretrained=True)
# Replace final layer
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 62)  # 62 classes
```

### 📝 Notebooks
- `ResNet-18/ResNet-18.ipynb` - Training
- `ResNet-18/ResNet-18_Image_Predictor.ipynb` - Prediction
- `ResNet-18/ResNet-18_Resolution_Comparison.ipynb` - Analysis

### 🌐 Web App
```bash
cd ResNet-18
python app.py  # http://localhost:5000
bash run_app.sh  # Alternative
```

---

## 6. Vision Transformer (ViT)

### 🏗️ Architecture
- **Type**: Transformer-based model
- **Framework**: TensorFlow/Keras
- **Input Size**: 64x64 or 128x128 images
- **Mechanism**: Self-attention on image patches

### 📊 Performance
- **Best Accuracy**: 92-97% ⭐ Highest!
- **Training Time**: ~60-120 minutes
- **Model Size**: ~80-150 MB
- **Inference Time**: ~50-100ms per image

### 🎯 Strengths
- Highest accuracy of all models
- State-of-the-art performance
- Better at capturing global patterns
- Excellent generalization

### ⚠️ Limitations
- Slowest inference time
- Largest model size
- More GPU memory required
- Requires more training data
- Longer training time

### 🔧 Configuration
```python
# Patch size and embedding dimension are key parameters
patch_size = 4
num_patches = (image_size // patch_size) ** 2
projection_dim = 64  # Embedding dimension
num_heads = 4
transformer_units = [128, 64]  # Feed-forward layers
```

### 📝 Notebooks
- `ViT/ViT_AlphaNum_Classification.ipynb` - Training
- `ViT/ViT_Image_Predictor.ipynb` - Prediction
- `ViT/ViT_Image_Size_Comparison.ipynb` - Analysis

---

## 📊 Model Comparison Table

| Aspect | KNN+HOG | SVM+HOG | LeNet-5 | MobileNetV2 | ResNet-18 | ViT |
|--------|---------|---------|---------|------------|-----------|-----|
| **Accuracy** | 80-85% | 82-87% | 85-88% | 88-92% | 90-95% | 92-97% |
| **Training Time** | ~10s | ~5min | ~20min | ~45min | ~90min | ~120min |
| **Inference Time** | 5-15ms | 2-5ms | 5-10ms | 15-20ms | 20-30ms | 50-100ms |
| **Model Size** | 100MB | 20MB | 1MB | 9MB | 50MB | 150MB |
| **Framework** | Sklearn | Sklearn | Keras | Keras | PyTorch | Keras |
| **Learning Curve** | Flat | Moderate | Steep | Moderate | Steep | Very Steep |
| **GPU Required** | ❌ | ❌ | ⚠️ Optional | ✅ Recommended | ✅ Required | ✅ Required |
| **Interpretability** | High | Medium | Medium | Low | Low | Very Low |
| **Deployment** | Easy | Easy | Medium | Medium | Medium | Hard |

---

## 🎯 Choosing the Right Model

### For Production Web Applications
**→ Use: ResNet-18 or MobileNetV2**
- Good accuracy-to-speed tradeoff
- Both have working web apps included
- Acceptable inference time

### For Mobile/Edge Devices
**→ Use: MobileNetV2**
- Lightweight
- Fast inference
- Reasonable accuracy

### For Maximum Accuracy
**→ Use: Vision Transformer**
- Best performance
- Worth the computational cost
- For non-real-time applications

### For Fast Inference (Real-time)
**→ Use: SVM + HOG**
- Fastest predictions (2-5ms)
- Good accuracy
- Minimal resources

### For Learning/Understanding
**→ Use: LeNet-5 or KNN + HOG**
- Simple architectures
- Easy to understand
- Good for education

### For Baseline Comparison
**→ Use: LeNet-5 + KNN**
- Fast training
- Easy to reproduce
- Good reference points

---

## 🔧 Hyperparameter Tuning

### Key Parameters by Model

**LeNet-5:**
- `learning_rate`: 0.001-0.01
- `batch_size`: 32-128
- `dropout_rate`: 0.1-0.3
- `dense_units`: 64-256

**KNN:**
- `n_neighbors`: 3-15 (usually 5 is good)
- `weights`: 'uniform' or 'distance'

**SVM:**
- `C`: 0.1-10 (regularization)
- `kernel`: 'linear', 'rbf', 'poly'
- `gamma`: 'scale' or 'auto'

**MobileNetV2:**
- `learning_rate`: 0.0001-0.001
- `batch_size`: 32-64
- `dropout_rate`: 0.2-0.5
- `optimizer`: 'adam' or 'rmsprop'

**ResNet-18:**
- `learning_rate`: 0.0001-0.001
- `batch_size`: 32-64
- `optimizer`: 'SGD' or 'Adam'
- `weight_decay`: 1e-4 to 1e-2

**ViT:**
- `patch_size`: 4-8
- `embedding_dim`: 64-256
- `num_heads`: 4-8
- `learning_rate`: 0.0001-0.001

---

## 📈 Performance Metrics Explanation

- **Accuracy**: Percentage of correct predictions
- **Precision**: Of predicted positives, how many are correct
- **Recall**: Of actual positives, how many were found
- **F1-Score**: Harmonic mean of Precision and Recall
- **Confusion Matrix**: Shows per-class prediction patterns

---

## 📞 Support

For issues or questions about specific models, see [RESULTS.md](RESULTS.md) for detailed performance analysis.

---

**Last Updated**: October 2024
