# Handwritten Alphanumeric Character Recognition using Machine Learning

This project provides a comprehensive preprocessing pipeline for handwritten alphanumeric character images, enabling effective machine learning model training and evaluation.

## Features
- Dataset download via KaggleHub
- Image resizing and inspection
- Grayscale conversion
- Data normalization
- Label encoding (folder names to numerical labels)
- Brightness augmentation
- Histogram of Oriented Gradients (HoG) feature extraction

## Preprocessing Steps
1. **Data Loading**: Download and inspect dataset structure.
2. **Resizing**: Standardize image sizes for model input.
3. **Grayscale Conversion**: Convert images to grayscale for simplicity.
4. **Normalization**: Scale pixel values to [0, 1] range.
5. **Label Encoding**: Map folder names to numerical labels for classification.
6. **Brightness Augmentation**: Enhance dataset diversity.
7. **HoG Feature Extraction**: Extract robust features for recognition tasks.

## Usage
Run the provided Jupyter notebook to execute each preprocessing step. Outputs are saved in the `results/outputs` directory for further model development.

## Requirements
- Python 3.x
- Jupyter Notebook
- OpenCV
- NumPy
- Matplotlib
- scikit-image
- pandas
- kagglehub

Install dependencies using pip:
```bash
pip install opencv-python numpy matplotlib scikit-image pandas kagglehub
```

## Authors
- IT24102319 Wickramasurya J.N: Data loading , inspection and resize
- IT 241022268 Aluthge M.N: Grayscale conversion
- IT24102253 Gunasekara K.Y.A: Data normalization
- IT24102296 Diwanjani B S S: Label encoding
- IT24102318 Wickrama Edirisooriya A.A.G: Brightness augmentation
- IT24102263 Perera M.L.T: HoG feature extraction

## License

This project is licensed under the MIT License.

