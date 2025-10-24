# MobileNetV2 Web Predictor

A Flask-based web application for predicting alphanumeric characters from images using a trained MobileNetV2 model.

## 🎯 Features

- **Web Interface**: Beautiful, responsive web UI for image upload and prediction
- **Drag & Drop**: Easy drag-and-drop image upload
- **Real-time Predictions**: Instant character recognition with confidence scores
- **Top-5 Results**: See the top 5 most likely predictions
- **Image Preprocessing Visualization**: View original and processed images side-by-side
- **Cross-platform**: Works on Windows, Linux, and Mac
- **Network Access**: Access from any device on your local network

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.x
- Flask
- Trained MobileNetV2 model (`mobilenetv2_alphanum.h5`)
- Training data folder to get class information

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd "/home/ubuntu/Desktop/AIML project/moble net"
pip install -r requirements.txt
```

Or install manually:
```bash
pip install flask werkzeug tensorflow pillow numpy
```

### 2. Verify Model and Data Paths

The app automatically detects paths, but ensure you have:
- ✅ Model file: `/home/ubuntu/Desktop/AIML project/results/MobileNetV2/mobilenetv2_alphanum.h5`
- ✅ Training data: `/home/ubuntu/Desktop/AIML project/AlphaNum/train/`

### 3. Run the Web Application

```bash
python app.py
```

### 4. Access the Web Interface

Open your browser and go to:
- **Local**: http://localhost:8080
- **Network**: http://YOUR_IP:8080

The IP addresses will be displayed when the server starts.

## 📱 Usage

1. **Upload Image**: 
   - Click "Choose Image" button or drag-and-drop an image
   - Supported formats: PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP

2. **Predict**: 
   - Click "🔮 Predict Character" button
   - Wait for the analysis to complete

3. **View Results**:
   - See the predicted character with confidence score
   - View original and processed (96x96 grayscale) images
   - Check top-5 predictions with percentages

## 🏗️ Architecture

### Model Configuration
- **Model**: MobileNetV2 (modified for grayscale input)
- **Input Size**: 96x96 pixels
- **Color Mode**: Grayscale
- **Classes**: 53 (a-z, A-Z, NULL)
- **Preprocessing**: Grayscale conversion → Resize → Normalize [0, 1]

### Flask Routes
- `GET /`: Home page with upload interface
- `POST /predict`: Prediction endpoint
- `GET /health`: Health check endpoint

## 📂 Project Structure

```
moble net/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── templates/
│   └── index.html           # Web interface
└── uploads/                 # Uploaded images (auto-created)
```

## 🔧 Configuration

### Auto-Path Detection
The app automatically detects paths in the following priority:

**Model Path**:
1. `PROJECT_ROOT/results/MobileNetV2/mobilenetv2_alphanum.h5`
2. `BASE_DIR/results/mobilenetv2_alphanum.h5`

**Training Data Path**:
1. `PROJECT_ROOT/AlphaNum/train`
2. `PROJECT_ROOT/AlphaNum2/train`
3. `BASE_DIR/train`

### Manual Configuration
Edit `app.py` if you need to change paths:
```python
MODEL_PATH = Path("your/custom/model/path.h5")
TRAIN_PATH = Path("your/custom/train/path")
```

## 🐛 Troubleshooting

### Model Not Found
```
❌ Model file not found: /path/to/model.h5
```
**Solution**: Train the MobileNetV2 model first using `MobileNetV2.ipynb`

### Import Errors
```
ModuleNotFoundError: No module named 'tensorflow'
```
**Solution**: Install dependencies: `pip install -r requirements.txt`

### Port Already in Use
```
Address already in use
```
**Solution**: Change the port in `app.py`:
```python
app.run(host='0.0.0.0', port=8081, debug=True)
```

### Class Count Mismatch
```
⚠️ WARNING: Model has X output classes but found Y classes in training data!
```
**Solution**: Verify model was trained on the same dataset

## 📊 API Endpoints

### Predict
```bash
curl -X POST -F "file=@image.png" http://localhost:8080/predict
```

**Response**:
```json
{
    "success": true,
    "predicted_character": "A",
    "ascii_code": 65,
    "confidence": 98.76,
    "top_predictions": [
        {"rank": 1, "character": "A", "ascii_code": 65, "confidence": 98.76},
        {"rank": 2, "character": "H", "ascii_code": 72, "confidence": 0.85}
    ],
    "original_image": "data:image/png;base64,...",
    "processed_image": "data:image/png;base64,...",
    "original_size": [500, 500],
    "timestamp": "2025-10-20 12:34:56"
}
```

### Health Check
```bash
curl http://localhost:8080/health
```

**Response**:
```json
{
    "status": "running",
    "model_loaded": true,
    "model_name": "MobileNetV2",
    "num_classes": 53,
    "tensorflow_version": "2.x.x"
}
```

## 🎨 Customization

### Change Port
Edit line in `app.py`:
```python
app.run(host='0.0.0.0', port=YOUR_PORT, debug=True)
```

### Change Upload Folder
```python
UPLOAD_FOLDER = Path("your/custom/upload/folder")
```

### Modify File Size Limit
```python
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB
```

## 📝 Notes

- The model expects **96x96 grayscale** images
- Images are automatically preprocessed to match training configuration
- Predictions are made on ASCII-coded characters (65-90: A-Z, 97-122: a-z)
- NULL class (999) represents invalid/unrecognized characters
- Upload folder is automatically created if it doesn't exist

## 🔐 Security Notes

- File uploads are size-limited (16MB default)
- Only allowed image formats are accepted
- Filenames are sanitized using `secure_filename()`
- Files are timestamped to prevent overwrites

## 📞 Support

For issues or questions:
1. Check the model is properly trained and saved
2. Verify all paths are correct
3. Ensure dependencies are installed
4. Check the console output for detailed error messages

## 🎓 Model Information

This web app uses a MobileNetV2 model trained on alphanumeric characters:
- **Training**: See `MobileNetV2.ipynb`
- **Prediction**: See `MobileNetV2_Image_Predictor.ipynb`
- **Dataset**: AlphaNum with 53 classes

---

**Built with ❤️ using Flask and TensorFlow**
