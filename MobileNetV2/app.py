"""
MobileNetV2 Image Predictor Web Application
A Flask web app for predicting alphanumeric characters from images using MobileNetV2

Features:
- Upload images through web interface
- Real-time prediction with confidence scores
- Top-5 predictions display
- Cross-platform support (Windows/Linux/Mac)
- Automatic model and class loading
- Image preprocessing visualization
"""

import os
import io
import json
import base64
import socket
from pathlib import Path
from datetime import datetime

# Flask imports
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# ML imports
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import numpy as np

# ============================================================================
# CONFIGURATION - Auto-detects paths based on current location
# ============================================================================

# Get the directory where this script is located
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Model paths - Check multiple possible locations
# Priority: Project results folder > Local results folder
if (PROJECT_ROOT / "results" / "MobileNetV2" / "mobilenetv2_alphanum.h5").exists():
    RESULTS_PATH = PROJECT_ROOT / "results" / "MobileNetV2"
elif (BASE_DIR / "results" / "mobilenetv2_alphanum.h5").exists():
    RESULTS_PATH = BASE_DIR / "results"
else:
    RESULTS_PATH = PROJECT_ROOT / "results" / "MobileNetV2"  # Default fallback

MODEL_PATH = RESULTS_PATH / "mobilenetv2_alphanum.h5"

# Training data path (to get class names)
# Priority: AlphaNum > AlphaNum2
if (PROJECT_ROOT / "AlphaNum" / "train").exists():
    TRAIN_PATH = PROJECT_ROOT / "AlphaNum" / "train"
elif (PROJECT_ROOT / "AlphaNum2" / "train").exists():
    TRAIN_PATH = PROJECT_ROOT / "AlphaNum2" / "train"
else:
    TRAIN_PATH = BASE_DIR / "train"  # Default fallback

# Upload folder
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

# Image preprocessing configuration (must match training)
IMG_HEIGHT = 96
IMG_WIDTH = 96
COLOR_MODE = "grayscale"

print("=" * 80)
print("🚀 MOBILENETV2 WEB PREDICTOR STARTING UP")
print("=" * 80)
print(f"📁 Base Directory: {BASE_DIR}")
print(f"📁 Project Root: {PROJECT_ROOT}")
print(f"📁 Results Path: {RESULTS_PATH}")
print(f"📁 Model Path: {MODEL_PATH}")
print(f"   Model exists: {'✅ YES' if MODEL_PATH.exists() else '❌ NO'}")
print(f"📁 Train Path: {TRAIN_PATH}")
print(f"   Train exists: {'✅ YES' if TRAIN_PATH.exists() else '❌ NO'}")
print(f"📁 Upload Folder: {UPLOAD_FOLDER}")
print(f"🖥️  TensorFlow Version: {tf.__version__}")
print(f"🖥️  GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
print("=" * 80)

# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

def load_class_names(train_path):
    """Load class names from training dataset."""
    try:
        allowed_ascii = list(range(ord('a'), ord('z') + 1)) + list(range(ord('A'), ord('Z') + 1))
        
        # Collect unique labels from training directory
        unique_labels = []
        for folder_name in sorted(os.listdir(str(train_path))):
            folder_path = train_path / folder_name
            if folder_path.is_dir():
                if folder_name == '999' or (folder_name.isdigit() and int(folder_name) in allowed_ascii):
                    label = int(folder_name) if folder_name.isdigit() else 999
                    unique_labels.append(label)
        
        unique_labels = sorted(list(set(unique_labels)))
        
        # Create reverse label mapping (model index -> ASCII code)
        reverse_label_mapping = {i: label for i, label in enumerate(unique_labels)}
        
        return unique_labels, reverse_label_mapping
    except Exception as e:
        print(f"❌ Error loading class names: {e}")
        return None, None

def load_mobilenetv2_model(model_path):
    """Load trained MobileNetV2 model."""
    try:
        model = load_model(str(model_path))
        print(f"✅ Model loaded successfully!")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output classes: {model.output_shape[-1]}")
        print(f"   Total parameters: {model.count_params():,}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# ============================================================================
# LOAD MODEL AT STARTUP
# ============================================================================

print("📚 Loading class names...")
class_list, reverse_label_mapping = load_class_names(TRAIN_PATH)
if class_list:
    print(f"✅ Loaded {len(class_list)} classes")
    num_classes = len(class_list)
else:
    print("❌ Failed to load class names")
    num_classes = 0

print("🔄 Loading MobileNetV2 model...")
loaded_model = load_mobilenetv2_model(MODEL_PATH) if num_classes > 0 else None
if loaded_model:
    print("✅ Model ready for predictions!")
else:
    print("❌ Failed to load model")

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ascii_to_character(ascii_code):
    """Convert ASCII code to readable character."""
    try:
        if ascii_code == 999:
            return 'NULL'
        return chr(int(ascii_code))
    except (ValueError, OverflowError):
        return 'UNKNOWN'

def image_to_base64(image):
    """Convert PIL image to base64 string for display in browser."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def preprocess_and_predict(image_path, top_k=5):
    """Preprocess image and make prediction."""
    try:
        # Load and preprocess image
        original_image = Image.open(image_path)
        
        # Convert to RGB if needed
        if original_image.mode not in ['RGB', 'L']:
            original_image = original_image.convert('RGB')
        
        # Load and preprocess for model (matches training)
        img = load_img(image_path, color_mode=COLOR_MODE, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Create processed image for visualization
        processed_image = original_image.convert('L').resize((IMG_HEIGHT, IMG_WIDTH))
        
        # Make prediction
        predictions = loaded_model.predict(img_array, verbose=0)
        predicted_class_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_index] * 100
        
        # Get top-k predictions
        top_k_indices = np.argsort(predictions[0])[-top_k:][::-1]
        top_k_probs = predictions[0][top_k_indices] * 100
        
        # Get predicted character
        predicted_original_label = reverse_label_mapping[predicted_class_index]
        predicted_char = ascii_to_character(predicted_original_label)
        
        # Create top-k results
        top_predictions = []
        for i, (idx, prob) in enumerate(zip(top_k_indices, top_k_probs)):
            original_label = reverse_label_mapping[idx]
            char = ascii_to_character(original_label)
            top_predictions.append({
                'rank': i + 1,
                'character': char,
                'ascii_code': original_label,
                'confidence': round(float(prob), 2)
            })
        
        # Convert images to base64 for web display
        original_base64 = image_to_base64(original_image.convert('RGB'))
        processed_base64 = image_to_base64(processed_image.convert('RGB'))
        
        results = {
            'success': True,
            'predicted_character': predicted_char,
            'ascii_code': int(predicted_original_label),
            'confidence': round(float(confidence), 2),
            'top_predictions': top_predictions,
            'original_image': original_base64,
            'processed_image': processed_base64,
            'original_size': original_image.size,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return results
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# ============================================================================
# GET IP ADDRESS
# ============================================================================

def get_local_ip():
    """Gets the local IP address of the machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'  # Fallback to loopback
    finally:
        s.close()
    return IP

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Home page."""
    model_status = "✅ Ready" if loaded_model else "❌ Not Loaded"
    return render_template('index.html', 
                         model_status=model_status,
                         num_classes=num_classes,
                         model_name="MobileNetV2",
                         image_size=f"{IMG_HEIGHT}x{IMG_WIDTH}")

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    if loaded_model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'})
    
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        results = preprocess_and_predict(filepath, top_k=5)
        
        return jsonify(results)
    else:
        return jsonify({'success': False, 'error': 'Invalid file type'})

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'running',
        'model_loaded': loaded_model is not None,
        'model_name': 'MobileNetV2',
        'num_classes': num_classes,
        'tensorflow_version': tf.__version__
    })

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    templates_dir = BASE_DIR / "templates"
    templates_dir.mkdir(exist_ok=True)
    
    local_ip = get_local_ip()
    
    print("\n" + "=" * 80)
    print("🌐 Starting Flask Web Server")
    print("=" * 80)
    print("📡 Access the web app at:")
    print("   - Local: http://localhost:8080")
    print("   - Network: http://0.0.0.0:8080")
    print(f"   - On Your Network 📱: http://{local_ip}:8080")
    print()
    print("⌨️  Press CTRL+C to stop the server")
    print("=" * 80 + "\n")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)
