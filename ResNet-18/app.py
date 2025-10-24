"""
ResNet-18 Image Predictor Web Application
A Flask web app for predicting alphanumeric characters from images using ResNet-18

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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision import transforms, datasets
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
if (PROJECT_ROOT / "results" / "ResNet-18" / "resnet18_model.pth").exists():
    RESULTS_PATH = PROJECT_ROOT / "results" / "ResNet-18"
elif (BASE_DIR / "results" / "resnet18_model.pth").exists():
    RESULTS_PATH = BASE_DIR / "results"
else:
    RESULTS_PATH = PROJECT_ROOT / "results" / "ResNet-18"  # Default fallback

MODEL_PATH = RESULTS_PATH / "resnet18_model.pth"

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

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("🚀 RESNET-18 WEB PREDICTOR STARTING UP")
print("=" * 80)
print(f"📁 Base Directory: {BASE_DIR}")
print(f"📁 Project Root: {PROJECT_ROOT}")
print(f"📁 Results Path: {RESULTS_PATH}")
print(f"📁 Model Path: {MODEL_PATH}")
print(f"   Model exists: {'✅ YES' if MODEL_PATH.exists() else '❌ NO'}")
print(f"📁 Train Path: {TRAIN_PATH}")
print(f"   Train exists: {'✅ YES' if TRAIN_PATH.exists() else '❌ NO'}")
print(f"📁 Upload Folder: {UPLOAD_FOLDER}")
print(f"🖥️  Device: {device}")
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
        temp_dataset = datasets.ImageFolder(root=str(train_path))
        class_names = sorted(temp_dataset.classes)
        return class_names
    except Exception as e:
        print(f"❌ Error loading class names: {e}")
        return None

def load_resnet18_model(model_path, num_classes, device):
    """Load trained ResNet-18 model."""
    try:
        # Create model architecture (must match training architecture)
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Keep maxpool for 64x64 images
        
        # IMPORTANT: Must use Sequential with Dropout to match training architecture!
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),  # Must match the training configuration
            nn.Linear(num_ftrs, num_classes)
        )
        
        # Load weights
        state_dict = torch.load(str(model_path), map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# ============================================================================
# LOAD MODEL AT STARTUP
# ============================================================================

print("📚 Loading class names...")
class_names = load_class_names(TRAIN_PATH)
if class_names:
    print(f"✅ Loaded {len(class_names)} classes")
    num_classes = len(class_names)
else:
    print("❌ Failed to load class names")
    num_classes = 0

print("🔄 Loading ResNet-18 model...")
loaded_model = load_resnet18_model(MODEL_PATH, num_classes, device) if num_classes > 0 else None
if loaded_model:
    print("✅ Model loaded successfully!")
else:
    print("❌ Failed to load model")

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ascii_to_character(ascii_code):
    """Convert ASCII code to readable character."""
    try:
        if ascii_code == '999':
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
        
        # Apply transformations
        image_tensor = data_transforms(original_image).unsqueeze(0).to(device)
        
        # Create processed image for visualization
        processed_image = original_image.convert('L').resize((64, 64))
        
        # Make prediction
        loaded_model.eval()
        with torch.no_grad():
            outputs = loaded_model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        # Get top-k predictions
        top_k_probs, top_k_indices = torch.topk(probabilities[0], min(top_k, len(class_names)))
        
        # Prepare results
        predicted_class = class_names[predicted_idx.item()]
        predicted_char = ascii_to_character(predicted_class)
        confidence_score = confidence.item() * 100
        
        # Create top-k results
        top_predictions = []
        for i, (prob, idx) in enumerate(zip(top_k_probs, top_k_indices)):
            class_name = class_names[idx.item()]
            char = ascii_to_character(class_name)
            prob_percent = prob.item() * 100
            top_predictions.append({
                'rank': i + 1,
                'character': char,
                'ascii_code': class_name,
                'confidence': round(prob_percent, 2)
            })
        
        # Convert images to base64 for web display
        original_base64 = image_to_base64(original_image.convert('RGB'))
        processed_base64 = image_to_base64(processed_image.convert('RGB'))
        
        results = {
            'success': True,
            'predicted_character': predicted_char,
            'ascii_code': predicted_class,
            'confidence': round(confidence_score, 2),
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
        IP = '127.0.0.1' # Fallback to loopback
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
                         device=str(device))

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
        'device': str(device),
        'num_classes': num_classes
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
