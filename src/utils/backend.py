# Import suppression utility at the very top
import os
import sys

# Add this at the very top
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from utils.suppress_tf_warnings import suppress_tf_warnings

# Call the function before any other imports
suppress_tf_warnings()

from flask import Flask, request, jsonify, render_template, send_from_directory, redirect
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os
import uuid
from werkzeug.utils import secure_filename
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Import the configuration
from config import Paths

# Create necessary directories
Paths.create_directories()

from flask_cors import CORS
app = Flask(__name__, static_folder='static', 
            template_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'app'))
CORS(app)  # This enables CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = Paths.UPLOADS_DIR
RESULT_FOLDER = Paths.RESULTS_DIR
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Import the vit_model modules if available
try:
    # Try to import the custom model classes if they exist
    from src.training.vit_model import Patches, PatchEncoder
    custom_objects = {
        'Patches': Patches,
        'PatchEncoder': PatchEncoder
    }
except ImportError:
    # If the imports fail, create dummy classes
    print("Could not import ViT model classes, will use dummy versions if needed")
    custom_objects = {}
    
    # Create dummy classes in case they're needed
    class Patches(tf.keras.layers.Layer):
        def __init__(self, patch_size):
            super().__init__()
            self.patch_size = patch_size
            
        def call(self, images):
            return images
            
        def get_config(self):
            config = super().get_config()
            config.update({"patch_size": self.patch_size})
            return config
            
    class PatchEncoder(tf.keras.layers.Layer):
        def __init__(self, num_patches, projection_dim):
            super().__init__()
            self.num_patches = num_patches
            self.projection_dim = projection_dim
            
        def call(self, patch):
            return patch
            
        def get_config(self):
            config = super().get_config()
            config.update({
                "num_patches": self.num_patches,
                "projection_dim": self.projection_dim
            })
            return config
            
    custom_objects = {
        'Patches': Patches,
        'PatchEncoder': PatchEncoder
    }

# Load the trained classifier model
model_loaded = False

# Try to load the autoencoder model first
if not model_loaded:
    try:
        print(f"Attempting to load autoencoder model from {Paths.AUTOENCODER_MODEL}")
        classifier = load_model(Paths.AUTOENCODER_MODEL)
        print("Autoencoder disease model loaded successfully!")
        model_loaded = True
    except Exception as e:
        print(f"Error loading autoencoder model: {e}")

# Try CNN model if autoencoder failed
if not model_loaded:
    try:
        print(f"Attempting to load CNN model from {Paths.CNN_MODEL}")
        classifier = load_model(Paths.CNN_MODEL)
        print("CNN disease model loaded successfully!")
        model_loaded = True
    except Exception as e:
        print(f"Error loading CNN model: {e}")

# Try ViT model if both previous models failed
if not model_loaded:
    try:
        print(f"Attempting to load ViT model from {Paths.VIT_MODEL}")
        classifier = load_model(Paths.VIT_MODEL, custom_objects=custom_objects)
        print("ViT disease model loaded successfully!")
        model_loaded = True
    except Exception as e:
        print(f"Error loading ViT model: {e}")

# If all models failed, create a simple dummy model for testing
if not model_loaded:
    print("All model loading attempts failed. Creating a dummy model for testing purposes.")
    inputs = tf.keras.layers.Input(shape=(128, 128, 3))
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(len(class_labels), activation='softmax')(x)
    classifier = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    classifier.compile(optimizer='adam', loss='categorical_crossentropy')
    print("Dummy model created!")
    model_loaded = True

# Load MobileNetV2 for leaf detection
try:
    leaf_detector = MobileNetV2(weights='imagenet', include_top=True)
    print("Leaf detection model loaded successfully!")
except Exception as e:
    print(f"Error loading leaf detection model: {e}")
    leaf_detector = None

# Define leaf-related categories from ImageNet
LEAF_RELATED_CATEGORIES = [
    'fig', 'leaf', 'daisy', 'corn', 'strawberry', 'artichoke', 'head_cabbage',
    'broccoli', 'cauliflower', 'cucumber', 'bell_pepper', 'mushroom', 'Granny_Smith',
    'banana', 'jackfruit', 'custard_apple', 'pomegranate', 'acorn', 'hip', 'ear', 'rapeseed',
    'yellow_lady\'s_slipper', 'corn', 'buckeye', 'agaric', 'gyromitra', 'earthstar',
    'hen-of-the-woods', 'bolete', 'cardoon', 'zucchini', 'spaghetti_squash'
]

# Define plant-related categories
PLANT_CATEGORIES = [
    'plant', 'herb', 'vegetable', 'fruit', 'flower', 'tree', 'shrub', 'garden', 
    'crop', 'leaf', 'petal', 'stem', 'root', 'seed', 'branch', 'flower'
]

# Define class labels and recommendations
class_labels = {
    'Apple___Apple_scab': {
        "name": "Apple Scab",
        "symptoms": "Olive-green to brown spots on leaves and fruit. Infected leaves may drop prematurely.",
        "treatments": [
            "Apply fungicides like Captan or Mancozeb",
            "Prune infected leaves and branches",
            "Remove fallen leaves to reduce inoculum",
            "Improve air circulation around trees"
        ],
        "prevention": [
            "Plant resistant apple varieties",
            "Maintain proper tree spacing",
            "Apply preventative fungicide sprays",
            "Practice good orchard sanitation"
        ]
    },
    'Apple___Black_rot': {
        "name": "Apple Black Rot",
        "symptoms": "Circular purple spots on leaves. Infected fruit develops concentric brown rings and may mummify.",
        "treatments": [
            "Remove infected fruit and cankers",
            "Use fungicides like copper sprays",
            "Prune out dead or infected branches",
            "Destroy mummified fruits"
        ],
        "prevention": [
            "Remove nearby wild apple trees",
            "Maintain tree vigor with proper fertilization",
            "Ensure adequate spacing between trees",
            "Apply dormant spray treatments"
        ]
    },
    'Apple___Cedar_apple_rust': {
        "name": "Cedar Apple Rust",
        "symptoms": "Bright orange-yellow spots on leaves and fruit. Raised orange lesions with small black dots.",
        "treatments": [
            "Apply fungicides during growing season",
            "Remove infected leaves",
            "Reduce humidity around trees",
            "Improve soil drainage"
        ],
        "prevention": [
            "Use resistant apple varieties",
            "Remove nearby juniper trees (alternate host)",
            "Apply protective fungicides before rainy periods",
            "Maintain good air circulation"
        ]
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        "name": "Corn Gray Leaf Spot",
        "symptoms": "Rectangular gray to tan lesions on leaves, following leaf veins. Lesions expand and may kill entire leaves.",
        "treatments": [
            "Rotate crops with non-host plants",
            "Apply foliar fungicides",
            "Remove and destroy crop residue",
            "Balance soil nutrition"
        ],
        "prevention": [
            "Plant resistant corn hybrids",
            "Avoid excessive nitrogen fertilization",
            "Practice crop rotation with non-host crops",
            "Ensure proper field drainage"
        ]
    },
    'Corn_(maize)___Common_rust_': {
        "name": "Corn Common Rust",
        "symptoms": "Small, round to elongated, reddish-brown pustules on leaf surfaces. Severe infections lead to leaf death.",
        "treatments": [
            "Apply appropriate fungicides",
            "Remove heavily infected plants",
            "Increase air circulation",
            "Balance soil nutrients"
        ],
        "prevention": [
            "Use rust-resistant varieties",
            "Plant early to avoid peak rust periods",
            "Monitor fields regularly for early detection",
            "Rotate with non-host crops"
        ]
    },
    'Grape___Esca_(Black_Measles)': {
        "name": "Grape Black Measles (Esca)",
        "symptoms": "Red-brown blotches on leaves with yellow margins. Affected berries show black spots. Wood shows dark streaking.",
        "treatments": [
            "Prune infected vines during dormancy",
            "Protect pruning wounds",
            "Avoid excessive fertilization",
            "Remove severely infected vines"
        ],
        "prevention": [
            "Use clean pruning tools",
            "Apply wound protectants",
            "Maintain balanced irrigation",
            "Avoid mechanical damage to vines"
        ]
    },
    'Potato___Late_blight': {
        "name": "Potato Late Blight",
        "symptoms": "Water-soaked pale to dark green spots on leaves, white fungal growth on leaf undersides. Brown lesions on tubers.",
        "treatments": [
            "Remove infected plants immediately",
            "Apply fungicides like chlorothalonil",
            "Increase plant spacing",
            "Avoid overhead irrigation"
        ],
        "prevention": [
            "Plant certified disease-free seed potatoes",
            "Destroy volunteer potatoes",
            "Practice crop rotation",
            "Use resistant varieties if available"
        ]
    },
    'Tomato___Bacterial_spot': {
        "name": "Tomato Bacterial Spot",
        "symptoms": "Small, water-soaked spots on leaves, stems and fruit that enlarge and turn dark brown. Affected fruit has raised rough spots.",
        "treatments": [
            "Apply copper-based bactericides",
            "Remove infected plant material",
            "Avoid overhead watering",
            "Provide good air circulation"
        ],
        "prevention": [
            "Use disease-free seeds and transplants",
            "Rotate crops for 2-3 years",
            "Avoid working with wet plants",
            "Maintain proper plant spacing"
        ]
    },
    'Tomato___Early_blight': {
        "name": "Tomato Early Blight",
        "symptoms": "Dark brown spots with concentric rings forming a target-like pattern. Affected leaves turn yellow and drop.",
        "treatments": [
            "Remove and destroy infected leaves",
            "Apply fungicide treatments",
            "Ensure proper plant spacing for airflow",
            "Water at the base of plants to avoid wetting leaves"
        ],
        "prevention": [
            "Use disease-resistant varieties",
            "Practice crop rotation",
            "Keep garden clean of plant debris",
            "Apply organic mulch to prevent splashing"
        ]
    },
    'Tomato___Leaf_Mold': {
        "name": "Tomato Leaf Mold",
        "symptoms": "Pale green or yellow spots on the upper leaf surface with olive-green to grayish-purple velvety growth on the undersides.",
        "treatments": [
            "Increase air circulation",
            "Reduce humidity in greenhouse settings",
            "Apply fungicides",
            "Remove severely infected leaves"
        ],
        "prevention": [
            "Use resistant varieties",
            "Avoid overhead watering",
            "Increase plant spacing",
            "Remove crop debris after harvest"
        ]
    }
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to check if image contains a leaf using MobileNetV2
def is_leaf_mobilenet(img_path, confidence_threshold=0.05):
    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict with MobileNetV2
        preds = leaf_detector.predict(img_array)
        decoded_preds = decode_predictions(preds, top=5)[0]
        
        # Create a visualization for debugging
        img_display = cv2.imread(img_path)
        img_display = cv2.resize(img_display, (224, 224))
        
        # Add text for top predictions
        for i, (_, label, conf) in enumerate(decoded_preds):
            text = f"{label}: {conf:.2f}"
            cv2.putText(img_display, text, (10, 30 + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if label in LEAF_RELATED_CATEGORIES else (0, 0, 255), 2)
        
        # Save the visualization
        vis_path = os.path.join(app.config['RESULT_FOLDER'], f"leaf_check_{uuid.uuid4()}.jpg")
        cv2.imwrite(vis_path, img_display)
        
        print(f"Leaf detection results: {decoded_preds}")
        
        # Check if image is likely a plant leaf using alternative approach
        is_leaf_by_color = is_leaf_by_color_analysis(img_path)
        
        # Log the results of both methods
        print(f"MobileNet predictions: {decoded_preds}")
        print(f"Color-based leaf detection: {is_leaf_by_color}")
        
        # Check if any top predictions are leaf-related
        is_leaf_by_mobilenet = False
        max_conf = 0
        leaf_pred = None
        
        for _, label, confidence in decoded_preds:
            # Check if label contains any plant-related terms
            if any(category in label for category in PLANT_CATEGORIES) or any(label in category for category in LEAF_RELATED_CATEGORIES):
                is_leaf_by_mobilenet = True
                if confidence > max_conf:
                    max_conf = confidence
                    leaf_pred = label
        
        # Combine both methods - if either method says it's a leaf, proceed
        # This makes the system more lenient
        is_leaf = is_leaf_by_mobilenet or is_leaf_by_color
        
        print(f"Final decision - Is leaf: {is_leaf}, by MobileNet: {is_leaf_by_mobilenet}, by color: {is_leaf_by_color}")
        
        return is_leaf, max_conf if max_conf > 0 else 0.5, os.path.basename(vis_path)
        
    except Exception as e:
        print(f"Error in leaf detection: {e}")
        # In case of error, assume it's a leaf to avoid rejecting valid images
        return True, 0.5, None

# Function to detect leaf using color analysis
def is_leaf_by_color_analysis(img_path, threshold=0.15):
    try:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            return False
            
        # Resize for faster processing
        img = cv2.resize(img, (224, 224))
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define multiple green ranges for different types of leaves
        # Lighter green
        lower_green1 = np.array([25, 20, 20])
        upper_green1 = np.array([95, 255, 255])
        
        # Darker/olive green
        lower_green2 = np.array([20, 10, 10])
        upper_green2 = np.array([80, 255, 180])
        
        # Brown/yellow for autumn leaves
        lower_brown = np.array([5, 20, 20])
        upper_brown = np.array([30, 255, 255])
        
        # Create masks
        green_mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
        green_mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(green_mask1, green_mask2)
        combined_mask = cv2.bitwise_or(combined_mask, brown_mask)
        
        # Calculate ratio of colored pixels to total pixels
        colored_ratio = np.count_nonzero(combined_mask) / (combined_mask.shape[0] * combined_mask.shape[1])
        
        # Check texture using Laplacian variance
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        print(f"Color analysis - colored ratio: {colored_ratio:.2f}, blur score: {blur_score:.2f}")
        
        # If significant portion is green/brown and has some texture, likely a leaf
        return colored_ratio > threshold and blur_score > 30
        
    except Exception as e:
        print(f"Error in color-based leaf detection: {e}")
        return True  # On error, assume it's a leaf

# Add this function for detailed color-based leaf validation
def validate_leaf_comprehensive(img_path, threshold=0.3):
    """
    Enhanced multi-method leaf validation system that combines:
    1. Color analysis
    2. Texture analysis
    3. Shape analysis
    4. Edge detection
    5. Background/foreground separation
    
    Args:
        img_path (str): Path to the image file
        threshold (float): Confidence threshold (0-1)
        
    Returns:
        tuple: (is_leaf, confidence, debug_image_path)
    """
    try:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image at {img_path}")
            return False, 0, None
            
        # Create a copy for visualization
        debug_img = img.copy()
        orig_img = img.copy()
        
        # Resize for faster processing
        img = cv2.resize(img, (224, 224))
        
        # =====================================================
        # TEST 1: COLOR ANALYSIS
        # =====================================================
        # Convert to HSV color space for better color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define multiple color ranges for different types of leaves
        # Green leaves (healthy)
        lower_green = np.array([25, 30, 30])
        upper_green = np.array([95, 255, 255])
        
        # Yellow/brown leaves (autumn/diseased)
        lower_yellow = np.array([15, 30, 30])
        upper_yellow = np.array([35, 255, 255])
        
        # Brown leaves (dried/diseased)
        lower_brown = np.array([0, 20, 20])
        upper_brown = np.array([20, 255, 180])
        
        # Dark green/olive leaves
        lower_olive = np.array([20, 10, 10])
        upper_olive = np.array([80, 255, 180])
        
        # Create masks for each color range
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        olive_mask = cv2.inRange(hsv, lower_olive, upper_olive)
        
        # Combine all masks for plant material detection
        plant_mask = cv2.bitwise_or(
            cv2.bitwise_or(
                cv2.bitwise_or(green_mask, yellow_mask),
                brown_mask
            ),
            olive_mask
        )
        
        # Apply morphology operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, kernel)
        plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN, kernel)
        
        # Calculate ratio of plant pixels to total pixels
        plant_ratio = np.count_nonzero(plant_mask) / (plant_mask.shape[0] * plant_mask.shape[1])
        
        # =====================================================
        # TEST 2: TEXTURE ANALYSIS
        # =====================================================
        # Calculate texture metrics - leaves have natural texture
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Analyze texture using Laplacian (high values indicate more texture)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Check for uniform regions (bad sign for leaf images)
        # Natural leaves shouldn't have large uniform regions
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours_uniform, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Detect if there are large rectangular regions (typical for documents, not leaves)
        has_rectangular_regions = False
        for c in contours_uniform:
            if cv2.contourArea(c) > (img.shape[0] * img.shape[1] * 0.4):  # Large region
                # Check if it's rectangular
                x, y, w, h = cv2.boundingRect(c)
                rect_ratio = cv2.contourArea(c) / (w * h)
                if rect_ratio > 0.85:  # Very rectangular
                    has_rectangular_regions = True
                    break
        
        # Penalize rectangular images heavily - they're likely documents, not leaves
        texture_score = 0 if has_rectangular_regions else min(blur_score / 500, 1.0)
        
        # =====================================================
        # TEST 3: SHAPE ANALYSIS
        # =====================================================
        # Find contours to analyze shape characteristics in the plant mask
        contours, _ = cv2.findContours(plant_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Initialize shape metrics
        shape_score = 0
        solidity = 0
        circularity = 0
        edge_complexity = 0
        
        # If significant contours found, analyze shape
        if contours and max([cv2.contourArea(c) for c in contours] or [0]) > 500:
            # Get the largest contour (main leaf shape)
            main_contour = max(contours, key=cv2.contourArea)
            
            # Calculate shape metrics
            area = cv2.contourArea(main_contour)
            perimeter = cv2.arcLength(main_contour, True)
            
            # Calculate circularity - leaves typically have moderate values
            # Perfect circle = 1, complex shape < 1
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Calculate solidity (area / convex hull area)
            hull = cv2.convexHull(main_contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
            
            # Calculate edge complexity (perimeter / sqrt of area)
            # Higher values indicate more complex edges (like leaf serrations)
            if area > 0:
                edge_complexity = perimeter / (2 * np.sqrt(np.pi * area))
            
            # Shape score calculation - optimized for leaf shapes
            # Documents/screenshots have simpler shapes than leaves
            
            # Penalize very rectangular or very circular shapes
            if circularity > 0.9:  # Too circular to be a leaf
                shape_score = 0
            elif solidity > 0.95:  # Too solid/rectangular to be a leaf
                shape_score = 0
            else:
                # Leaf shapes typically have solidity 0.6-0.9 and moderate circularity
                solidity_score = max(0, min(1, (solidity - 0.5) * 2.5)) if solidity < 0.95 else 0
                circularity_score = max(0, 0.8 - circularity) * 1.25  # Lower is better, but not too low
                complexity_score = min(edge_complexity / 2, 1.0)  # Reward complex edges
                
                shape_score = (solidity_score + circularity_score + complexity_score) / 3
            
            # Draw contour on debug image for visualization
            debug_img = cv2.resize(debug_img, (224, 224))
            cv2.drawContours(debug_img, [main_contour], 0, (0, 255, 0), 2)
        
        # =====================================================
        # TEST 4: TEXT DETECTION
        # =====================================================
        # Documents/screenshots often have text, leaves don't
        # We'll use edge detection to look for patterns indicating text
        
        # Use Canny edge detector with parameters suitable for text
        edges = cv2.Canny(gray, 100, 200)
        
        # Count edge pixels
        edge_ratio = np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1])
        
        # Calculate horizontal line pattern detection (common in text)
        horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_structure)
        horizontal_line_ratio = np.count_nonzero(horizontal_lines) / (horizontal_lines.shape[0] * horizontal_lines.shape[1])
        
        # Calculate if the edge distribution is uniform (like text lines) or random (like leaves)
        # Divide the image into rows and check edge distribution
        row_sums = np.sum(edges, axis=1) / edges.shape[1]
        edge_std = np.std(row_sums)  # Lower values suggest uniform distribution
        
        # Text typically has uniform edge distribution, leaves don't
        has_text_pattern = (horizontal_line_ratio > 0.15) or (edge_std < 0.1 and edge_ratio > 0.05)
        
        # Calculate text detection score (0 if text detected, 1 if no text)
        text_score = 0 if has_text_pattern else 1
        
        # =====================================================
        # FINAL SCORING
        # =====================================================
        # Calculate combined confidence score with weighted components
        color_weight = 0.30
        texture_weight = 0.25
        shape_weight = 0.25
        text_weight = 0.20
        
        # Calculate final confidence
        confidence = (
            plant_ratio * color_weight + 
            texture_score * texture_weight + 
            shape_score * shape_weight +
            text_score * text_weight
        )
        
        # Apply penalties for clear non-leaf indicators
        if has_rectangular_regions:
            confidence *= 0.3  # Heavy penalty for document-like images
        
        if has_text_pattern:
            confidence *= 0.4  # Heavy penalty for text patterns
            
        # If the image is very high resolution (typical for screenshots/documents)
        orig_resolution = orig_img.shape[0] * orig_img.shape[1]
        if orig_resolution > 2000 * 2000 and circularity > 0.8:
            confidence *= 0.5  # Penalty for high-res rectangular images
        
        # Determine if it's a leaf based on confidence threshold
        is_leaf = confidence > threshold
        
        # Add metrics as text on debug image
        metrics_text = [
            f"Plant ratio: {plant_ratio:.2f}",
            f"Texture: {texture_score:.2f}",
            f"Shape: {shape_score:.2f}",
            f"Text (inv): {text_score:.2f}",
            f"Rect regions: {has_rectangular_regions}",
            f"Text pattern: {has_text_pattern}",
            f"Confidence: {confidence:.2f}"
        ]
        
        for i, text in enumerate(metrics_text):
            cv2.putText(debug_img, text, (10, 25 + 25*i), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(debug_img, text, (10, 25 + 25*i), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Create visualization of analysis components
        plant_vis = cv2.bitwise_and(img, img, mask=plant_mask)
        edges_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Create a multi-panel visualization
        top_row = np.hstack((img, plant_vis))
        bottom_row = np.hstack((edges_vis, debug_img))
        
        # Combine into final debug visualization
        debug_vis = np.vstack((top_row, bottom_row))
        
        # Label the visualization sections
        cv2.putText(debug_vis, "Original", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_vis, "Plant Mask", (224 + 10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_vis, "Edge Detection", (10, 224 + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_vis, "Analysis", (224 + 10, 224 + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw a decision label at the bottom
        decision_text = "LEAF" if is_leaf else "NOT A LEAF"
        cv2.rectangle(debug_vis, (0, debug_vis.shape[0] - 40), 
                     (debug_vis.shape[1], debug_vis.shape[0]), (0, 0, 0), -1)
        cv2.putText(debug_vis, decision_text, (debug_vis.shape[1]//2 - 70, debug_vis.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if is_leaf else (0, 0, 255), 2)
        
        # Save debug image
        debug_path = os.path.join(app.config['RESULT_FOLDER'], f"leaf_validate_{uuid.uuid4()}.jpg") 
        cv2.imwrite(debug_path, debug_vis)
        
        print(f"Leaf validation results:")
        print(f"Plant ratio: {plant_ratio:.2f}, Texture: {texture_score:.2f}")
        print(f"Shape: {shape_score:.2f}, Text score: {text_score:.2f}")
        print(f"Final leaf confidence: {confidence:.2f}, is leaf: {is_leaf}")
        
        return is_leaf, confidence * 100, os.path.basename(debug_path)
        
    except Exception as e:
        import traceback
        print(f"Error in leaf validation: {e}")
        print(traceback.format_exc())
        
        # Return default values on error
        return False, 0, None

# Function to preprocess the image
def preprocess_image(img_path, target_size=(128, 128)):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file '{img_path}' not found.")
    
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand to batch format
    img_array = img_array / 255.0  # Normalize to [0,1]
    return img_array

# Function to predict the class
def predict_image(img_path):
    if classifier is None:
        return {"error": "Model not loaded properly"}
    
    try:
        img_array = preprocess_image(img_path)
        prediction = classifier.predict(img_array)
        predicted_class_idx = np.argmax(prediction, axis=1)[0]
        predicted_class_name = list(class_labels.keys())[predicted_class_idx]
        confidence = float(prediction[0][predicted_class_idx] * 100)
        
        return {
            "class_name": predicted_class_name,
            "display_name": class_labels[predicted_class_name]["name"],
            "confidence": confidence,
            "symptoms": class_labels[predicted_class_name]["symptoms"],
            "treatments": class_labels[predicted_class_name]["treatments"],
            "prevention": class_labels[predicted_class_name]["prevention"]
        }
    except Exception as e:
        return {"error": str(e)}

# Function to generate Grad-CAM heatmap
def generate_heatmap_image(img_path):
    if classifier is None:
        return None, {"error": "Model not loaded properly"}
    
    try:
        # Load original image
        original_img = cv2.imread(img_path)
        
        # Get image array for prediction
        img_array = preprocess_image(img_path)
        
        # Predict class
        prediction = classifier.predict(img_array)
        predicted_class_idx = np.argmax(prediction, axis=1)[0]
        predicted_class_name = list(class_labels.keys())[predicted_class_idx]
        
        # Generate Grad-CAM
        layer_name = "conv2d"  # Adjust based on your model
        grad_model = tf.keras.models.Model(
            inputs=classifier.input,
            outputs=[classifier.get_layer(layer_name).output, classifier.output]
        )
        
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(img_array)
            class_idx = predicted_class_idx
            loss = predictions[:, class_idx]
        
        grads = tape.gradient(loss, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_output = conv_output[0]
        heatmap = np.mean(conv_output * pooled_grads, axis=-1)
        
        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1
        
        # Resize and apply colormap
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superimpose heatmap on original image
        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
        
        # Create disease mask
        _, binary_mask = cv2.threshold(cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY), 
                                      int(0.3 * 255), 255, cv2.THRESH_BINARY)
        
        # Calculate affected area percentage
        affected_pixels = np.sum(binary_mask == 255)
        total_leaf_pixels = np.sum(cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY) > 0)
        
        if total_leaf_pixels == 0:
            affected_percentage = 0
        else:
            affected_percentage = (affected_pixels / total_leaf_pixels) * 100
        
        # Generate a filename for the result
        result_filename = f"{uuid.uuid4()}.jpg"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        
        # Save the result
        cv2.imwrite(result_path, superimposed_img)
        
        return result_filename, {
            "class_name": predicted_class_name,
            "display_name": class_labels[predicted_class_name]["name"],
            "confidence": float(prediction[0][predicted_class_idx] * 100),
            "affected_percentage": round(affected_percentage, 2),
            "symptoms": class_labels[predicted_class_name]["symptoms"],
            "treatments": class_labels[predicted_class_name]["treatments"],
            "prevention": class_labels[predicted_class_name]["prevention"]
        }
    except Exception as e:
        return None, {"error": str(e)}

@app.route('/')
def index():
    return render_template('frontend.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        # Generate a secure filename with UUID to avoid collisions
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        file.save(file_path)
        
        return jsonify({
            "success": True,
            "filename": unique_filename,
            "file_url": f"/uploads/{unique_filename}"
        })
    
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    
    if not data or 'filename' not in data:
        return jsonify({"error": "No filename provided"}), 400
    
    filename = data['filename']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    
    # Check if we should bypass validation (for debugging or testing)
    bypass_validation = data.get('bypass_validation', False)
    
    if not bypass_validation:
        # Use our enhanced comprehensive validation function
        is_leaf, confidence, debug_image = validate_leaf_comprehensive(file_path, threshold=0.3)
        
        # If not a leaf, return early with detailed feedback
        if not is_leaf:
            # Determine the most likely reason why it's not a leaf
            rejection_reason = "The image doesn't appear to contain a plant leaf"
            
            # Add more specific feedback based on confidence level
            if confidence < 10:
                rejection_reason = "This appears to be a document, screenshot, or text image, not a plant leaf"
            elif confidence < 20:
                rejection_reason = "The image doesn't have the natural color patterns and shapes of a plant leaf"
            
            return jsonify({
                "success": False,
                "error": "not_leaf",
                "message": rejection_reason,
                "confidence": float(confidence),
                "debug_image": f"/results/{debug_image}" if debug_image else None
            })
    else:
        # If bypassing validation, set default confidence
        confidence = 100.0
    
    try:
        # If we reach here, the image has passed validation (or validation was bypassed)
        # Continue with disease detection
        result_filename, prediction_result = generate_heatmap_image(file_path)
        
        if "error" in prediction_result:
            return jsonify({
                "success": False,
                "error": prediction_result["error"],
                "message": f"Error in disease analysis: {prediction_result['error']}"
            }), 500
        
        # Include leaf confidence in the response
        prediction_result["leaf_confidence"] = float(confidence)
        
        # Return successful response with prediction results
        return jsonify({
            "success": True,
            "prediction": prediction_result,
            "heatmap_url": f"/results/{result_filename}" if result_filename else None
        })
    
    except Exception as e:
        import traceback
        print(f"Error in prediction route: {e}")
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": "server_error",
            "message": f"Server error during processing: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True)