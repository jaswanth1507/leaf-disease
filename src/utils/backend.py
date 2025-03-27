from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os
import uuid
from werkzeug.utils import secure_filename

from flask_cors import CORS
app = Flask(__name__, static_folder='static')
CORS(app)  # This enables CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load the trained classifier model
try:
    classifier = load_model('classifier_dr_detection_multiclass.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    classifier = None

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
    return render_template('index.html')

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
    
    result_filename, prediction_result = generate_heatmap_image(file_path)
    
    if "error" in prediction_result:
        return jsonify(prediction_result), 500
    
    return jsonify({
        "success": True,
        "prediction": prediction_result,
        "heatmap_url": f"/results/{result_filename}" if result_filename else None
    })

if __name__ == '__main__':
    app.run(debug=True)