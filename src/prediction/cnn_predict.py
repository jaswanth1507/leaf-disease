import tensorflow as tf
import numpy as np
import cv2
import os
import json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Define paths
MODEL_PATH = 'cnn_leaf_disease_final.h5'
CLASS_INDICES_PATH = 'class_indices.json'
TEST_IMAGE_PATH = 'test_image.jpg'  # Change this to your test image path

# Load class indices
try:
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
        # Invert the dictionary to map indices to class names
        idx_to_class = {v: k for k, v in class_indices.items()}
except FileNotFoundError:
    print(f"Class indices file '{CLASS_INDICES_PATH}' not found.")
    print("Using default recommendations.")
    idx_to_class = {
        0: 'Apple___Apple_scab',
        1: 'Apple___Black_rot',
        2: 'Apple___Cedar_apple_rust',
        3: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        4: 'Corn_(maize)___Common_rust_',
        5: 'Grape___Esca_(Black_Measles)',
        6: 'Potato___Late_blight',
        7: 'Tomato___Bacterial_spot',
        8: 'Tomato___Early_blight',
        9: 'Tomato___Leaf_Mold'
    }

# Treatment recommendations for each disease
recommendations = {
    'Apple___Apple_scab': "Apply fungicides like Captan or Mancozeb. Prune infected leaves.",
    'Apple___Black_rot': "Remove infected fruit. Use fungicides like copper sprays.",
    'Apple___Cedar_apple_rust': "Use resistant apple varieties. Remove nearby juniper trees.",
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Rotate crops and apply foliar fungicides.",
    'Corn_(maize)___Common_rust_': "Use rust-resistant varieties and apply fungicides if needed.",
    'Grape___Esca_(Black_Measles)': "Prune infected vines and avoid excessive fertilization.",
    'Potato___Late_blight': "Remove infected plants and apply fungicides like chlorothalonil.",
    'Tomato___Bacterial_spot': "Avoid overhead watering and apply copper-based bactericides.",
    'Tomato___Early_blight': "Use mulch to prevent soil splash and apply fungicides early.",
    'Tomato___Leaf_Mold': "Increase air circulation and use resistant varieties."
}

# Image preprocessing
def preprocess_image(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

# Predict disease from image
def predict_disease(model, img_path):
    # Preprocess image
    img_array = preprocess_image(img_path)
    
    # Make prediction
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx] * 100
    
    # Get class name and recommendation
    class_name = idx_to_class.get(class_idx, f"Unknown Class {class_idx}")
    recommendation = recommendations.get(class_name, "No specific recommendation available.")
    
    return class_name, confidence, recommendation, img_array

# Generate feature map visualization
def generate_gradcam(model, img_path, layer_name='conv2d_4'):
    """
    Generate Grad-CAM visualization for the prediction
    """
    # Load and preprocess image
    img_array = preprocess_image(img_path)
    
    # Get the last convolutional layer
    last_conv_layer = model.get_layer(layer_name)
    
    # Create a model that outputs the last conv layer activation and the final output
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [last_conv_layer.output, model.output]
    )
    
    # Compute gradient of the top predicted class with respect to the last conv layer output
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        top_class_output = predictions[:, class_idx]
    
    # Gradient of the top class with respect to the conv layer output
    grads = tape.gradient(top_class_output, conv_outputs)
    
    # Remove dimension for batch size
    conv_outputs = conv_outputs[0]
    grads = grads[0]
    
    # Global average pooling
    weights = tf.reduce_mean(grads, axis=(0, 1))
    
    # Create a 2D array of the weighted sum of activation maps
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)
    
    # ReLU
    cam = tf.maximum(cam, 0)
    
    # Normalize
    cam = cam / tf.math.reduce_max(cam)
    
    # Resize to original image size
    cam = tf.expand_dims(cam, axis=-1)
    cam = tf.image.resize(cam, (128, 128))
    
    return cam.numpy()

# Create visualization
def create_visualization(img_path, heatmap):
    # Load and resize original image
    original = cv2.imread(img_path)
    original = cv2.resize(original, (128, 128))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose heatmap on original image
    superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    
    # Calculate affected area percentage
    # (Considering pixels above 50% in normalized heatmap as affected)
    affected_pixels = np.sum(heatmap[:, :, 0] > 127)
    total_pixels = heatmap.shape[0] * heatmap.shape[1]
    affected_percentage = (affected_pixels / total_pixels) * 100
    
    return superimposed, affected_percentage

# Main execution
if __name__ == "__main__":
    print("Leaf Disease Detection with CNN")
    print("-" * 50)
    
    # Load model
    try:
        print(f"Loading model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)
    
    # Get image path from user if not provided
    img_path = TEST_IMAGE_PATH
    if not os.path.exists(img_path):
        img_path = input("Enter path to leaf image: ")
        if not os.path.exists(img_path):
            print(f"Error: File {img_path} not found.")
            exit(1)
    
    # Predict disease
    print(f"Analyzing image: {img_path}")
    disease_class, confidence, treatment, _ = predict_disease(model, img_path)
    
    # Generate Grad-CAM visualization
    print("Generating disease heatmap...")
    heatmap = generate_gradcam(model, img_path)
    visualization, affected_area = create_visualization(img_path, heatmap)
    
    # Save results
    cv2.imwrite('cnn_disease_visualization.jpg', visualization)
    
    # Display results
    print("\nResults:")
    print(f"Detected Disease: {disease_class}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Affected Area: {affected_area:.2f}%")
    print("\nTreatment Recommendation:")
    print(treatment)
    
    # Severity assessment
    severity = "Low"
    if affected_area > 15:
        severity = "Medium"
    if affected_area > 30:
        severity = "High"
    
    print(f"\nDisease Severity: {severity}")
    print(f"Visualization saved as 'cnn_disease_visualization.jpg'")
    
    # Optional: display images if running in interactive environment
    try:
        # Check if running in interactive environment
        if 'DISPLAY' in os.environ or os.name == 'nt':
            cv2.imshow('Disease Visualization', visualization)
            print("\nPress any key to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except:
        print("Note: Interactive display not available.")