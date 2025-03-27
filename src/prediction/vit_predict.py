import tensorflow as tf
import numpy as np
import cv2
import os
import json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Import custom layers from vit_model.py
from src.training.vit_model import Patches, PatchEncoder

# Import the configuration
from src.utils.config import Paths

# Define paths
MODEL_PATH = Paths.VIT_MODEL
CLASS_INDICES_PATH = Paths.CLASS_INDICES
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

# Generate heatmap for visualization
def generate_heatmap(img_path):
    # Load image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    
    # Convert to HSV for better disease detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Extract saturation and value channels (usually high for diseased areas)
    s = hsv[:, :, 1].astype(float)
    v = hsv[:, :, 2].astype(float)
    
    # Create a simple heatmap highlighting potential disease areas
    # Higher saturation and lower value often indicate disease
    heatmap = s * (255 - v/2) / 255.0
    
    # Normalize and apply blur
    heatmap = heatmap / np.max(heatmap)
    heatmap = cv2.GaussianBlur(heatmap, (9, 9), 3)
    
    return heatmap

# Create visualization
def create_visualization(img_path, heatmap, threshold=0.6):
    # Load and resize original image
    original = cv2.imread(img_path)
    height, width = original.shape[:2]
    
    # Resize heatmap to match original image
    heatmap_resized = cv2.resize(heatmap, (width, height))
    
    # Create mask from heatmap
    heatmap_normalized = (heatmap_resized - np.min(heatmap_resized)) / (np.max(heatmap_resized) - np.min(heatmap_resized))
    mask = (heatmap_normalized > threshold).astype(np.uint8) * 255
    
    # Create colored overlay
    overlay = original.copy()
    overlay[mask > 0] = [0, 0, 255]  # Red color for affected areas
    
    # Blend images
    visualization = cv2.addWeighted(original, 0.7, overlay, 0.3, 0)
    
    # Calculate affected area percentage
    affected_pixels = np.sum(mask > 0)
    total_pixels = mask.shape[0] * mask.shape[1]
    affected_percentage = (affected_pixels / total_pixels) * 100
    
    return visualization, affected_percentage

# Main execution
if __name__ == "__main__":
    print("Leaf Disease Detection with Vision Transformer")
    print("-" * 50)
    
    # Load model
    try:
        print(f"Loading model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH, custom_objects={
            'Patches': Patches,
            'PatchEncoder': PatchEncoder
        })
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
    
    # Generate heatmap and visualization
    print("Generating disease heatmap...")
    heatmap = generate_heatmap(img_path)
    visualization, affected_area = create_visualization(img_path, heatmap)
    
    # Save results
    cv2.imwrite('disease_visualization.jpg', visualization)
    
    # Apply colormap to heatmap for display
    colored_heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_resized = cv2.resize(colored_heatmap, (visualization.shape[1], visualization.shape[0]))
    cv2.imwrite('disease_heatmap.jpg', heatmap_resized)
    
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
    print(f"Visualization saved as 'disease_visualization.jpg'")
    print(f"Heatmap saved as 'disease_heatmap.jpg'")
    
    # Optional: display images if running in interactive environment
    try:
        # Check if running in interactive environment
        if 'DISPLAY' in os.environ or os.name == 'nt':
            cv2.imshow('Disease Visualization', visualization)
            cv2.imshow('Disease Heatmap', heatmap_resized)
            print("\nPress any key to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except:
        print("Note: Interactive display not available.")