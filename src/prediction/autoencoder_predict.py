import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os

# Import the configuration
from src.utils.config import Paths

# Load the trained classifier model
classifier = load_model(Paths.AUTOENCODER_MODEL)

# Define class labels and recommendations
class_labels = {
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
    img_array = preprocess_image(img_path)
    prediction = classifier.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return predicted_class

# Function to generate Grad-CAM heatmap
def grad_cam(img_path, model, layer_name="conv2d"):
    img_array = preprocess_image(img_path)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_output = conv_output[0]
    heatmap = np.mean(conv_output * pooled_grads, axis=-1)

    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1

    return heatmap

# Function to create a binary mask from the heatmap
def create_disease_mask(heatmap, img_shape, threshold=0.3):
    heatmap_resized = cv2.resize(heatmap, (img_shape[1], img_shape[0]))

    # Normalize heatmap for better contrast
    heatmap_resized = (heatmap_resized * 255).astype(np.uint8)
    heatmap_resized = cv2.equalizeHist(heatmap_resized)  # Enhance contrast

    # Apply threshold
    _, binary_mask = cv2.threshold(heatmap_resized, int(threshold * 255), 255, cv2.THRESH_BINARY)
    
    return binary_mask

# Function to overlay the mask on the image with a specific color
def overlay_mask_on_image(img, mask, color=(0, 0, 255)):
    color_mask = np.zeros_like(img)
    color_mask[mask == 255] = color
    overlayed_image = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)
    return overlayed_image

# Function to calculate affected leaf area percentage
def calculate_affected_percentage(mask, original_img):
    affected_pixels = np.sum(mask == 255)  # White pixels in the mask
    total_leaf_pixels = np.sum(original_img > 0)  # Non-black pixels in the original image
    
    if total_leaf_pixels == 0:
        return 0  # Avoid division by zero
    
    affected_percentage = (affected_pixels / total_leaf_pixels) * 100
    return round(affected_percentage, 2)

# Image path
img_path = '1 (108).jpg'

# Load original image
original_img = cv2.imread(img_path)

# Display the original image
cv2.imshow("Original Image", original_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Predict class
predicted_class = predict_image(img_path)
predicted_label = list(class_labels.keys())[predicted_class]

# Generate heatmap
heatmap = grad_cam(img_path, classifier, layer_name="conv2d")

# Create disease mask
disease_mask = create_disease_mask(heatmap, original_img.shape)

# Overlay mask on the original image
highlighted_img = overlay_mask_on_image(original_img, disease_mask)

# Calculate affected area percentage
affected_percentage = calculate_affected_percentage(disease_mask, original_img)

# Save and display the result
cv2.imwrite("highlighted_disease_areas.jpg", highlighted_img)
cv2.imshow("Highlighted Diseased Areas", highlighted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print results
recommendation = class_labels[predicted_label]
print(f"\n🔍 **Prediction:** {predicted_label}")
print(f"✅ **{recommendation}**")
print(f"📊 **Leaf Affected Area: {affected_percentage:.2f}%**")
