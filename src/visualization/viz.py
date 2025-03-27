# comparison.py

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tensorflow.keras.preprocessing import image

# Import the configuration
from src.utils.config import Paths

# -------------------- SETUP --------------------
IMG_SIZE = (128, 128)
TEST_DIR = Paths.VAL_DIR

# Load class indices
with open(Paths.CLASS_INDICES, 'r') as f:
    class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}

# Model paths
cnn_model_path = Paths.CNN_MODEL
vit_model_path = Paths.VIT_MODEL
autoencoder_model_path = Paths.AUTOENCODER_MODEL

# Load models
cnn_model = load_model(cnn_model_path)
vit_model = load_model(vit_model_path, custom_objects={
    'Patches': __import__('src.training.vit_model').Patches,
    'PatchEncoder': __import__('src.training.vit_model').PatchEncoder
})
autoencoder_model = load_model(autoencoder_model_path)

models = {
    "CNN": cnn_model,
    "ViT": vit_model,
    "Autoencoder": autoencoder_model
}

# -------------------- IMAGE LOADER --------------------
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

# -------------------- EVALUATION --------------------
y_true = []
preds_by_model = {model_name: [] for model_name in models}
times_by_model = {model_name: [] for model_name in models}
confidences_by_model = {model_name: [] for model_name in models}

# Scan test images
for class_name in os.listdir(TEST_DIR):
    class_dir = os.path.join(TEST_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue
    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)
        try:
            img_array = preprocess_image(img_path)
            y_true.append(list(class_indices.keys()).index(class_name))

            for model_name, model in models.items():
                start_time = time.time()
                pred = model.predict(img_array)
                end_time = time.time()

                class_id = np.argmax(pred[0])
                confidence = pred[0][class_id]

                preds_by_model[model_name].append(class_id)
                confidences_by_model[model_name].append(confidence)
                times_by_model[model_name].append(end_time - start_time)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# -------------------- METRIC COMPARISON --------------------
for model_name in models:
    print(f"\n===== {model_name} Evaluation =====")
    print(classification_report(y_true, preds_by_model[model_name], target_names=list(class_indices.keys())))

    cm = confusion_matrix(y_true, preds_by_model[model_name])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_indices.keys(), yticklabels=class_indices.keys(), cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{model_name}.png")
    plt.show()
    plt.close()

# -------------------- AVG CONFIDENCE & TIME --------------------
avg_conf = {m: np.mean(confidences_by_model[m]) for m in models}
avg_time = {m: np.mean(times_by_model[m]) for m in models}

# Plot comparison
plt.figure(figsize=(10, 5))
plt.bar(avg_conf.keys(), avg_conf.values(), color='green')
plt.title("Average Confidence per Model")
plt.ylabel("Confidence")
plt.savefig("average_confidence_comparison.png")
plt.show()
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(avg_time.keys(), avg_time.values(), color='orange')
plt.title("Average Inference Time per Model")
plt.ylabel("Time (s)")
plt.savefig("average_time_comparison.png")
plt.show()
plt.close()

print("\nAll comparisons done. Confusion matrices and charts saved.")
