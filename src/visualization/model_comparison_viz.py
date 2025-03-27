import os
# Set TensorFlow logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time
from matplotlib.ticker import PercentFormatter
import sys

# Add parent directory to path so we can import modules from other src directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.vit_model import Patches, PatchEncoder

# ===== CONFIGURATION =====
# Define paths to models and data with absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATHS = {
    'CNN': os.path.join(BASE_DIR+"/models/saved/", 'cnn_leaf_disease_final.h5'),
    'ViT': os.path.join(BASE_DIR+"/models/saved/", 'vit_leaf_disease_final.h5'),
    'Autoencoder': os.path.join(BASE_DIR+"/models/saved/", 'classifier_dr_detection_multiclass.h5')
}

# Update paths to use absolute paths
TEST_DIR = os.path.join(BASE_DIR, "dataset/val")  # Using val directory if test is not available
IMG_SIZE = (128, 128)
RESULTS_DIR = os.path.join(BASE_DIR, "results/comparisons")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load class indices
class_indices_path = os.path.join(BASE_DIR, "models/config/class_indices.json")
if not os.path.exists(class_indices_path):
    class_indices_path = os.path.join(BASE_DIR, "class_indices.json")

try:
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
        print(f"Loaded class indices from {class_indices_path}")
except FileNotFoundError:
    print(f"Class indices file not found at {class_indices_path}, using default classes")
    class_indices = {
        'Apple___Apple_scab': 0,
        'Apple___Black_rot': 1,
        'Apple___Cedar_apple_rust': 2,
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 3,
        'Corn_(maize)___Common_rust_': 4,
        'Grape___Esca_(Black_Measles)': 5,
        'Potato___Late_blight': 6,
        'Tomato___Bacterial_spot': 7,
        'Tomato___Early_blight': 8,
        'Tomato___Leaf_Mold': 9
    }

class_names = list(class_indices.keys())
idx_to_class = {v: k for k, v in class_indices.items()}

# ===== HELPER FUNCTIONS =====
def load_models():
    """Load all three models"""
    models = {}
    
    # Standard CNN model
    try:
        # Explicitly register loss functions before loading model
        print(f"Attempting to load CNN model from: {MODEL_PATHS['CNN']}")
        if os.path.exists(MODEL_PATHS['CNN']):
            models['CNN'] = load_model(MODEL_PATHS['CNN'], compile=False)
            # Recompile with explicit loss and metrics
            models['CNN'].compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            print("CNN model loaded successfully!")
        else:
            print(f"CNN model file not found at: {MODEL_PATHS['CNN']}")
    except Exception as e:
        print(f"Error loading CNN model: {e}")
    
    # ViT model with custom layers
    try:
        print(f"Attempting to load ViT model from: {MODEL_PATHS['ViT']}")
        if os.path.exists(MODEL_PATHS['ViT']):
            # Using imported Patches and PatchEncoder from vit_model.py
            models['ViT'] = load_model(MODEL_PATHS['ViT'], 
                                     custom_objects={
                                         'Patches': Patches,
                                         'PatchEncoder': PatchEncoder
                                     }, 
                                     compile=False)
            models['ViT'].compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            print("ViT model loaded successfully!")
        else:
            print(f"ViT model file not found at: {MODEL_PATHS['ViT']}")
    except Exception as e:
        print(f"Error loading ViT model: {e}")
    
    # Autoencoder model
    try:
        print(f"Attempting to load Autoencoder model from: {MODEL_PATHS['Autoencoder']}")
        if os.path.exists(MODEL_PATHS['Autoencoder']):
            # Custom MSE loss registration
            models['Autoencoder'] = load_model(MODEL_PATHS['Autoencoder'], compile=False)
            models['Autoencoder'].compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            print("Autoencoder model loaded successfully!")
        else:
            print(f"Autoencoder model file not found at: {MODEL_PATHS['Autoencoder']}")
    except Exception as e:
        print(f"Error loading Autoencoder model: {e}")
    
    return {k: v for k, v in models.items() if v is not None}

def preprocess_image(img_path, target_size=IMG_SIZE):
    """Preprocess a single image for prediction"""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def load_test_data(test_dir=TEST_DIR):
    """Load test images and their true labels"""
    X_test = []
    y_true = []
    file_paths = []
    
    print(f"Loading test data from {test_dir}...")
    
    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        class_idx = class_indices.get(class_name)
        if class_idx is None:
            print(f"Warning: Class {class_name} not found in class_indices")
            continue
            
        for img_file in os.listdir(class_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(class_dir, img_file)
            try:
                X_test.append(preprocess_image(img_path)[0])  # Remove batch dimension
                y_true.append(class_idx)
                file_paths.append(img_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    return np.array(X_test), np.array(y_true), file_paths

def evaluate_models(models, X_test, y_true):
    """Evaluate all models and collect metrics"""
    results = {model_name: {} for model_name in models}
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        
        # Measure prediction time
        start_time = time.time()
        y_pred_prob = model.predict(X_test)
        end_time = time.time()
        
        # Get predictions
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Store results
        results[model_name] = {
            'y_pred': y_pred,
            'y_pred_prob': y_pred_prob,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'inference_time': (end_time - start_time) / len(X_test),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        # Print summary
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Average inference time: {results[model_name]['inference_time']*1000:.2f} ms per image")
        
    return results

# ===== VISUALIZATION FUNCTIONS =====
def plot_accuracy_comparison(results):
    """Plot accuracy metrics comparison between models"""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    models = list(results.keys())
    
    data = {
        'Model': [],
        'Metric': [],
        'Value': []
    }
    
    for model in models:
        for metric in metrics:
            data['Model'].append(model)
            data['Metric'].append(metric.capitalize())
            data['Value'].append(results[model][metric])
    
    df = pd.DataFrame(data)
    
    # Create grouped bar chart
    plt.figure(figsize=(12, 8))
    
    ax = sns.barplot(x='Metric', y='Value', hue='Model', data=df)
    
    plt.title('Model Performance Comparison', fontsize=16)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0.7, 1.0)  # Adjust this depending on your results
    
    # Convert y-axis to percentage
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    plt.legend(title='Model Architecture')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(f"{RESULTS_DIR}/accuracy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrices(results, y_true):
    """Plot confusion matrices for all models"""
    for model_name, model_results in results.items():
        plt.figure(figsize=(12, 10))
        cm = model_results['confusion_matrix']
        
        # Use display class names (without disease__ prefix)
        display_names = [name.split('___')[-1].replace('_', ' ') for name in class_names]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=display_names,
                    yticklabels=display_names)
        
        plt.title(f'Confusion Matrix - {model_name}', fontsize=16)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(f"{RESULTS_DIR}/confusion_matrix_{model_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

def plot_inference_time_comparison(results):
    """Plot inference time comparison"""
    models = list(results.keys())
    inference_times = [results[model]['inference_time'] * 1000 for model in models]  # Convert to ms
    
    plt.figure(figsize=(10, 6))
    
    bars = plt.bar(models, inference_times, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    # Add time values on top of bars
    for bar, time in zip(bars, inference_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{time:.2f} ms', ha='center', va='bottom', fontsize=12)
    
    plt.title('Average Inference Time per Image', fontsize=16)
    plt.ylabel('Time (milliseconds)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(f"{RESULTS_DIR}/inference_time_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_per_class_accuracy(results, y_true):
    """Plot per-class accuracy for all models"""
    # Prepare data
    per_class_results = {}
    for model_name, model_results in results.items():
        y_pred = model_results['y_pred']
        
        # Calculate per-class accuracy
        class_accuracies = {}
        for class_idx in range(len(class_names)):
            mask = y_true == class_idx
            if np.sum(mask) > 0:  # Ensure at least one sample for this class
                class_accuracies[class_idx] = np.mean(y_pred[mask] == class_idx)
            else:
                class_accuracies[class_idx] = 0
                
        per_class_results[model_name] = class_accuracies
    
    # Create DataFrame
    data = {
        'Class': [],
        'Accuracy': [],
        'Model': []
    }
    
    for model_name, class_accuracies in per_class_results.items():
        for class_idx, acc in class_accuracies.items():
            # Use display class names (without disease__ prefix)
            display_name = idx_to_class[class_idx].split('___')[-1].replace('_', ' ')
            data['Class'].append(display_name)
            data['Accuracy'].append(acc)
            data['Model'].append(model_name)
    
    df = pd.DataFrame(data)
    
    # Plot per-class accuracy comparison
    plt.figure(figsize=(14, 8))
    
    ax = sns.barplot(x='Class', y='Accuracy', hue='Model', data=df)
    
    plt.title('Per-Class Accuracy Comparison', fontsize=16)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Disease Class', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Convert y-axis to percentage
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    plt.legend(title='Model Architecture')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(f"{RESULTS_DIR}/per_class_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_confidence_distribution(results):
    """Plot confidence distribution for correct and incorrect predictions"""
    plt.figure(figsize=(15, 10))
    
    for i, (model_name, model_results) in enumerate(results.items()):
        y_pred = model_results['y_pred']
        y_pred_prob = model_results['y_pred_prob']
        
        # Get confidence scores
        confidence_scores = np.max(y_pred_prob, axis=1)
        
        # Separate correct and incorrect predictions
        correct_mask = y_pred == y_true
        correct_conf = confidence_scores[correct_mask]
        incorrect_conf = confidence_scores[~correct_mask]
        
        # Plot density
        plt.subplot(1, 3, i+1)
        
        sns.kdeplot(correct_conf, fill=True, color='green', label='Correct', alpha=0.7)
        sns.kdeplot(incorrect_conf, fill=True, color='red', label='Incorrect', alpha=0.7)
        
        plt.title(f'{model_name} Confidence Distribution', fontsize=14)
        plt.xlabel('Confidence Score', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.grid(linestyle='--', alpha=0.7)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/confidence_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_composite_metrics(results):
    """Create a composite metrics visualization"""
    # Extract metrics for comparison
    models = list(results.keys())
    metrics = {
        'Accuracy': [results[model]['accuracy'] for model in models],
        'Precision': [results[model]['precision'] for model in models],
        'Recall': [results[model]['recall'] for model in models],
        'F1 Score': [results[model]['f1_score'] for model in models],
        'Inference Time (ms)': [results[model]['inference_time'] * 1000 for model in models]
    }
    
    # Create a radar chart (spider plot)
    categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    N = len(categories)
    
    # Create angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Scale time differently since it's not a percentage
    max_time = max(metrics['Inference Time (ms)'])
    normalized_time = [time/max_time for time in metrics['Inference Time (ms)']]
    
    # Plot each model
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, model in enumerate(models):
        values = [metrics['Accuracy'][i], metrics['Precision'][i], 
                  metrics['Recall'][i], metrics['F1 Score'][i]]
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Set category labels
    plt.xticks(angles[:-1], categories)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Model Metrics Comparison', fontsize=18)
    
    # Save the radar chart
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/model_metrics_radar.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a summary table as an image
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Include the inference time in the table
    data = []
    for model in models:
        data.append([
            f"{results[model]['accuracy']:.4f}",
            f"{results[model]['precision']:.4f}",
            f"{results[model]['recall']:.4f}",
            f"{results[model]['f1_score']:.4f}",
            f"{results[model]['inference_time'] * 1000:.2f} ms"
        ])
    
    table = ax.table(
        cellText=data,
        rowLabels=models,
        colLabels=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Inference Time'],
        loc='center',
        cellLoc='center',
        colWidths=[0.15, 0.15, 0.15, 0.15, 0.2]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    # Color code the table
    for i, model in enumerate(models):
        for j in range(5):
            table[(i+1, j)].set_facecolor(colors[i] + '40')  # Add transparency to color
    
    plt.title('Model Performance Summary', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/model_metrics_table.png", dpi=300, bbox_inches='tight')
    plt.close()

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    print("=" * 50)
    print("Leaf Disease Model Comparison Visualizations")
    print("=" * 50)
    
    # Load models
    print("\nLoading models...")
    models = load_models()
    
    if not models:
        print("No models could be loaded. Please check the model paths.")
        exit(1)
        
    print(f"Successfully loaded {len(models)} models: {', '.join(models.keys())}")
    
    # Load test data
    X_test, y_true, file_paths = load_test_data()
    
    if len(X_test) == 0:
        print(f"No test images found in {TEST_DIR}. Please check the directory path.")
        exit(1)
        
    print(f"Loaded {len(X_test)} test images")
    
    # Evaluate models
    print("\nEvaluating models...")
    results = evaluate_models(models, X_test, y_true)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Generate all plots
    plot_accuracy_comparison(results)
    print("✓ Generated accuracy comparison chart")
    
    plot_confusion_matrices(results, y_true)
    print("✓ Generated confusion matrices")
    
    plot_inference_time_comparison(results)
    print("✓ Generated inference time comparison")
    
    plot_per_class_accuracy(results, y_true)
    print("✓ Generated per-class accuracy comparison")
    
    plot_confidence_distribution(results)
    print("✓ Generated confidence distribution plots")
    
    plot_composite_metrics(results)
    print("✓ Generated composite metrics visualization")
    
    print(f"\nAll visualizations saved to {RESULTS_DIR}")
    print("\nDone! 🍃")