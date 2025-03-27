import os
import platform

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure paths relative to project root
class Paths:
    # Data paths
    DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "val")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    
    # Model paths
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    SAVED_MODELS_DIR = os.path.join(MODELS_DIR, "saved")
    CONFIG_DIR = os.path.join(MODELS_DIR, "config")
    
    # Output paths
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    UPLOADS_DIR = os.path.join(PROJECT_ROOT, "uploads")
    
    # Filenames
    CNN_MODEL = os.path.join(SAVED_MODELS_DIR, "cnn_leaf_disease_final.h5")
    VIT_MODEL = os.path.join(SAVED_MODELS_DIR, "vit_leaf_disease_final.h5")
    AUTOENCODER_MODEL = os.path.join(SAVED_MODELS_DIR, "classifier_dr_detection_multiclass.h5")
    CLASS_INDICES = os.path.join(CONFIG_DIR, "class_indices.json")
    
    # Create directories if they don't exist
    @classmethod
    def create_directories(cls):
        os.makedirs(cls.TRAIN_DIR, exist_ok=True)
        os.makedirs(cls.VAL_DIR, exist_ok=True)
        os.makedirs(cls.TEST_DIR, exist_ok=True)
        os.makedirs(cls.SAVED_MODELS_DIR, exist_ok=True)
        os.makedirs(cls.CONFIG_DIR, exist_ok=True)
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)
        os.makedirs(cls.UPLOADS_DIR, exist_ok=True)