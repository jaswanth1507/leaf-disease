import os

class Paths:
    # Base directories
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    SAVED_MODELS_DIR = os.path.join(MODELS_DIR, 'saved')
    UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    
    # Model paths - update these to match your actual filenames
    AUTOENCODER_MODEL = os.path.join(SAVED_MODELS_DIR, 'autoencoder_dr_detection.h5')
    CNN_MODEL = os.path.join(SAVED_MODELS_DIR, 'cnn_leaf_disease_final.h5')
    VIT_MODEL = os.path.join(SAVED_MODELS_DIR, 'vit_leaf_disease_final.h5')
    
    @staticmethod
    def create_directories():
        """Creates necessary directories if they don't exist."""
        for path in [Paths.DATA_DIR, Paths.MODELS_DIR, Paths.SAVED_MODELS_DIR, 
                    Paths.UPLOADS_DIR, Paths.RESULTS_DIR]:
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"Created directory: {path}")