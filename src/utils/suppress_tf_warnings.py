import os
import tensorflow as tf
import logging

def suppress_tf_warnings():
    """Suppresses common TensorFlow warnings."""
    # Set TensorFlow logging level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
    
    # Disable TensorFlow plugin registration warnings
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    
    # Disable CUDA warnings
    tf.get_logger().setLevel('ERROR')
    
    # Disable ABSL logging
    logging.getLogger('absl').setLevel(logging.ERROR)
    
    # Suppress other specific warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)