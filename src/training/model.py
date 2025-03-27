import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Autoencoder Architecture
def build_autoencoder(input_shape=(128, 128, 3)):
    input_img = Input(shape=input_shape)
    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    # Decoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = Model(input_img, decoded)
    return autoencoder

# Load the autoencoder model
autoencoder = build_autoencoder(input_shape=(128, 128, 3))
autoencoder.compile(optimizer='adam', loss='mse')

# Image preprocessing for autoencoder
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_directory(
    'dataset', target_size=(128, 128), batch_size=32, class_mode='input', subset='training')
val_generator = datagen.flow_from_directory(
    'dataset', target_size=(128, 128), batch_size=32, class_mode='input', subset='validation')

# Train the autoencoder & store history
autoencoder_history = autoencoder.fit(train_generator, validation_data=val_generator, epochs=10)
autoencoder.save('autoencoder_dr_detection.h5')

# Plot Autoencoder Loss
plt.figure(figsize=(8, 5))
plt.plot(autoencoder_history.history['loss'], label='Train Loss', color='blue')
plt.plot(autoencoder_history.history['val_loss'], label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Autoencoder Training & Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Extract the encoder part for classification
encoder = Model(autoencoder.input, autoencoder.layers[4].output)

# Build the classifier on top of the encoder for four-class classification
def build_classifier(encoder):
    for layer in encoder.layers:
        layer.trainable = False  # Freeze encoder layers
    x = Flatten()(encoder.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)  # 10 classes, using softmax for multi-class classification
    classifier = Model(encoder.input, output)
    return classifier

classifier = build_classifier(encoder)
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Image preprocessing for classification (four classes)
train_gen_class = datagen.flow_from_directory(
    'dataset', target_size=(128, 128), batch_size=32, class_mode='categorical', subset='training')
val_gen_class = datagen.flow_from_directory(
    'dataset', target_size=(128, 128), batch_size=32, class_mode='categorical', subset='validation')

# Train the classifier & store history
classifier_history = classifier.fit(train_gen_class, validation_data=val_gen_class, epochs=50)
classifier.save('classifier_dr_detection_multiclass.h5')

# Plot Classifier Loss
plt.figure(figsize=(8, 5))
plt.plot(classifier_history.history['loss'], label='Train Loss', color='blue')
plt.plot(classifier_history.history['val_loss'], label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Classifier Training & Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

