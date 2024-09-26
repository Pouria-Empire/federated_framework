import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import requests
import gzip
import csv

# URL for the MNIST dataset
BASE_URL = "http://yann.lecun.com/exdb/mnist/"
FILE_NAMES = {
    'train_images': 'train-images-idx3-ubyte.gz',
    'train_labels': 'train-labels-idx1-ubyte.gz',
}

def download_file(file_name):
    url = BASE_URL + file_name
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_name, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {file_name}")
    else:
        print(f"Failed to download {file_name}")

def load_mnist():
    for file_name in FILE_NAMES.values():
        if not os.path.exists(file_name):
            download_file(file_name)

    with gzip.open(FILE_NAMES['train_images'], 'rb') as f:
        train_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28, 1) / 255.0
    with gzip.open(FILE_NAMES['train_labels'], 'rb') as f:
        train_labels = np.frombuffer(f.read(), np.uint8, offset=8)

    return (train_images, train_labels)

def create_tinyml_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_save_model(client_id, images, labels):
    model = create_tinyml_model()
    model.fit(images, labels, epochs=5, verbose=1)  # Adjust epochs as needed

    # Save model weights as CSV
    weights = model.get_weights()
    weights_flat = np.concatenate([w.flatten() for w in weights])
    
    with open(f'client_{client_id}_weights.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(weights_flat)

if __name__ == "__main__":
    # Load MNIST dataset
    train_images, train_labels = load_mnist()
    
    # Train the model for this client
    client_id = 3  # Adjust for each client (e.g., 1, 2, 3)
    train_and_save_model(client_id, train_images, train_labels)
