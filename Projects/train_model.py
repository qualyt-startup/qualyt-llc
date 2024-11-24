import tensorflow as tf
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def create_model():
    """
    Creates and compiles the model.
    Returns:
        model: A compiled Keras model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    """
    Train the model using distributed strategy with multiple GPUs.
    """
    # Create the MirroredStrategy for distributed training across GPUs
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    # Prepare the dataset
    X, y = make_classification(n_samples=10000, n_features=20, n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create and compile the model within the strategy scope
    with strategy.scope():
        model = create_model()

    # Train the model
    model.fit(X_train, y_train, epochs=5, batch_size=64)
    
    # Evaluate the model
    model.evaluate(X_test, y_test)
    model.save('/path/to/saved_model_directory')

if __name__ == "__main__":
    train_model()
