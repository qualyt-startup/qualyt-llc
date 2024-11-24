import tensorflow as tf

def convert_to_tflite(saved_model_dir, tflite_model_path):
    """
    Converts the saved model to TensorFlow Lite format.
    Args:
        saved_model_dir (str): Directory where the model is saved.
        tflite_model_path (str): Path where the TensorFlow Lite model will be saved.
    """
    # Load the saved model
    model = tf.keras.models.load_model(saved_model_dir)

    # Convert the model to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"Converted model saved to {tflite_model_path}")

# Usage:
# convert_to_tflite('/path/to/saved_model', '/path/to/output/model.tflite')
