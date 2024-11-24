from google.cloud import storage
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    """
    Preprocess the training data.
    Assumes the data is in CSV format. Customize the parsing based on your actual data format.
    Args:
        data (str): The raw CSV data as a string.
    Returns:
        np.array: The preprocessed features as a numpy array.
    """
    # Convert the data string into a pandas dataframe
    df = pd.read_csv(pd.compat.StringIO(data))  # Assuming the data is in CSV format

    # Example: drop columns that are not needed
    df = df.drop(columns=['unnecessary_column1', 'unnecessary_column2'], errors='ignore')

    # Standardize the feature columns (assuming all columns except the last one are features)
    features = df.iloc[:, :-1]  # All columns except the last are features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return features

def preprocess_labels(data):
    """
    Preprocess the labels from the incoming data.
    Assumes the label is the last column in the dataset.
    Args:
        data (str): The raw CSV data as a string.
    Returns:
        np.array: The preprocessed labels as a numpy array.
    """
    # Convert the data string into a pandas dataframe
    df = pd.read_csv(pd.compat.StringIO(data))

    # Extract the labels (assuming the label is in the last column)
    labels = df.iloc[:, -1].values

    return labels

def retrain_model(event, context):
    """
    Triggered by a change to a Cloud Storage bucket. Retrains the model when new data is uploaded.
    Args:
        event (dict): The event data.
        context (google.cloud.functions.Context): The metadata of the event.
    """
    # Define the model directory and path to the saved model
    model_dir = '/Users/qualytwork/qualyt-llc-project/saved_model_directory'
    bucket_name = event['bucket']
    file_name = event['name']

    # Initialize the Google Cloud Storage client
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Download the new training data from Cloud Storage
    data = blob.download_as_text()

    # Load the model
    model = tf.keras.models.load_model(model_dir)

    # Preprocess the data and labels
    X_train = preprocess_data(data)
    y_train = preprocess_labels(data)

    # Retrain the model with the new data
    print(f"Retraining model with {len(X_train)} samples.")
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Save the retrained model
    model.save(model_dir)
    print(f"Model retrained and saved at {model_dir}")
