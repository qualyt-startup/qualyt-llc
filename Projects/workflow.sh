#!/bin/bash

# Set Google Cloud project and region
PROJECT_ID="qualyt-llc"
MODEL_NAME="marco-o1"
VERSION_NAME="v1"
REGION="us-central1"
BUCKET_NAME="qualyt-llc-bucket"
SAVED_MODEL_DIR="/Users/qualytwork/qualyt-llc-project/saved_model_directory"
TFLITE_MODEL_PATH="/Users/qualytwork/qualyt-llc-project/model_optimized.tflite"

# Step 1: Activate virtual environment
echo "Activating virtual environment..."
source /path/to/qualyt-llc-project/venv/bin/activate

# Step 2: Install required packages
echo "Installing dependencies..."
pip install -r /path/to/qualyt-llc-project/requirements.txt

# Step 3: Distributed training using TensorFlow
echo "Starting distributed training with TensorFlow on multiple GPUs..."
python /path/to/qualyt-llc-project/train_model.py --distributed --gpus=4

# Step 4: Convert model to TensorFlow Lite format
echo "Converting model to TensorFlow Lite..."
python /path/to/qualyt-llc-project/convert_to_tflite.py --saved_model_dir="$SAVED_MODEL_DIR" --tflite_model_path="$TFLITE_MODEL_PATH"

# Step 5: Upload model to Google Cloud Storage
echo "Uploading TensorFlow Lite model to Google Cloud Storage..."
gsutil cp "$TFLITE_MODEL_PATH" gs://"$BUCKET_NAME"/models/"$MODEL_NAME"/

# Step 6: Deploy model to Google Cloud AI Platform
echo "Deploying TensorFlow Lite model to AI Platform..."
gcloud ai-platform models upload \
    --name="$MODEL_NAME" \
    --region="$REGION" \
    --model-dir="gs://$BUCKET_NAME/models/$MODEL_NAME/"

gcloud ai-platform versions create "$VERSION_NAME" \
    --model="$MODEL_NAME" \
    --origin="gs://$BUCKET_NAME/models/$MODEL_NAME/" \
    --runtime-version=2.11 \
    --framework=TENSORFLOW \
    --python-version=3.8 \
    --region="$REGION"

# Step 7: Set up Google Cloud Function for retraining model
echo "Setting up Cloud Function for automated retraining..."
gcloud functions deploy retrain_model \
    --runtime python39 \
    --trigger-resource "gs://$BUCKET_NAME/data/" \
    --trigger-event google.storage.object.finalize \
    --entry-point retrain_model \
    --project="$PROJECT_ID"

# Step 8: Deploy Prometheus and Grafana on Kubernetes
echo "Setting up Kubernetes for Prometheus and Grafana..."
kubectl apply -f /path/to/qualyt-llc-project/prometheus-deployment.yaml
kubectl apply -f /path/to/qualyt-llc-project/prometheus-service.yaml
kubectl apply -f /path/to/qualyt-llc-project/grafana-deployment.yaml
kubectl apply -f /path/to/qualyt-llc-project/grafana-service.yaml
kubectl apply -f /path/to/qualyt-llc-project/prometheus-configmap.yaml

# Step 9: Start Prometheus and Grafana servers
echo "Starting Prometheus server..."
start_http_server 9090
echo "Starting Grafana server..."
start_http_server 3000

echo "Workflow complete!"
