import mlflow
from mlflow.tracking import MlflowClient
import os
import shutil
import sys

# Dagshub configuration
DAGSHUB_USER = os.getenv("DAGSHUB_USERNAME")
REPO_NAME = "Real-Time-Predictive-System"
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
MODEL_NAME = "Stock_Sentiment_Predictor"

# Set up MLflow tracking URI and authentication
mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USER}/{REPO_NAME}.mlflow")
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

def main():
    client = MlflowClient()  # Initialize MLflow client

    print(f"🔍 Looking for @champion alias of {MODEL_NAME}.")
    try:
        model_version = client.get_model_version_by_alias(MODEL_NAME, "champion")  # Fetch champion model version
        print(f"✅ Found Champion: Version {model_version.version}")

        source_uri = model_version.source  # Get the source URI of the model artifact

        local_path = "inference/model_dir"  # Local directory to download the model
        if os.path.exists(local_path):
            shutil.rmtree(local_path)  # Remove existing directory if it exists

        print(f"⬇️ Downloading artifact from {source_uri}.")
        mlflow.artifacts.download_artifacts(artifact_uri=source_uri, dst_path=local_path)  # Download the model artifact
        print("✅ Model downloaded to 'model_dir/'")

    except Exception as e:  # Handle exceptions during model fetching
        print(f"❌ Failed to fetch champion model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
