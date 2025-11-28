import os
import json
import requests
import pandas as pd
import pendulum
import subprocess
import mlflow
import shutil
from datetime import datetime, timedelta
from textblob import TextBlob
from pathlib import Path
from airflow.models import Variable
from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator
from airflow.exceptions import AirflowFailException, AirflowSkipException
from ydata_profiling import ProfileReport
from io import StringIO

# Imports for model training
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.tracking import MlflowClient

DAG_ID = "stock_news_pipeline"
RAW_DATA_PATH = "data/raw/daily_news.json" # Raw JSON data from API
PROCESSED_DATA_PATH = "data/processed/daily_news.csv" # Cleaned + Merged CSV
REPORT_PATH = "data/reports/daily_quality_report.html" # Profiling Report

# Load Secrets from Airflow Variables
# GNews API Key
GNEWS_API_KEY = Variable.get("gnews_api_key")

# Dagshub Access Credentials for DVC
DAGSHUB_ACCESS = Variable.get("dagshub_access_key")
DAGSHUB_SECRET = Variable.get("dagshub_secret_key")

# Dagshub User Credentials for MLflow
DAGSHUB_USER = Variable.get("dagshub_username")
DAGSHUB_TOKEN = Variable.get("dagshub_token")
REPO_NAME = "Real-Time-Predictive-System" # Dagshub Repository Name for MLflow tracking

# GitHub Credentials for pushing commits
GITHUB_TOKEN = Variable.get("github_token")
GITHUB_USER = Variable.get("github_username")

GITHUB_REPO = "Real-Time-Predictive-System-ETL-Pipeline" # GitHub Repository Name for pushing commits

DVC_ENV = { # DVC Environment Variables for BashOperator
    "AWS_ACCESS_KEY_ID": DAGSHUB_ACCESS,
    "AWS_SECRET_ACCESS_KEY": DAGSHUB_SECRET,
    "AWS_REGION": "us-east-1",
}

@dag( # Define the DAG
    dag_id=DAG_ID,
    start_date=pendulum.datetime(2025, 11, 24, tz="UTC"),
    schedule="@daily", # Runs every day at midnight UTC
    catchup=False, # No backfilling
    doc_md="DAG for fetching daily technology/stock-related news, profiling and versioning",
)
def stock_news_pipeline(): # Main DAG function
    
    @task(retries=0) # No retries on failure
    def extract_live_data(**kwargs) -> str: # Extract data from GNews API and perform quality checks
        execution_date = kwargs.get('ds') # Get execution date in 'YYYY-MM-DD' format

        target_date = datetime.strptime(execution_date, '%Y-%m-%d') - timedelta(days=1) # Fetch previous day's date
        #str_date = target_date.strftime('%Y-%m-%dT00:00:00Z') # Start of day
        #end_date = target_date.strftime('%Y-%m-%dT23:59:59Z') # End of day

        #hardcode dates
        str_date = "2025-11-18T00:00:00Z"
        end_date = "2025-11-18T23:59:59Z"

        url = ( # GNews API endpoint for technology news
            f"https://gnews.io/api/v4/search?q=technology&from={str_date}&to={end_date}"
            f"&max=10&lang=en&apikey={GNEWS_API_KEY}"
        )
        print(f"Fetching from: {url}")

        try:
            response = requests.get(url) # Make API request
            data = response.json() # Parse JSON response
        except Exception as e: # Handle connection errors
            raise AirflowFailException(f"API Connection Failed: {str(e)}")

        if response.status_code != 200: # Check for API errors
            raise AirflowFailException(f"API Error detected: {response.text}")

        articles = data.get("articles", []) # Extract articles list

        if not articles: # Quality check: Ensure articles are returned
            raise AirflowFailException(f"QUALITY CHECK FAILED: No articles returned for {execution_date}.")

        validated_articles = [] # List to hold validated articles
        required_fields = ['title', 'description', 'publishedAt', 'source'] # Required fields in each article
        
        for article in articles: # Validate each article
            for field in required_fields: # Check each required field
                if field not in article or article[field] is None: # Missing or null field
                    raise AirflowFailException(f"QUALITY CHECK FAILED: Missing or null field '{field}' in one of the articles.")
            
            article['collection_time'] = datetime.now().isoformat() # Add collection timestamp
            validated_articles.append(article) # Add to validated list
        
        payload = { # Prepare payload for next task
            "articles": validated_articles,
            "period_start": str_date,
            "period_end": end_date
        }

        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True) # Ensure directory exists
        with open(RAW_DATA_PATH, 'w') as f:
            json.dump(validated_articles, f, indent=4) # Save raw data to JSON file

        print(f"Extracted {len(validated_articles)} articles. Saved to {RAW_DATA_PATH}")

        return json.dumps(payload) # Return payload as JSON string for next task
    
    @task
    def pull_dvc_history() -> str: # Pull historical data from DVC
        tmp_dir = "/tmp/repo_pull" # Temporary directory for cloning repo
        if os.path.exists(tmp_dir): shutil.rmtree(tmp_dir) # Clean up existing temp dir
        
        repo_url = f"https://{GITHUB_USER}:{GITHUB_TOKEN}@github.com/{GITHUB_USER}/{GITHUB_REPO}.git" # GitHub repo URL
        subprocess.run(["git", "clone", repo_url, tmp_dir], check=True) # Clone repo
        
        commands = (
            "dvc remote add -d -f origin s3://dvc && " # Add DVC remote storage
            f"dvc remote modify origin endpointurl https://dagshub.com/{DAGSHUB_USER}/{REPO_NAME}.s3 && " # Set endpoint URL
            f"dvc pull {PROCESSED_DATA_PATH}" # Pull processed data file
        )
        
        try:
            subprocess.run(commands, cwd=tmp_dir, env=DVC_ENV, check=True, shell=True, executable="/bin/bash") # Run DVC commands
            
            csv_path = os.path.join(tmp_dir, PROCESSED_DATA_PATH) # Path to pulled CSV file
            
            if os.path.exists(csv_path): # Check if file exists
                df = pd.read_csv(csv_path) # Load CSV into DataFrame
                print(f"Pulled history: {len(df)} rows") # Log number of rows pulled
                return df.to_json(orient='split', date_format='iso')  # Return DataFrame as JSON string
        except subprocess.CalledProcessError as e: # Handle DVC pull errors
            print(f"DVC Pull failed (Likely first run or file missing): {e}")
        except Exception as e: # Handle other errors
            print(f"Unexpected error in pull history: {e}")
        
        return "{}" # Return empty JSON if no history
    
    @task
    def transform_and_profile(payload_json: str, history_json: str, **kwargs) -> str: # Transform data and generate profiling report
        #print history_json
        print(f"History: {history_json[:100]}...")  # Print first 100 characters of history JSON for debugging

        payload = json.loads(payload_json) # Parse JSON string back to dict
        raw_data = payload.get("articles") # Extract articles
        
        # Get start and end dates for reporting
        str_date = payload.get("period_start")
        end_date = payload.get("period_end")
        
        df = pd.DataFrame(raw_data) # Load articles into DataFrame

        df['publishedAt'] = pd.to_datetime(df['publishedAt'], utc=True) # Convert to datetime
        df['hour_of_day'] = df['publishedAt'].dt.hour # Extract hour
        df['day_of_week'] = df['publishedAt'].dt.dayofweek # Extract day of week
        df['source_name'] = df['source'].apply(lambda x: x.get('name') if isinstance(x, dict) else None) # Extract source name

        def get_sentiment(text):
            return TextBlob(str(text)).sentiment.polarity # Sentiment analysis using TextBlob

        df['title_sentiment'] = df['title'].apply(get_sentiment) # Sentiment for title
        df['desc_sentiment'] = df['description'].apply(get_sentiment) # Sentiment for description
        df['content_sentiment'] = df['content'].apply(get_sentiment) # Sentiment for content

        df = df.drop(columns=['image', 'url', 'source', 'id']) # Drop unnecessary columns

        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True) # Ensure report directory exists

        report_period = f"{str_date.replace('T', ' ').replace('Z', '')} to {end_date.replace('T', ' ').replace('Z', '')}" # Human-readable period

        report_title = f"Data Quality Report ({report_period})" # Report title
        
        profile = ProfileReport(df, title=report_title, minimal=True) # Generate profiling report
        profile.to_file(REPORT_PATH) # Save report to HTML file
        print(f"Quality Report generated at {REPORT_PATH}")

        mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USER}/{REPO_NAME}.mlflow") # Set MLflow tracking URI
        os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER # Set MLflow username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN # Set MLflow token

        with mlflow.start_run(run_name=f"Daily_Report_{kwargs.get('ds')}"): # Start MLflow run
            mlflow.log_artifact(REPORT_PATH) # Log profiling report as artifact
            
            mlflow.log_param("execution_date", kwargs.get('ds')) # Log execution date
            mlflow.log_param("period_start", str_date) # Log period start
            mlflow.log_param("period_end", end_date) # Log period end
            mlflow.log_param("report_type", "minimal") # Log report type
            
            mlflow.set_tag("meta_period_start", str_date) # Set meta tag for period start
            mlflow.set_tag("meta_period_end", end_date) # Set meta tag for period end
            
            avg_sentiment = df['title_sentiment'].mean() # Average sentiment score
            row_count = len(df) # Number of rows/articles
            unique_sources = df['source_name'].nunique() # Number of unique sources
            
            mlflow.log_metric("avg_sentiment", avg_sentiment) # Log average sentiment
            mlflow.log_metric("row_count", row_count) # Log row count
            mlflow.log_metric("unique_sources", unique_sources) # Log unique sources count
            
            print("Logged artifacts and metrics to MLflow.")
        
        history_len = 0        
        if history_json and history_json != "{}": # If historical data exists
            try: # Attempt to parse historical data
                df_history = pd.read_json(history_json, orient='split') # Load historical data
                
                df_history['publishedAt'] = pd.to_datetime(df_history['publishedAt'], utc=True) # Ensure datetime format
                
                history_len = len(df_history) # Get length of historical data
                print(f"Loaded history from XCom: {history_len} rows")
                
                df_combined = pd.concat([df_history, df]) # Combine historical and new data
            except ValueError as e: # Handle JSON parsing errors
                print(f"Error parsing history JSON: {e}. Starting fresh.")
                df_combined = df # Start fresh if error occurs
        else: # No historical data
            print("No history found (fresh start)")
            df_combined = df # Use current data as combined data

        df_combined['dedupe_date'] = df_combined['publishedAt'].astype(str) # Temporary column for deduplication
        
        df_combined = df_combined.drop_duplicates(subset=['title', 'dedupe_date'], keep='last') # Deduplicate based on title and published date
        
        df_combined = df_combined.drop(columns=['dedupe_date']) # Remove temporary deduplication column
        
        if len(df_combined) <= history_len and history_len > 0: # No new unique data
            print("Duplication check complete: No new unique data found.")
            raise AirflowSkipException("Data is identical to history. Skipping write and push.") # Skip this task and downstream tasks
        
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True) # Ensure directory exists
        
        df_combined.to_csv(PROCESSED_DATA_PATH, index=False) # Save combined data to CSV file
        
        new_rows_count = len(df_combined) - history_len # Calculate number of new rows added
        print(f"Update detected! Total rows: {len(df_combined)} (Added {new_rows_count} new rows)")

        return df_combined.to_json(orient='split', date_format='iso') # Return merged data as JSON string

    @task
    def dvc_add_and_push(merged_json: str) -> str: # Version data with DVC
        df = pd.read_json(StringIO(merged_json), orient='split') # Load merged data from JSON string 
        
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True) # Ensure directory exists
        df.to_csv(PROCESSED_DATA_PATH, index=False) # Save merged data to CSV file
        
        commands = (
            "dvc init --no-scm && " # Initialize DVC without Git
            "dvc remote add -d origin s3://dvc && " # Add DVC remote storage
            f"dvc remote modify origin endpointurl https://dagshub.com/{DAGSHUB_USER}/{REPO_NAME}.s3 && " # Set endpoint URL
            f"dvc add {PROCESSED_DATA_PATH} && " # Add processed data to DVC tracking
            "dvc push" # Push data to DVC remote storage
        )
        
        subprocess.run(commands, shell=True, check=True, env=DVC_ENV, executable="/bin/bash") # Run DVC commands
        
        dvc_path = f"{PROCESSED_DATA_PATH}.dvc" # Path to DVC file

        with open(dvc_path, 'r') as f: # Read DVC file content
            dvc_content = f.read() # Store DVC file content
            
        return dvc_content # Return DVC file content for Git commit

    @task
    def git_commit_and_push(dvc_content: str, **kwargs): # Commit and push changes to GitHub
        tmp_dir = "/tmp/repo_git" # Temporary directory for Git operations
        if os.path.exists(tmp_dir): shutil.rmtree(tmp_dir) # Clean up existing temp dir
        
        repo_url = f"https://{GITHUB_USER}:{GITHUB_TOKEN}@github.com/{GITHUB_USER}/{GITHUB_REPO}.git" # GitHub repo URL
        subprocess.run(["git", "clone", repo_url, tmp_dir], check=True) # Clone repo
        
        dvc_file_path = os.path.join(tmp_dir, f"{PROCESSED_DATA_PATH}.dvc") # Path to DVC file in cloned repo
        os.makedirs(os.path.dirname(dvc_file_path), exist_ok=True) # Ensure directory exists
        
        with open(dvc_file_path, 'w') as f: # Write DVC content to file
            f.write(dvc_content) # Write DVC file content
            
        cwd = tmp_dir # Set current working directory for Git commands
        
        # Configure Git user detail
        subprocess.run(["git", "config", "user.email", "baseersoomro2013@gmail.com"], cwd=cwd, check=True)
        subprocess.run(["git", "config", "user.name", "Noor Ul Baseer (Airflow)"], cwd=cwd, check=True)
        
        subprocess.run(["git", "add", "."], cwd=cwd, check=True) # Stage all changes
        
        subprocess.run(["git", "commit", "-m", f"ETL Update: {kwargs.get('ds')}"], cwd=cwd, check=False) # Commit changes with message
        
        subprocess.run(["git", "push", "origin", "master"], cwd=cwd, check=True) # Push changes to remote repository
        
        print("Git push successful.")
    
    @task
    def train_model(dvc_content: str, **kwargs): # Train and log model with MLflow
        if not os.path.exists(PROCESSED_DATA_PATH): # Check if processed data exists
            raise AirflowSkipException("No processed data found to train on.")
        
        with open(PROCESSED_DATA_PATH, 'r') as f: # Print the first 500 characters of the CSV for debugging
            print(f"Processed Data Preview:\n{f.read(500)}")
            
        df = pd.read_csv(PROCESSED_DATA_PATH) # Load processed data
        
        if len(df) < 5: # Ensure enough data to train (at least 5 rows)
            print("Not enough data to train (need at least 5 rows). Skipping training.")
            return "Skipped"

        features = ['hour_of_day', 'day_of_week', 'title_sentiment', 'desc_sentiment', 'content_sentiment'] # Feature columns for training
        X = df[features] # Feature matrix for training
        
        # Create synthetic target variable for demonstration purposes
        df['market_change'] = (df['title_sentiment'] + df['content_sentiment']) * 10 # Synthetic target variable
        y = df['market_change'] # Target variable

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Train-test split
        
        # Set up MLflow tracking
        mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USER}/{REPO_NAME}.mlflow") # Set MLflow tracking URI
        os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER # Set MLflow username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN # Set MLflow token

        mlflow.set_experiment("Stock_Price_Prediction") # Set MLflow experiment

        # Train RandomForestRegressor model
        with mlflow.start_run(run_name=f"Train_{kwargs.get('ds')}") as run: # Start MLflow run
            # Hyperparameters
            n_estimators = 100 # Number of trees in the forest
            max_depth = 10 # Maximum depth of the tree
            
            # Log hyperparameters
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42) # Initialize model with hyperparameters
            model.fit(X_train, y_train) # Train model on training data

            # Evaluate model performance on test data
            predictions = model.predict(X_test) # Make predictions on test data
            rmse = np.sqrt(mean_squared_error(y_test, predictions)) # Calculate RMSE (Root Mean Squared Error)
            mae = mean_absolute_error(y_test, predictions) # Calculate MAE (Mean Absolute Error)
            r2 = r2_score(y_test, predictions) # Calculate R2 score (Coefficient of Determination)

            # Log evaluation metrics
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)
            
            print(f"Training Complete. RMSE: {rmse}")
            
            # Save and log the trained model using joblib
            model_filename = "stock_sentiment_model.pkl" # Model filename 
            joblib.dump(model, model_filename) # Save model to file
            mlflow.log_artifact(model_filename, artifact_path="model") # Log model file as MLflow artifact
            print(f"Model logged successfully as {model_filename}")
            
            # Manual model registration in MLflow Model Registry
            try: # Attempt to register the model
                client = MlflowClient() # Initialize MLflow client
                registered_name = "Stock_Sentiment_Predictor" # Registered model name
                
                try: # Try to create a new registered model
                    client.create_registered_model(registered_name) # Create registered model
                    print(f"Created new registered model: {registered_name}")
                except Exception: # If it already exists, catch the exception
                    print(f"Registered model {registered_name} already exists (Expected).")

                source_uri = f"runs:/{run.info.run_id}/model/{model_filename}" # Source URI for the model version
                
                result = client.create_model_version( # Create a new model version
                    name=registered_name,
                    source=source_uri,
                    run_id=run.info.run_id
                )
                print(f"✅ Successfully Registered Version {result.version}!")
                
            except Exception as e: # Handle registration errors
                print(f"❌ Manual Registration Failed: {e}")
            
            return "Model Trained and Registered"
    
    raw_payload = extract_live_data() # Extract live data from GNews API
    history_json = pull_dvc_history() # Pull historical data from DVC
    merged_json = transform_and_profile(raw_payload, history_json) # Transform and profile data
    
    dvc_content = dvc_add_and_push(merged_json) # Version data with DVC
    train_model(dvc_content) # Train and log model with MLflow
    
    git_commit_and_push(dvc_content) # Commit and push changes to GitHub

stock_news_pipeline() # Instantiate the DAG
