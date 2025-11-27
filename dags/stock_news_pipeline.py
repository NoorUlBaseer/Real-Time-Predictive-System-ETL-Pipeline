import os
import json
import requests
import pandas as pd
import pendulum
import subprocess
import mlflow
from datetime import datetime, timedelta
from textblob import TextBlob
from pathlib import Path
from airflow.models import Variable
from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator
from airflow.exceptions import AirflowFailException, AirflowSkipException
from ydata_profiling import ProfileReport

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

@dag(
    dag_id=DAG_ID,
    start_date=pendulum.datetime(2025, 11, 24, tz="UTC"),
    schedule="@daily", # Runs every day at midnight UTC
    catchup=False, # No backfilling
    doc_md="DAG for fetching daily technology/stock-related news, profiling and versioning",
)
def stock_news_pipeline(): # Main DAG function
    
    @task(retries=0) # No retries on failure
    def extract_live_data(**kwargs) -> str:
        execution_date = kwargs.get('ds') # Get execution date in 'YYYY-MM-DD' format

        target_date = datetime.strptime(execution_date, '%Y-%m-%d') - timedelta(days=1) # Fetch previous day's date
        str_date = target_date.strftime('%Y-%m-%dT00:00:00Z') # Start of day
        end_date = target_date.strftime('%Y-%m-%dT23:59:59Z') # End of day

        #hardcode dates
        #str_date = "2025-11-12T00:00:00Z"
        #end_date = "2025-11-12T23:59:59Z"

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
    
    task_pull_history = BashOperator( # Task to pull DVC history
        task_id='pull_dvc_history',
        bash_command=(
            "set -e; " # Fail Task if any command fails

            "rm -rf /tmp/repo_pull && " # Clean up any existing temp repo
            f"git clone https://{GITHUB_USER}:{GITHUB_TOKEN}@github.com/{GITHUB_USER}/{GITHUB_REPO}.git /tmp/repo_pull && " # Clone GitHub repo
            "cd /tmp/repo_pull && " # Change to repo directory
            
            f"dvc pull {PROCESSED_DATA_PATH} || echo 'Remote file not found, skipping pull'; " # Pull DVC tracked file, ignore if not found
            
            f"mkdir -p /usr/local/airflow/$(dirname {PROCESSED_DATA_PATH}) && " # Ensure target directory exists
            f"cp {PROCESSED_DATA_PATH} /usr/local/airflow/{PROCESSED_DATA_PATH} || echo 'No history to copy'" # Copy pulled file to Airflow directory, ignore if not found
        ),
        env=DVC_ENV, # DVC Environment Variables
    )

    @task
    def transform_and_profile(payload_json: str, **kwargs) -> str:
        payload = json.loads(payload_json) # Parse JSON string back to dict
        raw_data = payload.get("articles") # Extract articles
        
        # Get start and end dates for reporting
        str_date = payload.get("period_start")
        end_date = payload.get("period_end")
        
        df = pd.DataFrame(raw_data) # Load articles into DataFrame

        df['publishedAt'] = pd.to_datetime(df['publishedAt']) # Convert to datetime
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

        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True) # Ensure processed data directory exists
        
        history_len = 0 # Initialize history length
        if os.path.exists(PROCESSED_DATA_PATH) and os.path.getsize(PROCESSED_DATA_PATH) > 0: # Check if history file exists and is non-empty
            try:
                df_history = pd.read_csv(PROCESSED_DATA_PATH) # Load historical data
                df_history['publishedAt'] = pd.to_datetime(df_history['publishedAt']) # Convert to datetime
                
                history_len = len(df_history) # Get length of historical data
                print(f"Loaded history: {history_len} rows") # Log history length
                
                df_combined = pd.concat([df_history, df]) # Combine historical and new data
            except pd.errors.EmptyDataError: # Handle empty/corrupt file
                print("History file exists but is corrupt/empty. Starting fresh.")
                df_combined = df # Use only new data
        else: # No history file found
            print("No history found (fresh start)")
            df_combined = df # Use only new data

        df_combined = df_combined.drop_duplicates(subset=['title', 'publishedAt'], keep='last') # Remove duplicates based on title and publishedAt and keep last occurrence
        
        if len(df_combined) <= history_len and history_len > 0: # No new unique data
            print("Duplication check complete: No new unique data found.")
            raise AirflowSkipException("Data is identical to history. Skipping write and push.") # Skip this and downstream tasks
        
        df_combined.to_csv(PROCESSED_DATA_PATH, index=False) # Save combined data to CSV
        print(f"Update detected! Total rows: {len(df_combined)} (+{len(df_combined) - history_len})")
        print(f"Saved to {PROCESSED_DATA_PATH}")

        return PROCESSED_DATA_PATH # Return path to processed data for next task

    task_dvc = BashOperator( # Task to version and push processed data to DVC
        task_id='dvc_version_and_push',
        bash_command=(
            "set -euo pipefail; " # Fail task if any command fails

            "dvc init --no-scm; " # Initialize DVC without SCM
            "dvc remote add -d origin s3://dvc; " # Add DVC remote named 'origin'
            f"dvc remote modify origin endpointurl https://dagshub.com/{DAGSHUB_USER}/{REPO_NAME}.s3; " # Set custom endpoint URL
            
            f"dvc add {PROCESSED_DATA_PATH} && " # Track processed data with DVC
            "dvc push" # Push data to DVC remote
        ),
        env=DVC_ENV, # DVC Environment Variables
        cwd='.', # Working directory
    )

    task_git_commit = BashOperator( # Task to commit DVC changes to Git
        task_id='git_commit_and_push',
        bash_command=(
            "set -euo pipefail; " # Fail task if any command fails
            
            "rm -rf /tmp/repo_git && " # Clean up any existing temp repo
            f"git clone https://{GITHUB_USER}:{GITHUB_TOKEN}@github.com/{GITHUB_USER}/{GITHUB_REPO}.git /tmp/repo_git && " # Clone GitHub repo

            f"mkdir -p /tmp/repo_git/$(dirname {PROCESSED_DATA_PATH}) && " # Ensure target directory exists
            f"cp {PROCESSED_DATA_PATH}.dvc /tmp/repo_git/{PROCESSED_DATA_PATH}.dvc && " # Copy DVC file to temp repo

            "cd /tmp/repo_git && " # Change to repo directory

            "git config user.email 'baseersoomro2013@gmail.com' && " # Configure Git user email
            "git config user.name 'Noor Ul Baseer (Airflow)' && " # Configure Git user name

            f"git add {PROCESSED_DATA_PATH}.dvc && " # Stage DVC file for commit
            "git commit -m 'ETL Update: Processed data for {{ ds }}'; " # Commit changes with message
            "git push origin master" # Push changes to GitHub
        ),
    )

    raw_payload = extract_live_data() # Extract raw data task
    
    # Define task dependencies
    raw_payload >> task_pull_history >> transform_and_profile(raw_payload) >> task_dvc >> task_git_commit

stock_news_pipeline() # Instantiate the DAG
