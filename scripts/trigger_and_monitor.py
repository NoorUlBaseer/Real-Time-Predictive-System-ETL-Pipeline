import os
import time
import requests
import sys

# Load environment variables
RAW_URL = os.environ.get("ASTRO_AIRFLOW_URL", "")
API_TOKEN = os.environ.get("ASTRO_API_TOKEN")
DAG_ID = "stock_news_pipeline"

def get_clean_url(url):  # Function to clean and standardize the Airflow URL
    if not url:
        return ""
    
    url = url.strip()
    
    # 1. Remove trailing slash (The cause of your 405 error)
    if url.endswith("/"):
        url = url[:-1]
        
    # 2. Remove '/home' if copied from browser address bar
    if url.endswith("/home"):
        url = url.replace("/home", "")
        
    return url

# Clean the URL once at startup
AIRFLOW_URL = get_clean_url(RAW_URL)

def trigger_dag_with_retry(max_retries=60, delay=10):
    url = f"{AIRFLOW_URL}/api/v2/dags/{DAG_ID}/dagRuns"
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }

    print(f"🚀 Attempting to trigger DAG: {DAG_ID}")
    print(f"🔗 Using Endpoint: {url}") # Debug print to confirm fix

    for attempt in range(max_retries):
        try:
            # We pass 'conf' to ensure we can parameterize the run if needed later
            response = requests.post(url, headers=headers, json={"conf": {}})

            if response.status_code == 200:
                run_id = response.json()["dag_run_id"]
                print(f"✅ Success! DAG Triggered. Run ID: {run_id}")
                return run_id

            elif response.status_code in [502, 503, 504]:
                print(f"⚠️ Deployment sleeping/updating (Status {response.status_code}). Retrying in {delay}s... ({attempt+1}/{max_retries})")
            
            # Specific catch for the 405 error to be helpful
            elif response.status_code == 405:
                print(f"❌ Error 405: Method Not Allowed. Your URL might still be malformed: {url}")
                sys.exit(1)

            else:
                print(f"❌ API Error: {response.status_code} - {response.text}")
                sys.exit(1)

        except requests.exceptions.ConnectionError:
            print(f"⚠️ Connection Refused (Webserver down). Retrying in {delay}s... ({attempt+1}/{max_retries})")

        time.sleep(delay)

    print("❌ Failed to wake up Deployment after multiple attempts.")
    sys.exit(1)

def monitor_dag(run_id):
    url = f"{AIRFLOW_URL}/api/v2/dags/{DAG_ID}/dagRuns/{run_id}"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    print("⏳ Monitoring DAG execution...")

    start_time = time.time()
    timeout = 1200  # 20 minutes

    while (time.time() - start_time) < timeout:
        try:
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                state = response.json()["state"]
                if state == "success":
                    print("✅ DAG completed successfully!")
                    return True
                elif state == "failed":
                    print("❌ DAG Failed!")
                    sys.exit(1)
                print(f"   Status: {state}...")
            else:
                print(f"⚠️ Failed to check status: {response.status_code}")
                
            time.sleep(10)

        except Exception as e:
            print(f"⚠️ Monitor glitch: {e}. Retrying...")
            time.sleep(10)

    print("❌ Timeout: DAG took too long to finish.")
    sys.exit(1)

if __name__ == "__main__":
    if not AIRFLOW_URL or not API_TOKEN:
        print("❌ Error: Secrets ASTRO_AIRFLOW_URL or ASTRO_API_TOKEN not found.")
        sys.exit(1)

    run_id = trigger_dag_with_retry()
    monitor_dag(run_id)
