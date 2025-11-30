import os
import time
import requests
import sys
from datetime import datetime, timezone

# Load environment variables
RAW_URL = os.environ.get("ASTRO_AIRFLOW_URL", "")
API_TOKEN = os.environ.get("ASTRO_API_TOKEN")
DAG_ID = "stock_news_pipeline"

def get_clean_url(url):
    """Cleans the URL to be API-ready."""
    if not url: return ""
    url = url.strip().rstrip("/")
    # Remove common suffixes if present
    for suffix in ["/home", "/api/v1", "/api/v2"]:
        if url.endswith(suffix):
            url = url[:-len(suffix)]
    return url

AIRFLOW_URL = get_clean_url(RAW_URL)
HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

def trigger_dag_with_retry(max_retries=60, delay=10):
    # Note: Using /api/v1 as standard. If your logs showed v2, change this to v2.
    endpoint = f"{AIRFLOW_URL}/api/v2/dags/{DAG_ID}/dagRuns"

    print(f"\n🚀 Attempting Trigger: {endpoint}")

    for attempt in range(max_retries):
        try:
            # FIX: Explicitly provide logical_date (Required for Airflow 3 / Strict APIs)
            current_date = datetime.now(timezone.utc).isoformat()
            payload = {
                "conf": {},
                "logical_date": datetime.now(timezone.utc).isoformat()
            }

            # Trigger the DAG
            response = requests.post(endpoint, headers=HEADERS, json=payload, allow_redirects=False)

            if response.status_code == 201:
                run_id = response.json()["dag_run_id"]
                print(f"✅ Success! Run ID: {run_id}")
                return run_id

            # Handle Redirects (The 405 Cause)
            elif 300 <= response.status_code < 400:
                print(f"⚠️ Redirect Detected ({response.status_code})!")
                print(f"   Server wants to send us to: {response.headers.get('Location')}")
                sys.exit(1)

            # Handle Cold Start
            elif response.status_code in [502, 503, 504]:
                print(f"💤 Server sleeping/booting ({response.status_code}). Retry {attempt+1}/{max_retries}...")

            # Handle Validation Errors (422)
            elif response.status_code == 422:
                print(f"❌ Error 422: Validation Failed.")
                print(f"   Response: {response.text}")
                sys.exit(1)

            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                sys.exit(1)

        except Exception as e:
            print(f"⚠️ Connection Error: {e}. Retrying...")

        time.sleep(delay)

    print("❌ Timeout waiting for deployment.")
    sys.exit(1)

def monitor_dag(run_id):
    endpoint = f"{AIRFLOW_URL}/api/v2/dags/{DAG_ID}/dagRuns/{run_id}"
    print(f"\nTitle: Monitoring Run {run_id}...")

    start = time.time()
    while (time.time() - start) < 1200: # 20 mins
        try:
            r = requests.get(endpoint, headers=HEADERS)
            if r.status_code == 200:
                state = r.json()['state']
                if state == 'success':
                    print("✅ DAG Succeeded!")
                    return
                elif state == 'failed':
                    print("❌ DAG Failed!")
                    sys.exit(1)
                print(f"   Status: {state}...")
            time.sleep(10)
        except Exception:
            time.sleep(10)

    print("❌ Timeout.")
    sys.exit(1)

if __name__ == "__main__":
    if not AIRFLOW_URL or not API_TOKEN:
        print("❌ Error: Secrets ASTRO_AIRFLOW_URL or ASTRO_API_TOKEN not found.")
        sys.exit(1)

    run_id = trigger_dag_with_retry()
    monitor_dag(run_id)
