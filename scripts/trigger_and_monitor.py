import os
import time
import requests
import sys

# Load environment variables for Airflow connection
AIRFLOW_URL = os.environ.get("ASTRO_AIRFLOW_URL")
API_TOKEN = os.environ.get("ASTRO_API_TOKEN")
DAG_ID = "stock_news_pipeline"


def trigger_dag_with_retry(max_retries=20, delay=10):  # Retry logic to trigger DAG
    url = f"{AIRFLOW_URL}/api/v1/dags/{DAG_ID}/dagRuns"  # Endpoint to trigger DAG
    headers = {  # Headers for authentication
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }

    print(f"🚀 Attempting to trigger DAG: {DAG_ID}")

    for attempt in range(max_retries):  # Retry loop
        try:  # Attempt to trigger DAG
            response = requests.post(url, headers=headers, json={})

            if response.status_code == 200:  # Success
                run_id = response.json()["dag_run_id"]
                print(f"✅ Success! DAG Triggered. Run ID: {run_id}")
                return run_id

            elif response.status_code in [502, 503, 504]:  # Transient errors
                print(f"""⚠️ Deployment sleeping/updating (Status {response.status_code}).
                      Retrying in {delay}s... ({attempt+1}/{max_retries})""")

            else:  # Other errors
                print(f"❌ API Error: {response.status_code} - {response.text}")
                # Don't retry on 401 (Auth) or 404 (DAG missing)
                sys.exit(1)

        except requests.exceptions.ConnectionError:  # Connection issues
            print(f"⚠️ Connection Refused (Webserver down). Retrying in {delay}s... ({attempt+1}/{max_retries})")

        time.sleep(delay)  # Wait before retrying

    print("❌ Failed to wake up Deployment after multiple attempts.")
    sys.exit(1)


def monitor_dag(run_id):  # Monitor DAG execution status
    url = f"{AIRFLOW_URL}/api/v1/dags/{DAG_ID}/dagRuns/{run_id}"  # Endpoint to check DAG run status
    headers = {"Authorization": f"Bearer {API_TOKEN}"}  # Headers for authentication

    print("⏳ Monitoring DAG execution...")

    start_time = time.time()  # Start time for timeout calculation
    timeout = 1200  # 20 minutes

    while (time.time() - start_time) < timeout:  # Loop until timeout
        try:  # Attempt to get DAG run status
            response = requests.get(url, headers=headers)  # Get DAG run status
            state = response.json()["state"]

            if state == "success":  # DAG succeeded
                print("✅ DAG completed successfully!")
                return True
            elif state == "failed":  # DAG failed
                print("❌ DAG Failed!")
                sys.exit(1)

            print(f"   Status: {state}...")
            time.sleep(10)  # Wait before next status check

        except Exception as e:  # Handle exceptions during monitoring
            print(f"⚠️ Monitor glitch: {e}. Retrying...")
            time.sleep(10)

    print("❌ Timeout: DAG took too long to finish.")
    sys.exit(1)


if __name__ == "__main__":
    if not AIRFLOW_URL or not API_TOKEN:  # Check for required secrets
        print("❌ Error: Secrets ASTRO_AIRFLOW_URL or ASTRO_API_TOKEN not found.")
        sys.exit(1)

    run_id = trigger_dag_with_retry()  # Trigger the DAG with retry logic
    monitor_dag(run_id)  # Monitor the DAG execution status
