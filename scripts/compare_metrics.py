import mlflow
from mlflow.tracking import MlflowClient
import os
import sys

# Configure MLflow to use DagsHub
mlflow.set_tracking_uri(f"https://dagshub.com/{os.getenv('DAGSHUB_USERNAME')}/{os.getenv('REPO_NAME')}.mlflow")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")


def main():
    # Fetch the experiment
    experiment_name = "Stock_Price_Prediction"
    model_name = "Stock_Sentiment_Predictor"

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if not experiment:  # Experiment not found
        print("❌ Error: Experiment not found.")
        sys.exit(1)

    runs = mlflow.search_runs(  # Get the two most recent runs
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=2
    )

    if len(runs) < 1:  # No runs found
        print("❌ No runs found.")
        sys.exit(1)

    # Extract RMSE metrics
    candidate_rmse = runs.iloc[0]["metrics.rmse"]
    candidate_id = runs.iloc[0]["run_id"]

    if len(runs) > 1:  # Baseline exists
        baseline_rmse = runs.iloc[1]["metrics.rmse"]
        baseline_id = runs.iloc[1]["run_id"]
        print(f"🔍 Comparing Candidate ({candidate_rmse:.4f}) vs Baseline ({baseline_rmse:.4f})")
    else:  # No baseline, first run
        print("⚠️ First run detected. Using default threshold.")
        baseline_rmse = 2.5
        baseline_id = "Default"

    is_better = candidate_rmse <= baseline_rmse  # Determine if candidate is better

    # Generate markdown report
    report = f"""
# 📊 Model Comparison Report

| Role | Run ID | RMSE | Result |
|------|--------|------|--------|
| **Candidate** (New) | `{candidate_id[:7]}` | `{candidate_rmse:.4f}` | {"✅ Better" if is_better else "❌ Worse"} |
| **Baseline** (Prod) | `{baseline_id[:7]}` | `{baseline_rmse:.4f}` | 🛡️ |

"""

    with open("report.md", "w") as f:  # Write report to markdown file
        f.write(report)

    if not is_better:  # Candidate did not beat baseline
        print(f"❌ Blocked: Candidate RMSE ({candidate_rmse}) is worse than Baseline ({baseline_rmse}).")
        sys.exit(1)  # Fail the pipeline

    print(f"✅ Approved: Candidate ({candidate_rmse}) beat Baseline ({baseline_rmse}).")

    print("🚀 Promoting Candidate Model to @champion...")
    client = MlflowClient()  # Create MLflow client

    # Fetch model versions for the candidate run
    versions = client.search_model_versions(f"name='{model_name}' and run_id='{candidate_id}'")

    if not versions:  # No model version found
        print(f"⚠️ Warning: Run {candidate_id} succeeded, but no registered model version was found.")
        print("Did the training task fail to register the model?")
        sys.exit(1)

    candidate_version = versions[0].version  # Get the candidate model version

    client.set_registered_model_alias(  # Promote candidate model to champion alias
        name=model_name,
        alias="champion",
        version=candidate_version
    )

    print(f"👑 Success! Version {candidate_version} is now the @champion.")


if __name__ == "__main__":
    main()
