import mlflow
from mlflow.tracking import MlflowClient
import os
import sys

# Configure MLflow to use DagsHub
mlflow.set_tracking_uri(f"https://dagshub.com/{os.getenv('DAGSHUB_USERNAME')}/{os.getenv('REPO_NAME')}.mlflow")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")


def main():
    # Config
    experiment_name = "Stock_Price_Prediction"
    model_name = "Stock_Sentiment_Predictor"

    client = MlflowClient()

    # Fetch Candidate (The Latest Run)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print("❌ Error: Experiment not found.")
        sys.exit(1)

    runs = mlflow.search_runs(  # Fetch all runs in the experiment
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )

    if runs.empty:  # If no runs found, fail the pipeline
        print("❌ No runs found.")
        sys.exit(1)

    # Extract Candidate Metrics
    candidate_rmse = runs.iloc[0]["metrics.rmse"]
    candidate_run_id = runs.iloc[0]["run_id"]
    print(f"🆕 Candidate Run ID: {candidate_run_id} (RMSE: {candidate_rmse:.4f})")

    # Fetch Baseline (The Current Champion)
    try:  # Ask Registry: "Who is the current champion?"
        champion_version = client.get_model_version_by_alias(model_name, "champion")
        champion_run_id = champion_version.run_id

        # Fetch metrics for the Champion
        champion_run = mlflow.get_run(champion_run_id)
        baseline_rmse = champion_run.data.metrics.get("rmse", 0)  # Default to 0 if missing, but typically shouldn't be
        baseline_id = champion_run_id
        baseline_source = f"Champion (v{champion_version.version})"
        print(f"🛡 Found Champion Model (v{champion_version.version}) with RMSE: {baseline_rmse:.4f}")

    except Exception:  # If no champion exists (First run ever), set a default fallback
        print("⚠ No Champion found (First run?). Using default threshold.")
        baseline_rmse = 2.5
        baseline_id = "Default"
        baseline_source = "Static Threshold"

    is_better = candidate_rmse <= baseline_rmse  # comparison logic (less is better for RMSE)

    print(f"🔍 Comparing Candidate ({candidate_rmse:.4f}) vs {baseline_source} ({baseline_rmse:.4f})")

    # Generate CML Report
    report = f"""
# 📊 Model Comparison Report

| Role | Source | Run ID | RMSE | Status |
|------|--------|--------|------|--------|
| *Candidate* | New Run | {candidate_run_id[:7]} | {candidate_rmse:.4f} | {"✅ Better" if is_better else "❌ Worse"} |
| *Baseline* | {baseline_source} | {baseline_id[:7]} | {baseline_rmse:.4f} | 🛡 |

"""

    with open("report.md", "w") as f:  # Write the report to a markdown file
        f.write(report)

    if not is_better:  # If candidate is worse, exit without promotion
        print(f"⚠ Candidate RMSE ({candidate_rmse}) is worse than Baseline ({baseline_rmse}).")
        print("⛔ Skipping model promotion.")
        sys.exit(0)  # Exit cleanly without running the promotion code below

    print(f"✅ Approved: Candidate beat {baseline_source}.")

    print("🚀 Promoting Candidate Model to @champion...")

    # Find the registered model version for this specific run
    versions = client.search_model_versions(f"name='{model_name}' and run_id='{candidate_run_id}'")

    if not versions:  # If no version found, something went wrong
        print(f"⚠ Warning: Run {candidate_run_id} succeeded, but no registered model version was found.")
        print("Did the training task fail to register the model?")
        sys.exit(1)

    candidate_version_number = versions[0].version  # Fetch the version number of the candidate

    client.set_registered_model_alias(  # Promote the candidate to champion
        name=model_name,
        alias="champion",
        version=candidate_version_number
    )

    print(f"👑 Success! Version {candidate_version_number} is now the @champion.")


if __name__ == "__main__":
    main()
