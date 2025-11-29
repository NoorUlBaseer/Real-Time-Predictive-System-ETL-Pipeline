import pytest
from unittest.mock import patch

# Mock Airflow Variables to prevent connection errors during CI tests
with patch("airflow.models.Variable.get") as mock_get:
    mock_get.return_value = "dummy_value"
    from airflow.models import DagBag


class TestStockNewsDAG:
    @pytest.fixture(scope="class")
    def dag(self):  # Load the DAG for testing
        dag_bag = DagBag(dag_folder="dags/", include_examples=False)  # Load DAGs from the specified folder

        # Ensure there are no import errors
        assert len(dag_bag.import_errors) == 0, f"DAG Import Errors: {dag_bag.import_errors}"

        # Retrieve the specific DAG to be tested
        dag_id = "stock_news_pipeline"
        assert dag_id in dag_bag.dags, f"DAG {dag_id} not found"
        return dag_bag.get_dag(dag_id)

    def test_task_count(self, dag):  # Verify the number of tasks in the DAG
        expected_tasks = {
            "extract_live_data",
            "pull_dvc_history",
            "transform_and_profile",
            "dvc_add_and_push",
            "train_model",
            "git_commit_and_push"
        }

        # Check that all expected tasks are present in the DAG
        actual_tasks = set(task.task_id for task in dag.tasks)
        assert expected_tasks == actual_tasks, f"Task mismatch! Found: {actual_tasks}"

    def test_dependencies(self, dag):  # Verify task dependencies
        # Get tasks by their IDs
        transform = dag.get_task("transform_and_profile")
        dvc = dag.get_task("dvc_add_and_push")
        train = dag.get_task("train_model")
        git = dag.get_task("git_commit_and_push")

        # Check upstream dependencies for the transform task
        upstream_transform = set(t.task_id for t in transform.upstream_list)
        assert "extract_live_data" in upstream_transform
        assert "pull_dvc_history" in upstream_transform

        # Check upstream dependencies for the dvc task
        upstream_dvc = set(t.task_id for t in dvc.upstream_list)
        assert "transform_and_profile" in upstream_dvc

        # Check upstream dependencies for the train task
        upstream_train = set(t.task_id for t in train.upstream_list)
        assert "dvc_add_and_push" in upstream_train

        # Check upstream dependencies for the git task
        upstream_git = set(t.task_id for t in git.upstream_list)
        assert "dvc_add_and_push" in upstream_git

    def test_extract_retries(self, dag):  # Verify retries for the extract task
        extract_task = dag.get_task("extract_live_data")  # Get the extract task
        assert extract_task.retries == 0  # Ensure retries are set to 0
