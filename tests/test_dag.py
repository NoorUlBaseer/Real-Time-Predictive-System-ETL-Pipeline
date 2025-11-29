from unittest.mock import patch


@patch("airflow.models.Variable.get")  # Mocking Airflow Variable.get method
def test_dag_structure_and_integrity(mock_variable_get):  # Test to validate DAG structure and task dependencies
    mock_variable_get.return_value = "dummy_value"  # Mock return value for Airflow Variables

    from airflow.models import DagBag  # Importing DagBag to load DAGs

    dag_bag = DagBag(dag_folder="dags/", include_examples=False)  # Loading DAGs from the specified folder

    # Ensure there are no import errors in the DAGs
    assert len(dag_bag.import_errors) == 0, f"DAG Import Errors: {dag_bag.import_errors}"

    dag_id = "stock_news_pipeline"  # The ID of the DAG to test
    assert dag_id in dag_bag.dags, f"DAG {dag_id} not found in DagBag"  # Check if the DAG is present

    dag = dag_bag.get_dag(dag_id)  # Retrieve the DAG object

    expected_tasks = {  # Expected tasks in the DAG
        "extract_live_data",
        "pull_dvc_history",
        "transform_and_profile",
        "dvc_add_and_push",
        "train_model",
        "git_commit_and_push"
    }
    actual_tasks = set(task.task_id for task in dag.tasks)  # Actual tasks found in the DAG
    assert expected_tasks == actual_tasks, f"Task mismatch! Found: {actual_tasks}"  # Validate tasks

    # Retrieve tasks from the DAG
    transform = dag.get_task("transform_and_profile")
    dvc = dag.get_task("dvc_add_and_push")
    train = dag.get_task("train_model")
    git = dag.get_task("git_commit_and_push")

    # Validate upstream tasks for transform_and_profile
    upstream_transform = set(t.task_id for t in transform.upstream_list)
    assert "extract_live_data" in upstream_transform
    assert "pull_dvc_history" in upstream_transform

    # Validate upstream tasks for dvc_add_and_push
    upstream_dvc = set(t.task_id for t in dvc.upstream_list)
    assert "transform_and_profile" in upstream_dvc

    # Validate upstream tasks for train_model and git_commit_and_push
    upstream_train = set(t.task_id for t in train.upstream_list)
    assert "dvc_add_and_push" in upstream_train

    # Validate upstream tasks for git_commit_and_push
    upstream_git = set(t.task_id for t in git.upstream_list)
    assert "dvc_add_and_push" in upstream_git

    # Check that the extract_live_data task has zero retries
    extract_task = dag.get_task("extract_live_data")
    assert extract_task.retries == 0