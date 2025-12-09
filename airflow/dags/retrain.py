from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import mlflow
from mlflow import MlflowClient
import os
import logging

logger = logging.getLogger(__name__)

MLFLOW_URL = "http://127.0.0.1:8090"
APP_URL = "http://127.0.0.1:8000"

# Define the MLflow client
mlflow.set_tracking_uri(MLFLOW_URL)
client = MlflowClient()

# Set the default model name
model_name = "wineyard_bin_clf"

# Path to the CSV file
csv_file_path = "./data/winequality-red.csv"


def train_model():
    # Call the train endpoint with the CSV file
    with open(csv_file_path, 'rb') as f:
        files = {'file': ("train.csv", f), "delimiter": ";"}
        response = requests.post(f"{APP_URL}/model/train", files=files)

    # Check if the request was successful
    if response.status_code != 200:
        data = response.json()
        raise Exception(
            f"Training failed: {
                data.get(
                    'error',
                    'Unknown error')}")

    # Parse response data
    data = response.json()

    logger.info("Training response: ", data)
    new_model_name = data["model_name"]
    new_accuracy = data["in_sample_score"]

    search_result = client.search_registered_models(
        filter_string="tag.environment='staging'",
        order_by=["creation_timestamp DESC"],
        max_results=1)

    if len(search_result) == 1:
        registered_model = search_result[0]
        latest_version = registered_model.latest_versions[0]
        run_id = latest_version.run_id
        run = client.get_run(run_id)
        old_accuracy = run.data.metrics.get("in_sample_score")
        logger.info(
            f"old accuracy: {old_accuracy}, new accuracy: {new_accuracy}")
        if new_accuracy >= old_accuracy:
            logger.info(f"Setting new model {new_model_name} as staging.")
            client.set_registered_model_tag(
                new_model_name, "environment", "staging")
            logger.info(f"Set new model {new_model_name} as staging.")
        else:
            logger.info(f"NOT set new model {new_model_name} as staging.")
    else:
        logger.info(f"Setting model {new_model_name} as staging.")
        client.set_registered_model_tag(
            new_model_name, "environment", "staging")
        logger.info(f"Set model {new_model_name} as staging.")

    return True


# Define the Airflow DAG
with DAG(
    dag_id="model_training",
    start_date=datetime(2023, 11, 1),
    catchup=False,
) as dag:

    # Define a task that calls the train_model function
    train_and_compare_task = PythonOperator(
        task_id="train_and_compare_model",
        python_callable=train_model,
        retries=3,
        retry_delay=timedelta(minutes=5),
    )

    train_and_compare_task
