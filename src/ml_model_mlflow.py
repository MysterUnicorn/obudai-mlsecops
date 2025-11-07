import mlflow

import pandas as pd
import numpy as np
import datetime

import ml_model as ml_model


class MLModelWithMLFlow():
    def __init__(self,
                 mlflow_uri="http://localhost:8080"):
        self.ml_model = ml_model.MLModel()
        self.mlflow_uri = mlflow_uri
        self.mlflow_experiment_name = "wineyard_bin_clf"
        self.mlflow_model = None
        self.mlflow_client = None
        self._init_mlflow()
        self._load_last_staging_model()
    
    def _init_mlflow(self):
        mlflow.set_tracking_uri(uri=self.mlflow_uri)
        if mlflow.get_experiment_by_name(self.mlflow_experiment_name) is None:
            mlflow.create_experiment(self.mlflow_experiment_name)
        mlflow.set_experiment(experiment_name=self.mlflow_experiment_name)

        self.mlflow_client = mlflow.MlflowClient()

    def predict(self, input_data:pd.DataFrame) -> np.array:
        return self.ml_model.predict(input_data)

    def train_and_save(self, input_data:pd.DataFrame) -> dict:
        model_name = f"{self.mlflow_experiment_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=model_name) as run:

            result = self.ml_model.train(input_data)
            mlflow.log_metric("train accuracy", result["in_sample_score"])
            mlflow.log_metric("test accuracy", result["out_of_sample_score"])

            mlflow.sklearn.log_model(
                sk_model=self.ml_model.model,
                artifact_path="model",
                input_example=input_data.drop(["quality"], axis=1).loc[:0],
                signature=mlflow.models.infer_signature(
                    input_data.drop(["quality"], axis=1),
                    pd.DataFrame(np.where(input_data["quality"] < 7, 0, 1))
                )
            )

            model_uri = f"runs:/{run.info.run_id}/model"

            self.mlflow_model = mlflow.register_model(model_uri=model_uri, name=model_name)
            self._mark_model_as_staging()

            return result
        
    def _mark_model_as_staging(self):
        if self.mlflow_model is None:
            print("ERROR: Cannot stage model: No model mlflow model loaded to the class.")
        else:
            model_name = self.mlflow_model.name
            self.mlflow_client.set_registered_model_tag(model_name, "environment", "staging")

    def _load_last_staging_model(self):
        search_result = self.mlflow_client.search_registered_models(
            filter_string="tag.environment='staging'", 
            order_by=["creation_timestamp DESC"], 
            max_results=1)
        if len(search_result) == 1:
            latest_staging_model = search_result[0]
            source = latest_staging_model.latest_versions[0].source
            loaded_model = mlflow.sklearn.load_model(source)

            self.ml_model = ml_model.MLModel(model=loaded_model)
            self.mlflow_model = latest_staging_model

            print(f"INFO: Inital model was loaded: {self.mlflow_model.name}.")
        else:
            print("INFO: No inital model was loaded. -- No staging model was found.")

    def get_model_name(self):
        return self.mlflow_model.name

