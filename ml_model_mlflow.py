import mlflow
import pandas as pd
import numpy as np
import datetime

import ml_model

class MLModelWithMLFlow():
    def __init__(self,
                 mlflow_uri="http://localhost:8080"):
        self.ml_model = ml_model.MLModel()
        self.mlflow_uri = mlflow_uri
        self.mlflow_experiment_name = "wineyard_bin_clf"
        self.model_name = None
        self._init_mlflow()
    
    def _init_mlflow(self):
        mlflow.set_tracking_uri(uri=self.mlflow_uri)
        if mlflow.get_experiment_by_name(self.mlflow_experiment_name) is None:
            mlflow.create_experiment(self.mlflow_experiment_name)
        mlflow.set_experiment(experiment_name=self.mlflow_experiment_name)        
        
    
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
            
            mlflow.register_model(model_uri=model_uri, name=model_name)

            self.model_name = model_name

            return result
