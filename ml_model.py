import pandas as pd
import numpy as np
import pickle as pkl

from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, make_scorer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import TomekLinks



class MLModel:
    def __init__(self, 
                 random_seed=42):
        self.model = None
        self.scorer = make_scorer(f1_score, pos_label=1)
        self.random_seed = random_seed



    def load_model(self, model_path="./artifacts/model.pkl"):
        """
        Deserializes the model to the class from the provided `model_path`.
        """
        try:
            with open(model_path, 'rb') as file:
                self.model = pkl.load(file)
        except Exception as exception:
            print(exception)

    def save_model(self, model_path="./artifacts/model.pkl"):
        """
        Dumps the model parameters of the trained model to `model_path`.
        In case the folder does not exist, it creates it.
        """
        # Create the parent folder
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        # Dump the model to the target path
        with open(model_path, 'wb') as file:
            pkl.dump(self.model, file)

    def predict(self, input_data: pd.DataFrame) -> int:
        """
        Makes a prediction with the `input_data`.
        Returns 1, if the wine is of good quality, otherwise returns 0.
        """
        return self.model.predict(input_data)

    def train(self, input_data: pd.DataFrame) -> dict:
        """
        Trains the model with the provided `input_data`.
        It returns the in-the-sample and out-of-the-sample F1 score.
        """
        df = input_data
        df = df.drop_duplicates()

        df["target"] = np.where(df["quality"] < 7, 0, 1)

        y = df["target"]
        X = df.drop(["quality", "target"], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, train_size=.9, random_state=self.random_seed)

        params = {
            'knn__n_neighbors': 25,
            'knn__weights': 'uniform',
            'oversampler__sampling_strategy': 0.4,
            'oversampler__shrinkage': 2
        }

        model = Pipeline([
            ("select_params", ColumnTransformer(
                transformers=[('select_best_2', StandardScaler(), [
                               "volatile acidity", "sulphates", "alcohol"])]
            )),
            ("unskew", PowerTransformer()),
            ("pca", PCA()),
            ("undersampler", TomekLinks()),
            ("oversampler", RandomOverSampler(random_state=self.random_seed)),
            ("knn", KNeighborsClassifier())
        ])

        model.set_params(**params)

        model.fit(X_train, y_train)

        self.model = model

        return {
            "in_sample_score": self.scorer(model, X_train, y_train),
            "out_of_sample_score": self.scorer(model, X_test, y_test)
        }
