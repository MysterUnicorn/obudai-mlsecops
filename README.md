# Wine Quality Prediction API

Simple API to perform wine quality prediction based on chemical parameters.
The API also provides an interface to retrain it.

We support building a container of our app.

The models are managed by MLFlow.

Retraining of the model can be performed with Airflow.

In order to measure data drift, a streamlit server can host evidentlyAI reports. 


## Commands
We use [mise-en-place](https://mise.jdx.dev/). 
Installation steps can be found [here](https://mise.jdx.dev/installing-mise.html).

To display the available commands, run:
```
mise tasks
```

To run the dev version of the app server:
```
mise docker:run    -- Build and run the application in development mode on port 8000.
mise run:ariflow   -- Run Airflow on localhost:8080 with credentials `admin:admin`.
mise run:mlflow    -- Run Mlflow tracking server on localhost:8090.
mise format:python -- PEP8 formats the sourcecode.
mise test          -- Runs the unittest suite.
```

## Folder Structure

```
├── airflow           -- airflow dags, AIRFLOW_HOME when running ariflow locally
├── data              -- csvs of the dataset
├── Dockerfile
├── docs              -- documentation for the class
├── mise.toml         -- build tool's configfile
├── monitoring        -- streamlit app with notebook to create evidentlyAI report
├── notebooks         -- Notebooks about experimentations
├── README.md
├── requirements.txt  
├── src               -- app's sourcecode
└── tests             -- unit tests
```
