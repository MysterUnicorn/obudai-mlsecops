import os

import pandas as pd
from io import StringIO

from flask import Flask
from flask_restx import Api, Resource, fields
from werkzeug.datastructures import FileStorage

import ml_model_mlflow
model = ml_model_mlflow.MLModelWithMLFlow()

app = Flask(__name__)
api = Api(
    app,
    version='1.0',
    title='Wine Quality Prediction API',
    description='A simple API for predicting wine quality'
)

namespace = api.namespace('wine', description='Wine quality operations')

api.add_namespace(namespace)

prediction_model = api.model('Prediction', {
    'fixed acidity': fields.Float(required=True, description='Fixed Acidity'),
    'volatile acidity': fields.Float(required=True, description='Volatile Acidity'),
    'citric acid': fields.Float(required=True, description='Citric Acid'),
    'residual sugar': fields.Float(required=True, description='Residual Sugar'),
    'chlorides': fields.Float(required=True, description='Chlorides'),
    'free sulfur dioxide': fields.Float(required=True, description='Free Sulfur Dioxide'),
    'total sulfur dioxide': fields.Float(required=True, description='Total Sulfur Dioxide'),
    'density': fields.Float(required=True, description='Density'),
    'pH': fields.Float(required=True, description='pH'),
    'sulphates': fields.Float(required=True, description='Sulphates'),
    'alcohol': fields.Float(required=True, description='Alcohol'),
})

file_upload = api.parser()
file_upload.add_argument(
    'file',
    location='files',
    type=FileStorage,
    required=True,
    help='CSV file containing training data.'
)
file_upload.add_argument(
    "delimiter",
    type=str,
    required=False,
    help="Delimiter of the uploaded CSV file."
)


@namespace.route('/predict')
class Predict(Resource):
    @api.expect(prediction_model)
    def post(self):
        data = pd.DataFrame([api.payload])
        prediction = model.predict(data)
        return {
            'message': 'Prediction made successfully',
            'predicted_quality': int(prediction[0])
        }


@namespace.route('/train')
class Train(Resource):
    @api.expect(file_upload)
    def post(self):
        args = file_upload.parse_args()
        uploaded_file = args['file']
        delimiter = args["delimiter"]

        if os.path.splitext(uploaded_file.filename)[1] != '.csv':
            return {'message': 'Invalid file format. Please upload a CSV file.'}, 400
        data = uploaded_file.read().decode('utf-8')

        pandas_data = pd.read_csv(StringIO(data), sep=delimiter)
        print(pandas_data.head())

        train_results = model.train_and_save(pandas_data)

        return {
            'message': 'Model trained successfully',
            **train_results
        }


if __name__ == '__main__':
    app.run(debug=True)
