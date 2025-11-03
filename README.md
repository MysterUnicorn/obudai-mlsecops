## Change Note
In the current incrment, you can run the application with mlflow. 
Whenever you train the app, a model artifact is saved to mlflow.
TODO By default the latest staging app loads upon app init.
TODO You can load specific revisions of the model by an endpoint.

## The build tool
We use [mise-en-place](https://mise.jdx.dev/). 
Installation steps can be found [here](https://mise.jdx.dev/installing-mise.html).

To display the available commands, run:
```
mise tasks
```

To run the dev version of the app server:
```
mise run:dev
```
The app is gonna be available at `localhost:8000`
MLFlow is gonna be available at `localhost:8080`

To run tests:
```
mise test
```

To pep8 format the repo:
```
mise format:python
```

## Data prep
What we need:
- dedup
- scaling
- dropping dims
- PCA
