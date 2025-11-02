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
