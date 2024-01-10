# MLDS MLOps
Repo for MLOps Course homework @ MLDS Program, HSE

The project utilises the code for the CIAN Price prediction project, being conducted @ HSE.
The data is collected from CIAN website using beautifulsoap.

Data collected is already cleansed and preprocessed. The train & test dataset consists of flats for sale (from studios to 5+ rooms) in Moscow, Russia.
After multiple experiments conducted in Jupyter notebooks the best model so far is Random Forest Regressor, which is being used here for training and inference.

This repo utilises several tools covered in the course:
- poetry for dependency management
- pre-commit hooks for code quality control
- dvc for data version control (for dataset & trained model)
- hydra for training parameters
- mlflow for experiment tracking

How to setup the repo after cloning:
- create new environment using `conda`
- do `poetry install`
- do `pre-commit install`
- do `dvc pull` to get the datasets and trained model
