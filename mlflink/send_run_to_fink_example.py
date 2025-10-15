""" Example script showing how to send a successful traing run to the Fink remote servers 

The way a successful run is "transfered" to Fink is by running it again but changing the 
tracking URI, a.k.a the destination where the run is saved. 

This script shows you how to grab all the information you need locally to reproduce the run
and save the necessary artifacts to send to Fink so that your algorithms and their 
associated environments can be loaded up Fink-side. 

The expectation before you run this is that:
1) You have been in touch with Julien Peloton and acquired a username and password for the Fink MLFlow server
2) Set the following environment variables 
* export MLFLOW_TRACKING_USERNAME="your_username"                                                                                                        
* export MLFLOW_TRACKING_PASSWORD="your_password"                                                                         
* export MLFLOW_TRACKING_URI="https://mlflow-dev.fink-broker.org"  # TO UPDATE  

3) You have run at least the example local run in mlflow_run_example.py so that you understand what's going on
but mostly so you have an actual run to use for this example here. 
"""
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from mlflink.preprocessing import preprocessing as pp
from importlib import resources
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models import infer_signature
import json
import numpy as np
from mlflink.utils.env_utils import generate_requirements_txt_from_imports
from importlib import resources
import yaml

FINK_MLFLOW_TRACKING_URI = "https://mlflow-dev.fink-broker.org"  # TO UPDATE
# #############################
# SET UP TO FIND YOUR RUN
# ############################
# 
# UPDATE THE FOLLOWING TO YOUR VALUES
# Note: root_dir is the directory where you ran `mlflow server --host 127.0.0.1 --port 6969`
# to run your experiments. It should contain subdirectories called `mlartifacts` and `mlruns`
# TODO: Jupyter notebooks essential to have screenshots for this. @Farid?
root_dir="/home/stevance/Science/fink-vra-notebooks/mlflink/"
experiment_ID="278379884379192837/"
run_ID="99942a2e92e34890a6af9d4c9d8812b2/"

# #############################
# GET ALL LOCAL INFORMATION NEEDED TO REPRODUCE THE SUCCESSFUL RUN
# #############################

# YOUR ENVIRONMENT
# MLflow saves some information about the environment bu tonly what is relevant
# to running inferance (running your model to get predictions)
# The problem is that Fink also needs to know how to run your preprocessing
# The code below looks at the code you have written in mlflink.preprocessing.preprocessing.py
# and creates a requirements.txt file with the relevant dependencies
with resources.path("mlflink", "preprocessing") as module_path:
    PKG_DIR = module_path

dependencies_path = generate_requirements_txt_from_imports(PKG_DIR, 
                                             "/tmp/requirements.txt", 
                                             include_self=False)

# YOUR DATA 
# Here we just grab the training data 
path_X = root_dir+"mlartifacts/"+experiment_ID+run_ID+"artifacts/X_train.parquet"
path_y = root_dir+"mlartifacts/"+experiment_ID+run_ID+"artifacts/y_train.parquet"
X = pd.read_parquet(path_X)
y = pd.read_parquet(path_y)

# YOUR PARAMETERS
with open(root_dir+"mlartifacts/"+experiment_ID+run_ID+"artifacts/meta.json", "r") as f:
    PARAMS = json.load(f)['params']


# #############################
# SET UP MLFLOW TO POINT TO FINK SERVERS
# #############################
mlflow.set_tracking_uri(FINK_MLFLOW_TRACKING_URI)
mlflow.set_experiment("tutorial")

client = MlflowClient()

with mlflow.start_run(run_name=f"test_LR_{PARAMS['learning_rate']}"
                      ):

    # 1. Train your model as you normally would
    model = HistGradientBoostingClassifier(**PARAMS)   # Instantiate the model
    model.fit(X.values, y)                             # train it 
    y_pred_new = model.predict(X.values)               # make predictions so you can evaluate

    # TODO: Having a nice small training and test set so the predictions and metrics
    # are meaningful would be good so people can see things are working. 

    # 2. Mlflow logging

    # save the hyper parameters
    mlflow.log_params(PARAMS)
    
    # save the model
    signature = infer_signature(X, y_pred_new)
    mlflow.sklearn.log_model(
        model,
        name="testMlFlink",
        signature=signature,
        input_example=X.iloc[:1],
        )


    # Evaluate and log some metrics. Pick the ones 
    # that you are interested in, this is a general example. 
    acc = accuracy_score(y, y_pred_new)
    mlflow.log_metric("accuracy", acc)

    prec = precision_score(y, y_pred_new)
    mlflow.log_metric("precision", prec)

    recall = recall_score(y, y_pred_new)
    mlflow.log_metric("recall", recall)

    f1 = f1_score(y, y_pred_new)
    mlflow.log_metric("f1-score", f1)

    # Log Data
    # NOTE: If your training data is large you DO NOT WANT TO DO THAT SYSTEMATICALLY
    # it might be a good thing to do once you have a run producing a model that you
    # know is working well for you so that you have all the ingredients to reproduce it
    # This shows you that you can save your data using .log_table
    # note that mlflow DOES NOT support .csv so you will want to save them as .parquet
    # they are just as trivial to load with pandas. 
    mlflow.log_table(X, "X_train.parquet")
    mlflow.log_table(y, "y_train.parquet")

    # Log MetaData
    # if there is additional information that you want to save you can do si easily 
    # by saving a temprorary json file and logging it as an artifact
    # Here I log the parameters a second time because it puts them in a convenient place
    # for when we want to rerun the models to save in Fink
    meta_info = {
        "params": PARAMS,
        "extra_info_you_care_about": 42
    }

    with open("meta.json", "w") as f:
        json.dump(meta_info, f, indent=2)
    mlflow.log_artifact("meta.json")

    # NECESSARY FOR FINK - NOT NEEDED FOR LOCAL: DEPEDENCIES AND DATA PROCESSING CODE
    # Here we log the requirements.txt we created above and we save the subpackage
    # that contains the preprocessing code
    mlflow.log_artifact(dependencies_path)
    mlflow.log_artifacts(resources.path("mlflink", "preprocessing").__str__(),
                          artifact_path="code")

