""" Example script showing how to run mlfow experiment and where to add your code to make it you own. 

Before Running this you will need to have your local mlflow server running
In your command line run: ``mlflow server --host [HOST] --port [PORT]``
for exmaple for me this looks like: 

```
mlflow server --host 127.0.0.1 --port 6969
```

"""
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from mlflink.processing import preprocessing as pp
from importlib import resources
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models import infer_signature
import json
import numpy as np

# #############################
# SET UP MLFLOW 
# ############################

HOST = "http://127.0.0.1" # YOUR LOCAL HOST
PORT = 6969 # WHATEVER PORT YOU WANT TO SET THAT IS NOT CURRENTLY IN USE


mlflow.set_tracking_uri(f"{HOST}:{PORT}")

EXPERIMENT = "tutorial"
mlflow.set_experiment(EXPERIMENT)

# We instanciate the Mlflow client
client = MlflowClient()

# #############################
# DO PREPROCESSING
# ############################

# Get data from local file, live stream or mocked data stream
with resources.path("mlflink.data", "test_alerts.parquet") as parquet_path:
    PARQUET_FILE = parquet_path
alerts_df = pd.read_parquet(PARQUET_FILE)

# If you are listening to a broad topic such as the one
# running on the live Fink servers you will HAVE TO make some basic cuts
# In most cases these should be the criteria you have selected for your 
# favourite or bespoke Kafka topic.
cut_alerts_df = pp.make_cut(alerts_df) 

# Clean up the data by selectingonly the columns you care about
clean_df = pp.raw2clean(cut_alerts_df)

# OPTIONAL: further curate the data. For example by calling a third-party
# cross-matching service or running additional preprocessing you require
# before making you features or X matrix
curated_df = pp.run_sherlock(clean_df)

# Finally, make your features
X, ids = pp.make_X(curated_df)

# this is an example so we are going to fake the labels for now
y = np.array([0]*X.shape[0])
# needs to be a pandas dataframe
y = pd.DataFrame(y, columns=["labels"])

# #############################
# DO YOUR MLFLOW RUN
# ############################

# First let's stor our Model Hyperparameters in a dictionnary
# The keys of dictionary have to be the same hyperparameters (with the same names)
# as used in the model.
PARAMS = {
    "learning_rate": 0.1,
    "random_state": 42
}

# We now start our MLFlow run
# We have to give it a name which you will see in the mlflow UI
# I like to give them names that specify what I am changing in a particular run
# so I can distinguish between them. Here we are setting the learning rate
# and we may change it to another value in a subsequent run, so I am adding this to the name

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


