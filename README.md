# Tutorial and template package to create Fink Science modules using MlFlow

This goal of this repository is to teach you how to train models that can be easily deployed in Fink, 
as well as provide you with a template package that you can copy and tailor to host your own codes. 

_Here have info about how to get access to the Fink Ml Flow server_

## The Tutorials
There are currently two scripts in the main repo: 
* `mlflow_local_run_example.py`: Walks you through how to use ML Flow locally 
* `send_run_to_fink_example.py`: Walks you through how to upload a local Ml Flow run to the Fink ML Flow servers

## Where does my bespoke code go?
- Any preprocessing you do with your data should go in `processing.prepocessing.py`. 
- Your models of choice should replace the `HistogramBasedGradientBoostedClassifier` in the examples (once you have copied their content to your own scripts.) 
- You should make sure the `processing.processor.py` script actually does your data processing as you intended. 


## Dev notes
- the `processor.py` file needs to be fed the model. Julien I am not sure how we get this from the Fink server/
- `mlflow_local_run_example.py` and `send_run_to_fink_example.py` could be turned into jupyter notebooks so can include screenshots of where things (like artifacts) are in the MLFlow UI. 
- Need better example data so can train a quick model that makes sense 
- Need bettr explanation of where user code gos


