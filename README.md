# Dissesrtation
UCL dissertation

ERP_Ray.py is the main scripts for EEGNet model and main-Thin-ResNet.py is the main scripts for EEG-DL model



# By Luke

Make sure that you have the data in an appropriate data folder. I use `./data` where `.` is the repository root. The care home data, should be in a subfolder called `CARE_HOME_DATA` (i.e. in `./data/CARE_HOME_DATA`).

The first bit of code to be run is the 

`python3 prepare_evaluation_data.py`

This merges the observational data from the study with the EEG data in the edf files and produces a bunch of folds (hold one group out, where each participant represents a group). It places this in a subfolder of the data folder called `evaluation/<DATETIME>` (e,g, in `./data/evaluation/20230713194411/`). Subfolders of this folder are named `fold<N>` one for each fold.

Then you need to also clone the repository arl-eegmodels (https://github.com/vlawhern/arl-eegmodels) and place this in your `PYTHONPATH`. So, I put this repository in the folder `~/git/external/arl-eegmodels/`, then run the command:

`export PYTHONPATH=~/git/external/arl-eegmodels/:${PYTHONPATH}`

Now you can test this by running `python3` and trying `import EEGModels` (you will need all the dependencies for `arl-eegmodels` too.

Now you can run the grid-search code (as it currently stands) with the command

`python3 eegnet_evaluation.py data/evaluation/20230713194411/fold0`

Which will run the grid search on just the 0th fold.

My aim is instead, to input something like

`python3 eegnet_evaluation.py data/evaluation/20230713194411 data/results/performance.csv`

And for it to a) load all previous performance results from `data/results/performance.csv` associated with all hypterparameter choices of interest, b) sample a new set of hyperparameter choices from a bayesian optimisation module (for the hyperparameters that we are searching), c) run `epoch` epochs of training on the model with those hyperparameter choices on every fold in `data/evaluation/20230713194411`, d) take the average performance over all folds as the performance for those hyperparametter choices (watch out for NaN), e) save the hyperparameter choices and avg performance results to `data/results/performance.csv`.



