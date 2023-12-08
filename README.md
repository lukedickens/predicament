# Dissesrtation
UCL dissertation

ERP_Ray.py is the main scripts for EEGNet model and main-Thin-ResNet.py is the main scripts for EEG-DL model



# By Luke

## Partitioning data

Make sure that you have the data in an appropriate data folder. I use `./data` where `.` is the repository root. The care home data, should be in a subfolder called `CARE_HOME_DATA` (i.e. in `./data/CARE_HOME_DATA`).

The first bit of code to be run is the 

`python3 prepare_evaluation_data.py --between -g <channel-group> -w <window-size>`

The default is for window size `1024` and channel group `dreem-minimal`, but a possibly better option is `dreem-eeg`. So a good run would be:

```
python3 prepare_evaluation_data.py --between -g dreem-eeg -w 1024
```

Good values to try for `-w` are 256, 512, (768), 1024, (1536,) 2048. At the moment the default value for window step is `<window-size>/8`. You can specify a different window step with the `-s` flag. It might make sense to have a step size of 128 for all window sizes.

Ifd you want to specify the channels to use directly you can do so with the `-c` flag and specify the channels as a comma separated string of channel names, e.g.

```python3 prepare_evaluation_data.py --between -c "EEG Fpz-O1,EEG Fpz-O2,EEG Fpz-F7"
```

This merges the observational data from the study with the EEG data in the edf files and produces a bunch of folds (hold one group out, where each participant represents a group). It places this in a subfolder of the data folder called `evaluation/<DATETIME>` (e,g, in `./data/evaluation/20230713194411/`). Subfolders of this folder are named `fold<N>` one for each fold.

## Running on Luke's eeg-implementation.

Then you need to also clone the repository arl-eegmodels (https://github.com/vlawhern/arl-eegmodels) and place this in your `PYTHONPATH`. So, I put this repository in the folder `~/git/external/arl-eegmodels/`, then run the command:

`export PYTHONPATH=~/git/external/arl-eegmodels/:${PYTHONPATH}`

Now you can test this by running `python3` and trying `import EEGModels` (you will need all the dependencies for `arl-eegmodels` too.

Now you can run the grid-search code (as it currently stands) with the command

`python3 eegnet_evaluation.py data/evaluation/20230713194411/fold0`

Which will run the grid search on just the 0th fold.

My aim is instead, to input something like

`python3 eegnet_evaluation.py data/evaluation/20230713194411 data/results/performance.csv`

And for it to a) load all previous performance results from `data/results/performance.csv` associated with all hypterparameter choices of interest, b) sample a new set of hyperparameter choices from a bayesian optimisation module (for the hyperparameters that we are searching), c) run `epoch` epochs of training on the model with those hyperparameter choices on every fold in `data/evaluation/20230713194411`, d) take the average performance over all folds as the performance for those hyperparametter choices (watch out for NaN), e) save the hyperparameter choices and avg performance results to `data/results/performance.csv`.


## Featured data and experiments

For these experiments we have to convert time-series windows into feature vectors so they can be processed by conventional machine learning models, such as Random Forests. First we need to load the data to windowed files of a chosen length. For the dreem data (edf format) we use

```python3 prepare_evaluation_data.py --mode windowed -f dreem```

For the E4 data (csv files) we use:

```python3 prepare_evaluation_data.py --mode windowed -f E4```

Either of these calls will generate a data file and config file in an appropriate directory, and normally this is time-stamped. For instance, on 6th Dec 2023 at 19:35, I created folder relative to the pwd of `data/featured/20231206193533`. This corresponds to subdir of `20231206193533`. We can then create features from this, either by using `--subdir 20231206193533` or renaming the subdir on our system and using the new name in the call. I used

```python3 prepare_evaluation_data.py --mode featured --subdir 20231206193533```

By default this only produces a subset of the features, referred to as feature-group `stats`. An equivalent call would be 

```python3 prepare_evaluation_data.py --mode featured --subdir 20231206193533 --feature-group stats```

Some features take longer to create than others and you may want to break your processing down into stages. If I call the script with `--mode featured` again then it will augment pre-existing features with any new features, overwriting the pre-existing features with a new copy of identical data (effectively leaving them unchanged). I tried running this:

```python3 prepare_evaluation_data.py --mode featured --subdir 20231206193533 --feature-set arCoeff,Hurst,LyapunovExponent```

and 

```python3 prepare_evaluation_data.py --mode featured --subdir 20231206193533 --feature-set MaxFreqInd,MeanFreq,FreqSkewness,FreqKurtosis```

This extends the pre-existing featured data to include channel-wise features for each of 'arCoeff',  'Hurst', 'LyapunovExponent', 'MaxFreqInd', 'MeanFreq', 'FreqSkewness', 'FreqKurtosis'. At time of writing, the second call just above is equivalent to 

```python3 prepare_evaluation_data.py --mode featured --subdir 20231206193533 --feature-group freq```

One feature 'Entropy' can take extreme times to compute (>120 hours) and has been omitted until a more efficient approach can be found.python3 prepare_evaluation_data.py --mode featured --subdir 20231206193533 --feature-set arCoeff,Hurst,LyapunovExponent