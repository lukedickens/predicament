"""
"""
#

import configparser

import numpy as np
import pandas as pd
import os
import json
import itertools

# EEGNet-specific imports
try:
    from EEGModels import EEGNet
except:
    print("Have you put arg-eegmodels in the PATHONPATH?")
    raise
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
#
from predicament.utils.config import EVALUATION_BASE_PATH
from predicament.utils.config import RESULTS_BASE_PATH

def load_evaluation_data(
            subdir = '20230713194411/fold0'):
    #
    # data directory
    datadir = os.path.join(EVALUATION_BASE_PATH, subdir)
    #
    config = configparser.ConfigParser()
    details_fpath = os.path.join(datadir, 'details.cfg') 
    config.read(details_fpath)
    print(f"config = {config}")
    print(f"config.sections() = {config.sections()}")
    K.set_image_data_format('channels_last')
    # json converts string formatting of list to list (maybe json would be better
    # as a file for storing data details)
    conditions_str = config.get('LOAD','conditions')
    conditions_str = conditions_str.replace("'",'"')
    print(f"conditions_str = {conditions_str}")
    condition_list = json.loads(conditions_str)
    print(f"condition_list")
    nb_classes = len(condition_list)
    condition_map = { cond:i for i, cond in enumerate(condition_list)}
    print(f"condition_map = {condition_map}")
    n_channels = int(config.get('LOAD','n_channels'))
    print(f"n_channels = {n_channels}")
    # number of samples per datapoint
    n_samples = int(config.get('LOAD', 'window_size'))
    print(f"n_samples = {n_samples} (num samples per datapoint per channel)")
    
    # loading the data itself
    train_input_fpath = os.path.join(datadir, 'training_set.csv')
    train_label_fpath = os.path.join(datadir, 'training_label.csv')
    valid_input_fpath = os.path.join(datadir, 'test_set.csv')
    valid_label_fpath = os.path.join(datadir, 'test_label.csv')
    train_inputs = pd.read_csv(train_input_fpath, header=None)
    train_inputs = np.array(train_inputs).astype('float32')
    valid_inputs = pd.read_csv(valid_input_fpath, header=None)
    valid_inputs = np.array(valid_inputs).astype('float32')
    train_labels = pd.read_csv(train_label_fpath, header=None)
    train_labels = np.array(train_labels).astype('int32')
    valid_labels = pd.read_csv(valid_label_fpath, header=None)
    valid_labels = np.array(valid_labels).astype('int32')
    print(f"train_inputs.shape = {train_inputs.shape}")
    print(f"train_labels.shape = {train_labels.shape}")
    print(f"valid_inputs.shape = {valid_inputs.shape}")
    print(f"valid_labels.shape = {valid_labels.shape}")
    n_train = train_inputs.shape[0]
    train_inputs = train_inputs.reshape(n_train, n_channels, n_samples, 1)
    n_valid = valid_inputs.shape[0]
    valid_inputs = valid_inputs.reshape(n_valid, n_channels, n_samples, 1)
    train_labels = np_utils.to_categorical(train_labels, nb_classes)
    valid_labels = np_utils.to_categorical(valid_labels, nb_classes)
    print("After reshaping...")
    print(f"train_inputs.shape = {train_inputs.shape}")
    print(f"train_labels.shape = {train_labels.shape}")
    print(f"valid_inputs.shape = {valid_inputs.shape}")
    print(f"valid_labels.shape = {valid_labels.shape}")
    data_details = {}
    data_details['n_channels'] = n_channels
    data_details['n_samples'] = n_samples
    data_details['classes'] = condition_list
    data_details['nb_classes'] = nb_classes
    return train_inputs, train_labels, valid_inputs, valid_labels, data_details

# model setting
def run(
    train_inputs, train_labels, valid_inputs, valid_labels, epochs,
    n_channels=None, n_samples=None, nb_classes=None,
    batch_size=16, kernel_length=32, dropout_rate=0.5, subdir=None):
    model = EEGNet(
        nb_classes=nb_classes,
        Chans=n_channels,
        Samples = n_samples,
        dropoutRate = dropout_rate,
        kernLength = kernel_length,
        F1 = 8, D = 2, F2 = 16,
        dropoutType = 'Dropout')
    model.compile(
        loss='categorical_crossentropy', optimizer='adam',
        metrics = ['accuracy'])
#    tmp_file_path = r'./Dissertation/eegmodels/examples/tmp/checkpoint.h5'
    tmp_file_path = os.path.join(
        RESULTS_BASE_PATH, subdir, 'checkpoint.h5')
    checkpointer = ModelCheckpoint(filepath=tmp_file_path, verbose=1,
                                save_best_only=True)
    class_weights = {i:1 for i in range(nb_classes)}
    print(f"train_inputs.shape = {train_inputs.shape}")
    print(f"train_labels.shape = {train_labels.shape}")
    print(f"batch_size = {batch_size}")
    print(f"epochs = {epochs}")
    print(f"valid_inputs.shape = {valid_inputs.shape}")
    print(f"valid_labels.shape = {valid_labels.shape}")
    print(f"class_weights = {class_weights}")
    fittedModel = model.fit(
        train_inputs, train_labels, batch_size=batch_size, epochs=epochs,
        verbose = 2, validation_data=(valid_inputs, valid_labels),
        callbacks=[checkpointer], class_weight = class_weights)
    # save training result
    model.load_weights(tmp_file_path)
    res = pd.DataFrame({
        'acc': model.history.history['accuracy'],
        'val_acc': model.history.history['val_accuracy'],
        'loss': model.history.history['loss'],
        'val_loss': model.history.history['val_loss'],
    })
    res_fname = 'res_{}_{}_{}.csv'.format(
        dropout_rate, batch_size, kernel_length)
    res_fpath = os.path.join(
        RESULTS_BASE_PATH, subdir, res_fname)
    res.to_csv(res_fpath, index = None, encoding = 'utf8')
    probs       = model.predict(valid_inputs)
    preds       = probs.argmax(axis = -1)
    acc         = np.mean(preds == valid_labels.argmax(axis=-1))
    print("Classification accuracy: %f " % (acc))
    print()
    return acc

def grid_search(
        train_inputs, train_labels, valid_inputs, valid_labels,
        param_choices=None, **kwargs):
    """
    parameters
    ----------
    param_choices - is a dictionary whose keys are run argument parameter names and whose values are lists of values to try for that parameter.
    """
    param_names = list(param_choices.keys())
    print(f"param_names = {param_names}")
    param_lists = list([param_choices[k] for k in param_names])
    if len(param_lists) != len(param_names):
        raise ValueError('There must be one list of values per paarameter')
    for params in itertools.product(*param_lists):
        hyperparams = dict(zip(param_names, params))
        print(f"hyperparams = {hyperparams}")
        kwargs.update(hyperparams)
        print(f"kwargs = {kwargs}")
        run(
            train_inputs, train_labels, valid_inputs, valid_labels,
            **kwargs)
#    for batch_size in [16, 32]:
#        for dropout_rate in [0.5, 0.8]:
#            for kernel_length in [32, 64, 125]:
#                run(batch_size, kernel_length, dropout_rate)


def create_parser():
    import argparse
    description= """
        Provides runs a series of grid searches on different evaluation data"""
    parser = argparse.ArgumentParser(
        description=description,
        epilog='See git repository readme for more details.')

    parser.add_argument('subdirs', nargs='+', type=str,
        help='subfolders to run experiments on')
    return parser


if __name__ == '__main__':
#    subdir = '20230713194411/fold0'
    args = create_parser().parse_args()
    subdirs = vars(args)['subdirs']
    for subdir in subdirs:
        evaluation_data = load_evaluation_data(
            subdir=subdir)
        train_inputs, train_labels, valid_inputs, valid_labels = evaluation_data[:4]
        data_details = evaluation_data[-1]
        n_channels = data_details['n_channels']
        n_samples = data_details['n_samples'] 
        nb_classes = data_details['nb_classes']
        classes = data_details['classes']
        #   
        param_choices = {}
        param_choices['batch_size'] = [16, 32]
        param_choices['dropout_rate'] = [0.5, 0.8]
        param_choices['kernel_length'] = [32, 64, 125]
        grid_search(
            train_inputs, train_labels, valid_inputs, valid_labels,
            epochs=20, param_choices=param_choices,
            n_channels=n_channels, n_samples=n_samples,
            nb_classes=nb_classes,
            subdir=subdir)
