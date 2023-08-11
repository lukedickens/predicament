# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 18:09:32 2022

@author: Zerui Mu
@author: Luke Dickens
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import configparser

# import data_setup
from predicament.utils.config import DREEM_EEG_CHANNELS, DREEM_MINIMAL_CHANNELS
from predicament.utils.config import TARGET_CONDITIONS
from predicament.utils.config import DEFAULT_WINDOW_SIZE
from predicament.utils.config import DEFAULT_WINDOW_STEP
from predicament.utils.config import EVALUATION_BASE_PATH
from predicament.utils.config import FULL_PARTICIPANT_LIST

from predicament.data.timeseries import create_participant_data_edf_only
from predicament.data.windowed import window_all_participants_data
from predicament.data.windowed import merge_condition_data
from predicament.data.partitioning import between_subject_cv_partition


from predicament.utils import file_utils

def load_labelled_data(
        participant_list=FULL_PARTICIPANT_LIST,
        conditions=TARGET_CONDITIONS, channels=None,
        channel_group=None,
        window_size=DEFAULT_WINDOW_SIZE, window_step=None):
    if window_step is None:
        window_step = DEFAULT_WINDOW_SIZE//8
    if channels is None:
        if channel_group is None or channel_group == 'dreem-minimal':
            channels=DREEM_MINIMAL_CHANNELS
        elif channel_group == 'dreem-eeg':
            channels=DREEM_EEG_CHANNELS
    else:
        channels = channels.split(',')
            
    # load data from file
    all_participants_data = create_participant_data_edf_only(
        participant_list=participant_list)
    # window data by specified size and step
    all_windowed_data = window_all_participants_data(
            all_participants_data, conditions, channels, window_size,
            window_step, condition_fragile=False, channel_fragile=False,
            copy=False)
    data_by_participant, label_mapping = merge_condition_data(
        all_windowed_data)
    # record information about data
    first_ID = list(all_participants_data.keys())[0]
    # all sample rates are the same
    sample_rate = all_participants_data[first_ID].sample_rate
    config = configparser.ConfigParser()
    config['LOAD'] = dict(
        participant_list=participant_list,
        conditions=conditions, channels=channels,
        n_channels=len(channels), sample_rate=sample_rate,
        window_size=window_size, window_step=window_step,
        label_mapping=label_mapping)
        
    return data_by_participant, config
    
    
def iterate_between_subject_cv_partition(**loadargs):
    data_by_participant, config = load_labelled_data(**loadargs)
    config['PARTITION'] = {'type':'between_subject'}
    for fold in between_subject_cv_partition(data_by_participant):
        tr_dt, tr_lb, ts_dt, ts_lb, held_out_ID = fold
        config['PARTITION']['held_out_ID'] = held_out_ID
        yield (tr_dt, tr_lb, ts_dt, ts_lb, config)


def prepare_between_subject_cv_partition_files(**loadargs):
    # datetime object containing current date and time
    now = datetime.now()
     
    print("now =", now)

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%Y%m%d%H%M%S")
    print("date and time =", dt_string)
    parent_path = os.path.join(EVALUATION_BASE_PATH,dt_string)
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    print(f"saving to subfolders of {parent_path}")


    for f,fold in enumerate(iterate_between_subject_cv_partition(**loadargs)):
        trn_dat, trn_lab, tst_dat, tst_lab, config = fold
        n_train = trn_dat.shape[0]
        n_test = tst_dat.shape[0]
        # fold path is subdirectory of parent
        fold_path = os.path.join(parent_path, f'fold{f}')
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)
        print(f"Saving fold {f} data in {fold_path} with {n_train} train and {n_test} test points")
        # reshape label arrays
        trn_lab = np.reshape(trn_lab, (-1,1))
        tst_lab = np.reshape(tst_lab, (-1,1))
        # save config file with the 
        config['PARTITION']['within_or_between'] = 'between'
        config['PARTITION']['fold'] = str(f)
        config['PARTITION']['n_train'] = str(n_train)
        config['PARTITION']['n_test'] = str(n_test)
        config_fpath = os.path.join(fold_path, 'details.cfg')
        with open(config_fpath, 'w') as configfile:
            config.write(configfile)
        np.savetxt(
            os.path.join(fold_path, "training_set.csv"),
            trn_dat, delimiter=",")
        np.savetxt(
            os.path.join(fold_path, "training_label.csv"),
            trn_lab, delimiter=",")
        np.savetxt(
            os.path.join(fold_path, "test_set.csv"),
            tst_dat, delimiter=",")
        np.savetxt(
            os.path.join(fold_path, "test_label.csv"),
            tst_lab, delimiter=",")


def main(within_or_between='between', **kwargs):
    if within_or_between=='between':
        prepare_between_subject_cv_partition_files(**kwargs)
    elif within_or_between=='within':
        raise NotImplementedError('Within partition not yet implemented')
        
def create_parser():
    import argparse
    description= """
        Loads study data, then windows and (typically) partitions it according
        to the configuration parameters then saves down to file."""
    parser = argparse.ArgumentParser(
        description=description,
        epilog='See git repository readme for more details.')
    # partition arguments
    # specify within or between
    group = parser.add_mutually_exclusive_group()
    group.set_defaults(within_or_between="between")
    group.add_argument(
        '--between', dest='within_or_between',
        action='store_const', const='between')
    group.add_argument(
        '--within', dest='within_or_between',
        action='store_const', const='within')
    # other parameters to control loaded data
    parser.add_argument(
        '-g', '--channel-group', default='dreem-minimal')
    parser.add_argument(
        '-c', '--channels')
    parser.add_argument(
        '-w', '--window-size', type=int, default=DEFAULT_WINDOW_SIZE)
    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()
    kwargs = vars(args)
    main(**kwargs)
    
