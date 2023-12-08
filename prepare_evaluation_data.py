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
import json
from tqdm import tqdm
import copy

# import data_setup
from predicament.utils.config import DREEM_EEG_CHANNELS, DREEM_MINIMAL_CHANNELS
from predicament.utils.config import DREEM_INFORMING_CHANNELS
from predicament.utils.config import TARGET_CONDITIONS
from predicament.utils.config import DEFAULT_DREEM_WINDOW_SIZE
from predicament.utils.config import DEFAULT_E4_WINDOW_SIZE
from predicament.utils.config import DEFAULT_WINDOW_OVERLAP_FACTOR
from predicament.utils.config import CROSS_VALIDATION_BASE_PATH
from predicament.utils.config import WINDOWED_BASE_PATH
from predicament.utils.config import FEATURED_BASE_PATH
from predicament.utils.config import DREEM_PARTICIPANT_IDS
from predicament.utils.config import E4_PARTICIPANT_IDS
from predicament.utils.config import DEFAULT_E4_CHANNELS

from predicament.data.timeseries import create_participant_data
from predicament.data.windowed import window_all_participants_data
from predicament.data.windowed import merge_condition_data
from predicament.data.partitioning import between_subject_cv_partition

from predicament.data.features import MAXIMAL_FEATURE_GROUP
from predicament.data.features import STATS_FEATURE_GROUP
from predicament.data.features import INFO_FEATURE_GROUP
from predicament.data.features import FREQ_FEATURE_GROUP
from predicament.data.features import convert_timeseries_to_features

from predicament.utils import file_utils

def get_datetime_string():
    now = datetime.now()
    return now.strftime("%Y%m%d%H%M%S")

def get_parent_path(datatype, subdir):
    if datatype == 'cross_validation':
        base_path = CROSS_VALIDATION_BASE_PATH
    elif datatype == 'windowed':
        base_path = WINDOWED_BASE_PATH
    elif datatype == 'featured':
        base_path = FEATURED_BASE_PATH
    #
    parent_path = os.path.join(base_path, subdir)
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    return parent_path

def write_dataframe_and_config(
        dir_path, data, config, data_fname, config_fname='details.cfg'):
    data_path = os.path.join(dir_path, data_fname)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    print(f"writing data to {data_path}")
    # write with progress bar https://stackoverflow.com/questions/64695352/pandas-to-csv-progress-bar-with-tqdm 
    data.to_csv(data_path)
##    chunk_indices = np.linspace(0,len(data), 101)
##    chunk_slices = zip(chunk_indices[:-1], chunk_indices[1:])
##    for i, (start, end) in enumerate(tqdm(chunk_slices)):
##        if i == 0: # first row
##            data.loc[start:end].to_csv(data_path, mode='w', index=True)
##        else:
##            data.loc[start:end].to_csv(data_path, header=None, mode='a', index=True)
#    chunks = np.array_split(data.index, 100) # split into 100 chunks
#    for i, chunk in enumerate(tqdm(chunks)):
#        if i == 0: # first row
#            data.loc[chunk].to_csv(data_path, mode='w', index=True)
#        else:
#            data.loc[chunk].to_csv(data_path, header=None, mode='a', index=True)

    config_path = os.path.join(dir_path, config_fname)
    print(f"writing config to {config_path}")
    with open(config_path, 'w') as config_file:
        config.write(config_file)

def load_dataframe_and_config(
        dir_path, data_fname, config_fname='details.cfg', **readargs):
#    print(f"dir_path = {dir_path}")
#    print(f"data_fname = {data_fname}")
    data_path = os.path.join(dir_path, data_fname)
    data = pd.read_csv(data_path, index_col=0,  **readargs)
    config_path = os.path.join(dir_path, config_fname)
    config = configparser.ConfigParser()
    with open(config_path, 'r') as config_file:
        config.read_file(config_file)
    return data, config


def resolve_participant_data_settings(
        data_format, participant_list, channel_group, channels, window_size, window_step):
#    print(f"data_format, channel_group, channels, window_size, window_step = {(data_format, channel_group, channels, window_size, window_step)}")
    if data_format == 'dreem':
        if participant_list is None:
            participant_list=DREEM_PARTICIPANT_IDS
        if channels is None:
            if (channel_group is None) or (channel_group == 'dreem-informing'):
                channels=DREEM_MINIMAL_CHANNELS
            elif channel_group == 'dreem-minimal':
                channels=DREEM_INFORMING_CHANNELS
            elif channel_group == 'dreem-eeg':
                channels=DREEM_EEG_CHANNELS
            else:
                raise ValueError(f"Unrecognised channel group {channel_group}")
        elif type(channels) is str:
            channels = get_channels_from_string(channels)
        else:
            channels = list(channels)
        if window_size is None:
            window_size = DEFAULT_DREEM_WINDOW_SIZE
    elif data_format == 'E4':
        if participant_list is None:
            participant_list=E4_PARTICIPANT_IDS
        if channels is None:
            if (channel_group is None) or (channel_group == 'E4'):
                channels=DEFAULT_E4_CHANNELS
            else:
                raise ValueError("Unrecognised channel group")
        elif type(channels) is str:
            channels = get_channels_from_string(channels)
        else:
            channels = list(channels)
        if window_size is None:
            window_size = DEFAULT_E4_WINDOW_SIZE
    if window_step is None:
        window_step = window_size//DEFAULT_WINDOW_OVERLAP_FACTOR
    return participant_list, channels, window_size, window_step
    
def get_channels_from_string(channels_as_str):
    channels = channels_as_str.split(',')
    return channels




def load_and_window_participant_data(
        participant_list=None,
        conditions=TARGET_CONDITIONS, data_format=None, 
        channels=None, channel_group=None,
        window_size=None, window_step=None, **kwargs):

    participant_list, channels, window_size, window_step = resolve_participant_data_settings(
        data_format, participant_list, channel_group, channels, window_size, window_step)
    print(f"resolved data settings: participant_list, channels, window_size, window_step = {(participant_list, channels, window_size, window_step)}")
    # load data from file
    all_participants_data = create_participant_data(
        participant_list=participant_list, data_format=data_format)
#    print(f"all_participants_data.keys() = {list(all_participants_data.keys())}")
    # window data by specified size and step
    all_windowed_data = window_all_participants_data(
            all_participants_data, conditions, channels, window_size,
            window_step, condition_fragile=False, channel_fragile=False,
            copy=False)
#    print(f"all_windowed_data.keys() = {list(all_windowed_data.keys())}")
    data_by_participant, label_mapping = merge_condition_data(
        all_windowed_data)   
#    print(f"data_by_participant.keys() = {list(data_by_participant.keys())}")
    # record information about data
    first_ID = list(all_participants_data.keys())[0]
    # all sample rates are the same
    sample_rate = all_participants_data[first_ID].sample_rate
    config = configparser.ConfigParser()
    loadargs = dict(
        participant_list=participant_list,
        conditions=conditions, channels=channels,
        n_channels=len(channels), sample_rate=sample_rate,
        window_size=window_size, window_step=window_step,
        label_mapping=label_mapping, 
        data_format=data_format)
    loadargs = { k:str(v) for k, v in loadargs.items()}
    print(f"loadargs = {loadargs}")
    config['LOAD'] = loadargs        
    return data_by_participant, config
    
    
def iterate_between_subject_cv_partition(**loadargs):
    data_by_participant, config = load_and_window_participant_data(**loadargs)
    config['PARTITION'] = {'type':'between_subject'}
    for fold in between_subject_cv_partition(data_by_participant):
        tr_dt, tr_lb, ts_dt, ts_lb, held_out_ID = fold
        config['PARTITION']['held_out_ID'] = held_out_ID
        yield (tr_dt, tr_lb, ts_dt, ts_lb, config)


def prepare_between_subject_cv_partition_files(**loadargs):
    # datetime object containing current date and time
    now = datetime.now()
     
#    print("now =", now)

    # dd/mm/YY H:M:S
    dt_string = get_datetime_string()
#    print("date and time =", dt_string)
    parent_path = os.path.join(CROSS_VALIDATION_BASE_PATH,dt_string)
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
#    print(f"saving to subfolders of {parent_path}")


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


def consolidate_windowed_data(
        parent_path=WINDOWED_BASE_PATH, subdir=None,
        windowed_fname='windowed.csv', **loadargs):
    # can specify the sub directory otherwise a datetime-string is created
    if subdir is None:
        subdir = get_datetime_string()

    # prepare and save down time-series
    data_by_participant, config = load_and_window_participant_data(**loadargs)
    n_channels = int(config['LOAD']['n_channels'])
    channels = json.loads(config['LOAD']['channels'].replace("'",'"'))
    participant_list = json.loads(config['LOAD']['participant_list'].replace("'",'"'))
    Fs = int(config['LOAD']['sample_rate'])
    window_size = int(config['LOAD']['window_size'])
    time = window_size/Fs
    print(f"Loaded participant-wise data")
    print(f"Fs: {Fs}, n_samples = {window_size}, time: {time}s, n_channels: {n_channels}")
    print(f"Creating consolidated list-of-lists of windowed data:")
    timeseries_lol = []
    for part_ID, (part_data, part_conditions) in tqdm(data_by_participant.items(), total=len(data_by_participant)):
#        print(f"For {part_ID} have data:\t{part_data}")
#        print(f"With conditions:\t{part_conditions}")
        part_conditions = part_conditions.astype(int)
        last_condition = part_conditions[0]
        index = 0
        for row, condition in zip(part_data, part_conditions):
            timeseries_lol.append([part_ID, condition, index] + list(row))
            if (condition != last_condition):
                index += window_size
            else:
                index = 0
    #
    print(f"Constructing dataframe...")
    timepoints = np.arange(row.size//n_channels)
    label_cols = ['part_ID', 'condition', 'start time']
    columns = label_cols + [ ch+f'[{t}]' for ch in channels for t in timepoints]
    #timeseries_df = pd.DataFrame(timeseries_lol, columns=columns)
    #timeseries_lol_chunks = np.array_split(timeseries_lol, 100)
    n_rows = len(timeseries_lol)
    chunk_indices = np.linspace(0,n_rows,101).astype(int)
    timeseries_df = pd.DataFrame(columns=columns)
    for start, end in tqdm(
            zip(chunk_indices[:-1], chunk_indices[1:]),
            total=chunk_indices.size-1):
        timeseries_df = pd.concat(
            (timeseries_df, pd.DataFrame(timeseries_lol[start:end], columns=columns)),
            ignore_index=True)
    # save down to file.
    windowed_dir_path = os.path.join(parent_path, subdir)
    config['WINDOWED'] = {}
    config['WINDOWED']['group_col'] = 'part_ID'
    config['WINDOWED']['target_col'] = 'condition'
    config['WINDOWED']['label_cols'] = str(label_cols).replace("'",'"')
    write_dataframe_and_config(
        windowed_dir_path, timeseries_df, config, windowed_fname)

def prepare_feature_data(
        subdir, windowed_fname='windowed.csv', featured_fname='featured.csv',
        feature_set=None, feature_group=None,
        entropy_tol=0.6, hurst_kind='random_walk', **kwargs):
    if feature_set is None:
        if feature_group == 'stats':
            feature_set = STATS_FEATURE_GROUP
        elif feature_group == 'info':
            feature_set = INFO_FEATURE_GROUP
        elif feature_group == 'freq':
            feature_set = FREQ_FEATURE_GROUP
        else:
            raise ValueError(f'Unrecognised feature group {feature_group}')
    else:
        feature_set = set(feature_set.split(','))
        if len(feature_set) == 0:
            raise ValueError('')
    # 
    windowed_parent_path = get_parent_path('windowed', subdir)
    print("Loading windowed data")
    windowed_df, windowed_config = load_dataframe_and_config(
        windowed_parent_path, windowed_fname)
    n_channels = int(windowed_config['LOAD']['n_channels'])
    channels = json.loads(windowed_config['LOAD']['channels'].replace("'",'"'))
    participant_list = json.loads(windowed_config['LOAD']['participant_list'].replace("'",'"'))
    Fs = int(windowed_config['LOAD']['sample_rate'])
    window_size = int(windowed_config['LOAD']['window_size'])
    time = window_size/Fs
    print(f"Fs: {Fs}, n_samples = {window_size}, time: {time}s, n_channels: {n_channels}")
    label_cols = json. loads(windowed_config['WINDOWED']['label_cols'].replace("'",'"') )
    # featured data may or may not pre-exist, set corresponding DataFrame
    # to None if it does not
    featured_parent_path = get_parent_path('featured', subdir)
    print("Loading pre-existing featured data if it exists")
    try:
        featured_df, featured_config = load_dataframe_and_config(
            featured_parent_path, featured_fname)
    except FileNotFoundError:
        featured_df = None
        featured_config = windowed_config
    #
    timeseries_cols = [col for col in windowed_df.columns if not col in label_cols ]
    label_df = windowed_df[label_cols]
    windowed_timeseries_df = windowed_df[timeseries_cols]
    print(f"Creating featured list-of-lists for features:\n\t{feature_set}")
    features_lol = []
    for index in tqdm(np.arange(len(windowed_df))):
        label_row = label_df.iloc[index].to_list()
        value_row = windowed_timeseries_df.iloc[index].to_numpy()
        X = value_row.reshape((n_channels,-1))
        features, feature_names = convert_timeseries_to_features(
            X, feature_set, entropy_tol=entropy_tol, hurst_kind=hurst_kind)
        features_lol.append(        
                label_row + list(features))
    all_columns = label_cols + feature_names
    print(f"Creating featured list-of-lists for features:\n\t{feature_set}")
    if featured_df is None:
        chunk_indices = np.linspace(0,len(features_lol),1001).astype(int)
        chunk_slices = zip(chunk_indices[:-1], chunk_indices[1:])
        n_chunks = chunk_indices.size-1
        for i, (start, end) in enumerate(tqdm(chunk_slices, total=n_chunks)):
            if i == 0:
                featured_df = pd.DataFrame(columns=all_columns)
            if start != end:
                featured_df = pd.concat(
                    (featured_df, pd.DataFrame(features_lol[start:end], columns=all_columns)),
                    ignore_index=True)
#        featured_config['FEATURED'] = {}
#        featured_config['FEATURED']['feature_set'] = str(list(feature_set)).replace("'",'"')
#        featured_config['FEATURED']['features'] = str(feature_names).replace("'",'"')
    else:
        features_only_lol = [ row[len(label_cols):] for row in features_lol ]
        chunk_indices = np.linspace(0,len(features_lol),1001).astype(int)
        chunk_slices = zip(chunk_indices[:-1], chunk_indices[1:])
        n_chunks = chunk_indices.size-1
        # create blank columns for all the new features
        print(f"label_cols = {label_cols}")
        print(f"feature_names = {feature_names}")
        print(f"len(feature_names) = {len(feature_names)}")
        print(f"len(features_only_lol) = {len(features_only_lol)}")
        print(f"len(features_only_lol[0]) = {len(features_only_lol[0])}")
        featured_df[feature_names] = None
        # if using iloc need column indexes so
        feature_indexes = [featured_df.columns.get_loc(c) for c in feature_names]
        for start, end in tqdm(chunk_slices, total=n_chunks):
            if start != end:
                # loc is end-inclusive slicing, so we need end-1
                #featured_df.loc[start:end-1,feature_names] = features_only_lol[start:end]
                # iloc is end-exclusive slicing so we need just end
                # but we need the ids of the columns
                featured_df.iloc[start:end,feature_indexes] = features_only_lol[start:end]
        # incorporate pre-existing feature names and feature_set in config
        old_feature_names = json.loads(featured_config['FEATURED']['feature_names'].replace("'",'"'))
        old_feature_set = json.loads(featured_config['FEATURED']['feature_set'].replace("'",'"'))
        feature_set |= set(old_feature_set)
        feature_names = old_feature_names + list(feature_names)
    featured_config['FEATURED'] = {}
    featured_config['FEATURED']['feature_set'] = str(list(feature_set)).replace("'",'"')
    featured_config['FEATURED']['feature_names'] = str(feature_names).replace("'",'"')
    # update featured_config with correct details
    # save down to file.
    write_dataframe_and_config(
        featured_parent_path, featured_df, featured_config, 'featured.csv')




def main(mode=None, within_or_between='between', **kwargs):
    if mode == 'cross_validation':
        if within_or_between=='between':
            prepare_between_subject_cv_partition_files(**kwargs)
        elif within_or_between=='within':
            raise NotImplementedError('Within partition not yet implemented')
    elif mode == 'windowed':
        consolidate_windowed_data(**kwargs)
    elif mode == 'featured':
        prepare_feature_data(**kwargs)
    else:
        raise ValueError(f"Unrecognised preparation mode {mode}")
    
            
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
        '--mode',
        help="""How to prepare:
            cross_validation (save down as separate cross_validation folds)
            windowed (consolidate windowed time-series data in single file)
            featured (extract features from windowed time-series data)
            """)
    parser.add_argument(
        '--subdir',
        help="""Data subdirectory to use. TYpically a datetime string""")
    parser.add_argument(
        '--feature-group', default='stats',
        help="""Predefined group of features to use""")
    parser.add_argument(
        '--feature-set',
        help="""Specify feature types as comma separated string""")
    parser.add_argument(
        '-f', '--data-format', default='dreem')
    parser.add_argument(
        '-g', '--channel-group')
    parser.add_argument(
        '-c', '--channels')
    parser.add_argument(
        '-w', '--window-size', type=int)
    parser.add_argument(
        '-s', '--window-step', type=int)
    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()
    kwargs = vars(args)
    main(**kwargs)
    
