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
import logging
import logging.handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# always write everything to the rotating log files
if not os.path.exists('logs'): os.mkdir('logs')
log_file_handler = logging.handlers.TimedRotatingFileHandler('logs/args.log', when='M', interval=2)
log_file_handler.setFormatter( logging.Formatter('%(asctime)s [%(levelname)s](%(name)s:%(funcName)s:%(lineno)d): %(message)s') )
log_file_handler.setLevel(logging.DEBUG)
logger.addHandler(log_file_handler)

# also log to the console at a level determined by the --verbose flag
console_handler = logging.StreamHandler() # sys.stderr
console_handler.setLevel(logging.CRITICAL) # set later by set_log_level_from_verbose() in interactive sessions
console_handler.setFormatter( logging.Formatter('[%(levelname)s](%(name)s): %(message)s') )
logger.addHandler(console_handler)


# import data_setup
from predicament.utils.file_utils import load_dataframe_and_config
from predicament.utils.file_utils import write_dataframe_and_config
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
from predicament.data.features import SUPPORTED_FEATURE_GROUP
from predicament.data.features import convert_timeseries_to_features
from predicament.data.features import add_features_to_dataframe

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
    else:
        raise ValueError(f"Unrecognised data format {data_format}")
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
    """
    inputs
    ------
    participant_list - list of participants for whom data is to be loaded
    conditions - list of conditions for which the data is being loaded
    data_format -
        what is the raw data format type of the data to be loaded, e.g.
        dreem vs E4
    channels [optional] -
        which timeseries channels are to be included in output
    channel_group - 
        predefined set of channels by set name
    window_size - 
        size of window in samples
    window_step - 
        step size, in number of samples, between the start of the ith window and
        the start of the (i+1)th
    kwargs -
        input args that will be ignored [unsafe] TODO: make safe
    """
    participant_list, channels, window_size, window_step = resolve_participant_data_settings(
        data_format, participant_list, channel_group, channels, window_size, window_step)
    logger.info(f"resolved data settings: participant_list, channels, window_size, window_step = {(participant_list, channels, window_size, window_step)}")
    # load data from file
    all_participants_data = create_participant_data(
        participant_list=participant_list, data_format=data_format)
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
    loadargs = dict(
        participant_list=participant_list,
        conditions=conditions, channels=channels,
        n_channels=len(channels), sample_rate=sample_rate,
        window_size=window_size, window_step=window_step,
        label_mapping=label_mapping, 
        data_format=data_format)
    loadargs = { k:str(v) for k, v in loadargs.items()}
    logger.debug(f"loadargs = {loadargs}")
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

def write_data_by_participant_to_dataframe(
        data_by_participant, channels, window_size):
    timepoints = np.arange(window_size)
    label_cols = ['participant', 'condition', 'window index']
    window_columns = [ ch+f'[{t}]' for ch in channels for t in timepoints]
    # get windows and condition labels as two np arrays
    # stacked in participant order 
    windows_np, condition_labels_np = map(
        np.concatenate, 
        zip(*data_by_participant.values()))
    # windows and condition labels as dataframe and series object resp.
    windows = pd.DataFrame(windows_np, columns=window_columns)
    # to produce column of participant labels first we need to know how many
    # windows for each participant
    window_counts = {
        prt: len(cnds) for prt,(wndws,cnds) in data_by_participant.items()}
    # producing the participant labels
    participant_labels_np = np.repeat(
        list(window_counts), list(window_counts.values()))
    participant_condition_labels = pd.DataFrame(
        np.vstack((participant_labels_np, condition_labels_np.astype(int))).T,
        columns=('participant','condition') )
    # the participant labels then provide the window indices with the
    # groupby...cumcount combined call
    window_indices = participant_condition_labels.groupby(
        ["participant", "condition"]).cumcount().rename('window index')
    # finally we horizontally stack the first three series as label columns
    # and the windows dataframe
    windowed_df = pd.concat(
        [participant_condition_labels, window_indices, windows],
        axis=1)
    # the first three columns are label columns
    label_cols = list(windowed_df.columns[:3])
    return windowed_df, label_cols
    ## older buggy version
#    columns = label_cols + window_columns 
#    window_participant_iterator = tqdm(
#        data_by_participant.items(), total=len(data_by_participant))
#    lst = []
#    for participant, (part_windows, part_conditions) \
#            in window_participant_iterator:
#        uniques, counts = np.unique(
#                part_conditions, return_counts=True)
#        window_indices = np.concatenate(
#            [np.arange(count) for count in counts ])
#        this_N, this_K = part_windows.shape
#        these_participants = [participant]*this_N
#        this_df = pd.DataFrame(
#            zip(
#                these_participants, # participant
#                part_conditions, # condition
#                window_indices, # window index
#                *part_windows.T.tolist()
#                ), columns=columns)
#        lst.append(this_df)
#    timeseries_df = pd.concat(lst)
#    return timeseries_df, label_cols

    
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
    logger.info(f"Loaded participant-wise data")
    logger.info(f"Fs: {Fs}, n_samples = {window_size}, time: {time}s, n_channels: {n_channels}")
    # create a dataframe from this data
    timeseries_df, label_cols = write_data_by_participant_to_dataframe(
        data_by_participant, channels, window_size)
    # save down to file.
    windowed_dir_path = os.path.join(parent_path, subdir)
    config['WINDOWED'] = {}
    config['WINDOWED']['group_col'] = 'participant'
    config['WINDOWED']['target_col'] = 'condition'
    config['WINDOWED']['label_cols'] = str(label_cols).replace("'",'"')
    write_dataframe_and_config(
        windowed_dir_path, timeseries_df, config, windowed_fname)

def prepare_feature_data(
        subdir, windowed_fname='windowed.csv', featured_fname='featured.csv',
        feature_set=None, feature_group=None,
        bits_per=1, lzc_type=None, lzc_imp=None,
        entropy_tols=None, hurst_kind='random_walk', **kwargs):
    if feature_set is None:
        if feature_group == 'stats':
            feature_set = STATS_FEATURE_GROUP
        elif feature_group == 'info':
            feature_set = INFO_FEATURE_GROUP
        elif feature_group == 'freq':
            feature_set = FREQ_FEATURE_GROUP
        elif feature_group == 'supported':
            feature_set = SUPPORTED_FEATURE_GROUP
        else:
            raise ValueError(f'Unrecognised feature group {feature_group}')
    else:
        feature_set = set(feature_set.split(','))
        if len(feature_set) == 0:
            raise ValueError('')
    # load windowed data and get the relevant configuration
    windowed_parent_path = get_parent_path('windowed', subdir)
    logger.info("Loading windowed data")
    windowed_df, windowed_config = load_dataframe_and_config(
        windowed_parent_path, windowed_fname)
    n_channels = int(windowed_config['LOAD']['n_channels'])
    channels = json.loads(windowed_config['LOAD']['channels'].replace("'",'"'))
    participant_list = json.loads(windowed_config['LOAD']['participant_list'].replace("'",'"'))
    Fs = int(windowed_config['LOAD']['sample_rate'])
    window_size = int(windowed_config['LOAD']['window_size'])
    time = window_size/Fs
    logger.info(
        f"Fs: {Fs}, n_samples = {window_size}, time: {time}s, n_channels: {n_channels}")
    label_cols = json.loads(windowed_config['WINDOWED']['label_cols'].replace("'",'"') )
        
    #
    timeseries_cols = [col for col in windowed_df.columns if not col in label_cols ]
    label_df = windowed_df[label_cols]
    windowed_timeseries_df = windowed_df[timeseries_cols]
    logger.info(
        f"Creating featured list-of-lists for features:\n\t{feature_set}")
    # convert to numpy array and reshape into tensor of shape NxCxW
    # where N is number of windows, C is number of channels and W is window size
    windowed_data = windowed_timeseries_df.to_numpy()
    windowed_data = windowed_data.reshape((-1,n_channels,window_size))
    # get per channel means mins and maxes across dataset
    amp_means = np.mean(np.mean(windowed_data, axis=2), axis=0)
    amp_mins = np.min(np.min(windowed_data, axis=2), axis=0)
    amp_maxes = np.max(np.max(windowed_data, axis=2), axis=0)
    ## default value for entropy_tols
    if entropy_tols is None:
        entropy_tols = 0.2*np.mean(np.std(windowed_data
                , axis=2), axis=0)
        logger.debug(f"inferred entropy_tols = {entropy_tols}")
    # creating a list of list of features, one list per window
    logger.info(f"Creating featured list-of-lists for features:\n\t{feature_set}")
    features_lol = []
    for index in tqdm(np.arange(len(windowed_df))):
        label_row = label_df.iloc[index].to_list()
        value_row = windowed_timeseries_df.iloc[index].to_numpy()
        X = value_row.reshape((n_channels,-1))
        features, feature_names = convert_timeseries_to_features(
            X, feature_set,
            amp_means=amp_means, amp_mins=amp_mins, amp_maxes=amp_maxes,
            bits_per=bits_per, lzc_type=lzc_type, lzc_imp=lzc_imp,
            entropy_tols=entropy_tols, hurst_kind=hurst_kind)
        features_lol.append(        
                label_row + list(features))
                
    ## we will now construct a dataframe of the features
    # featured data may or may not pre-exist, set corresponding DataFrame
    # to None if it does not
    featured_parent_path = get_parent_path('featured', subdir)
    logger.info("Loading pre-existing featured data if it exists")
    try:
        featured_df, featured_config = load_dataframe_and_config(
            featured_parent_path, featured_fname)
    except FileNotFoundError:
        featured_df = None
        featured_config = windowed_config
    logger.info("Building new featured dataframe")
    # the columns of the new dataframe will include the label columns
    # and all the "new" feature columns
    all_columns = label_cols + feature_names
    chunk_indices = np.linspace(0,len(features_lol),1001).astype(int)
    chunk_slices = zip(chunk_indices[:-1], chunk_indices[1:])
    n_chunks = chunk_indices.size-1
    for i, (start, end) in enumerate(tqdm(chunk_slices, total=n_chunks)):
        if i == 0:
            new_featured_df = pd.DataFrame(columns=all_columns)
        if start != end:
            new_featured_df = pd.concat(
                (featured_df, pd.DataFrame(features_lol[start:end], columns=all_columns)),
                ignore_index=True)
    if featured_df is None:
        featured_df = new_featured_df
    else:
        #
        logger.info("Merging new featured dataframe with pre-existing dataframe")
        featured_df = add_features_to_dataframe(
            featured_df, new_featured_df, label_cols)
        # incorporate pre-existing feature names and feature_set in config
        old_feature_names = json.loads(
            featured_config['FEATURED']['feature_names'].replace("'",'"'))
        old_feature_set = json.loads(
            featured_config['FEATURED']['feature_set'].replace("'",'"'))
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
        '--feature-group', default='supported',
        help="""Predefined group of features to use""")
    parser.add_argument(
        '--feature-set',
        help="""Specify feature types as comma separated string""")
    parser.add_argument(
        '--bits-per', default=1, type=int,
        help="""The number of bits for quantized vectors, for use with Lempel Ziv calculations""")
    parser.add_argument(
        '--lzc-type', default='casali',
        help="""The general algorithm for calculating LempelZiv Complexity""")
    parser.add_argument(
        '--lzc-imp', default='flow',
        help="""The specific implementation of LempelZiv Complexity type""")
    parser.add_argument(
        '-f', '--data-format', default='dreem', choices=['dreem','E4'])
    parser.add_argument(
        '-g', '--channel-group')
    parser.add_argument(
        '-c', '--channels')
    parser.add_argument(
        '-w', '--window-size', type=int)
    parser.add_argument(
        '-s', '--window-step', type=int)
        
    # general        
    parser.add_argument('-V', '--version', action="version", version="%(prog)s 0.1")
    parser.add_argument('-v', '--verbose', action="count", help="verbose level... repeat up to three times.")
        
    return parser

def set_log_level_from_verbose(args):

    if not args.verbose:
        console_handler.setLevel('ERROR')
    elif args.verbose == 1:
        console_handler.setLevel('WARNING')
    elif args.verbose == 2:
        console_handler.setLevel('INFO')
    elif args.verbose >= 3:
        console_handler.setLevel('DEBUG')
    else:
        logger.critical("UNEXPLAINED NEGATIVE COUNT!")
        
if __name__ == '__main__':
    args = create_parser().parse_args()
    set_log_level_from_verbose(args)
    kwargs = vars(args)
    kwargs.pop('verbose')
    main(**kwargs)
    
