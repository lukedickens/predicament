# -*- coding: utf-8 -*-
"""
Utility functions
"""

import random
import time
import os
import pandas as pd
import numpy as np
import configparser

import logging
logger = logging.getLogger(__name__)

from predicament.utils.config import WINDOWED_BASE_PATH
from predicament.utils.config import FEATURED_BASE_PATH

from predicament.utils.config import CROSS_VALIDATION_BASE_PATH
from predicament.utils.config_parser import config_to_dict
from predicament.utils.config_parser import dict_to_config

from predicament.utils.dataframe_utils import drop_inf_cols
from predicament.utils.dataframe_utils import drop_nan_cols

def local2unix(datetime):
    """
    Convert date-time stamps in datafiles to unix times.
    """
    timeArray = time.strptime(datetime, "%d-%m-%Y %H:%M:%S")
    timestamp = time.mktime(timeArray)
    return timestamp

def unix2local(timestamp):
    """
    Convert unix times to date-time stamps in datafiles.
    """
    time_local = time.localtime(timestamp)
    datetime = time.strftime("%d-%m-%Y %H:%M:%S", time_local)
    return datetime

def gen_train_test_valid(data, test_ratio = 0.2, val_ratio = 0.2):
    """
    """
    if test_ratio + val_ratio > 0.5:
        print("Too much test & validation data!")
        return None, None, None
    part_num = len(data.keys())
    if test_ratio == 0.0:
        test_part_num = 0
    elif round(part_num * test_ratio) != 0:
        test_part_num = round(part_num * test_ratio)
    else: 
        test_part_num = 1
    if val_ratio == 0.0:
        val_part_num = 0
    elif round(part_num * val_ratio) != 0:
        val_part_num = round(part_num * val_ratio)
    else:
        val_part_num = 1
    # test_part_num = round(part_num * test_ratio) if round(part_num * test_ratio) != 0 else 1
    # val_part_num = round(part_num * val_ratio) if round(part_num * val_ratio) != 0 else 1

    test_part_list = random.sample(list(data.keys()), test_part_num)
    rest_part = [part for part in data.keys() if part not in test_part_list]
    val_part_list = random.sample(rest_part, val_part_num)
    train_part_list = [part for part in rest_part if part not in val_part_list]
    return train_part_list, test_part_list, val_part_list

def _test_local2unix():
    print(local2unix("08-09-2021 15:44:46"))

def _test_unix2local():
    print(unix2local(1631112286))
    
def get_evaluation_datadir(n_classes, func):
    data_folder = os.path.join(MOTOR_MOVEMENT_DATA_FOLDER, './Ray/{}classes_{}/'.format(n_classes, func))

def write_windowed_dataframe_and_config(subdir, df, config, data_fname='windowed.csv', **kwargs):
    data_dir = os.path.join(WINDOWED_BASE_PATH,subdir)
    write_dataframe_and_config(
        data_dir, df, config, data_fname, **kwargs)

def write_featured_dataframe_and_config(subdir, df, config, data_fname='featured.csv', **kwargs):
    data_dir = os.path.join(FEATURED_BASE_PATH,subdir)
    write_dataframe_and_config(
        data_dir, df, config, data_fname, **kwargs)

def write_dataframe_and_config(
        dir_path, df, config, data_fname, config_fname='details.cfg'):
    if type(config) is dict:
        config = dict_to_config(config)
    data_path = os.path.join(dir_path, data_fname)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    print(f"writing data to {data_path}")
    df.to_csv(data_path)
    # write with progress bar https://stackoverflow.com/questions/64695352/pandas-to-csv-progress-bar-with-tqdm 
##    chunk_indices = np.linspace(0,len(df), 101)
##    chunk_slices = zip(chunk_indices[:-1], chunk_indices[1:])
##    for i, (start, end) in enumerate(tqdm(chunk_slices)):
##        if i == 0: # first row
##            df.loc[start:end].to_csv(data_path, mode='w', index=True)
##        else:
##            df.loc[start:end].to_csv(data_path, header=None, mode='a', index=True)
#    chunks = np.array_split(df.index, 100) # split into 100 chunks
#    for i, chunk in enumerate(tqdm(chunks)):
#        if i == 0: # first row
#            df.loc[chunk].to_csv(data_path, mode='w', index=True)
#        else:
#            df.loc[chunk].to_csv(data_path, header=None, mode='a', index=True)

    config_path = os.path.join(dir_path, config_fname)
    print(f"writing config to {config_path}")
    with open(config_path, 'w') as config_file:
        config.write(config_file)

def load_windowed_dataframe_and_config(subdir, data_fname='windowed.csv', **kwargs):
    data_dir = os.path.join(WINDOWED_BASE_PATH,subdir)
    df, config = load_dataframe_and_config(
        data_dir, data_fname, **kwargs)
    return df, config    
    
def load_featured_dataframe_and_config(subdir, data_fname='featured.csv', drop_inf='cols', **kwargs):
    data_dir = os.path.join(FEATURED_BASE_PATH,subdir)
    df, config = load_dataframe_and_config(
        data_dir, data_fname, **kwargs)
    if drop_inf=='cols':
        df, removed_columns = drop_inf_cols(df)
        config['FEATURED']['removed_features'] = removed_columns
        feature_names = config['FEATURED']['feature_names']
        config['FEATURED']['feature_names'] = [
            f for f in feature_names if not f in removed_columns]
    elif not drop_inf:
        pass
    else:
        raise ValueError(f"Unrecognised value for drop_inf of {drop_inf}")
    return df, config    
    
def load_dataframe_and_config(
        dir_path, data_fname, config_fname='details.cfg', 
        **readargs):
    data_path = os.path.join(dir_path, data_fname)
    logger.info(f"Reading dataframe from {data_path}")
    df = pd.read_csv(data_path, index_col=0,  **readargs)
    config_path = os.path.join(dir_path, config_fname)
    logger.info(f"Reading config from {config_path}")
    raw_config = configparser.ConfigParser()
    with open(config_path, 'r') as config_file:
        raw_config.read_file(config_file)
    # convert config to dictionary
    dict_config =  config_to_dict(raw_config)
    return df, dict_config


# _test_local2unix()
# _test_unix2local()
