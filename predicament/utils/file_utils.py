# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 18:51:32 2022

@author: Zerui Mu

Utility functions
"""

import random
import time
import os
import pandas as pd
import numpy as np
import configparser

from predicament.utils.config import WINDOWED_BASE_PATH
from predicament.utils.config import FEATURED_BASE_PATH

from predicament.utils.config import CROSS_VALIDATION_BASE_PATH
from predicament.utils.config_parser import config_to_dict
from predicament.utils.config_parser import dict_to_config

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

def write_dataframe_and_config(
        dir_path, data, config, data_fname, config_fname='details.cfg'):
    if type(config) is dict:
        config = dict_to_config(config)
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

def load_windowed_data_and_config(subdir, drop_inf=None, **kwargs):
    data_dir = os.path.join(WINDOWED_BASE_PATH,subdir)
    df, config = load_dataframe_and_config(
        data_dir, 'windowed.csv', drop_inf=drop_inf, **kwargs)
    return df, config    
    
def load_dataframe_and_config(
        dir_path, data_fname, config_fname='details.cfg', 
        drop_inf='cols', **readargs):
    data_path = os.path.join(dir_path, data_fname)
    print(f"Reading dataframe from {data_path}")
    data = pd.read_csv(data_path, index_col=0,  **readargs)
    config_path = os.path.join(dir_path, config_fname)
    print(f"Reading config from {config_path}")
    config = configparser.ConfigParser()
    with open(config_path, 'r') as config_file:
        config.read_file(config_file)
    # convert config to dictionary
    dict_config =  config_to_dict(config)
    if drop_inf=='cols':
        data, removed_columns = drop_inf_cols(data)
        dict_config['FEATURED']['removed_features'] = removed_columns
        feature_names = dict_config['FEATURED']['feature_names']
        dict_config['FEATURED']['feature_names'] = [
            f for f in feature_names if not f in removed_columns]
    elif not drop_inf:
        pass
    else:
        raise ValueError(f"Unrecognised value for drop_inf of {drop_inf}")
    return data, dict_config

def drop_inf_cols(df):
    removed_columns = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if np.any(np.isinf(df[col])):
            removed_columns.append(col)
            del df[col]
    return df, removed_columns

def drop_nan_cols(df):
    removed_columns = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if np.any(np.isnan(df[col])):
            removed_columns.append(col)
            del df[col]
    return df, removed_columns

# _test_local2unix()
# _test_unix2local()
