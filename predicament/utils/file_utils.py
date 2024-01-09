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
import configparser

from predicament.utils.config import CROSS_VALIDATION_BASE_PATH


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


# _test_local2unix()
# _test_unix2local()
