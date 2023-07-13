# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 18:09:32 2022

@author: Zerui Mu
@author: Luke Dickens
"""

import os
import pandas as pd
from matplotlib import pyplot as plt

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
        conditions=TARGET_CONDITIONS, channels=DREEM_MINIMAL_CHANNELS,
        window_size=DEFAULT_WINDOW_SIZE, window_step=None):
    if window_step is None:
        window_step = DEFAULT_WINDOW_SIZE//8
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
    return data_by_participant, label_mapping
    
    
def iterate_between_subject_cv_partition(**loadargs):
    data_by_participant, label_mapping = load_labelled_data(**loadargs)
    for fold in between_subject_cv_partition(data_by_participant):
        tr_dt, tr_lb, ts_dt, ts_lb = fold
        yield (tr_dt, tr_lb, ts_dt, ts_lb,label_mapping)

if __name__ == '__main__':

    for fold in iterate_between_subject_cv_partition(
            participant_list=FULL_PARTICIPANT_LIST[:3]):
        tr_dt, tr_lb, ts_dt, ts_lb, lb_map = fold
        print(f"tr_dt.shape = {tr_dt.shape}")
        print(f"tr_lb.shape = {tr_lb.shape}")
        print(f"ts_dt.shape = {ts_dt.shape}")
        print(f"ts_lb.shape = {ts_lb.shape}")
        print(f"lb_map = {lb_map}")

