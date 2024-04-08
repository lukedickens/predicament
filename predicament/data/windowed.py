# -*- coding: utf-8 -*-
"""
"""

import os
import numpy as np

import logging
logger = logging.getLogger(__name__)

def merge_condition_data(nested_data):
    """
    Function takes nested dictionary of dictionaries which contains windowed data for participants and conditions. 
    
    parameters
    ----------
    nested_data : is a nested dictionary first by participant, with subdictionaries by condition.
    
    returns
    -------
    merged_data :
        A dictionary in which each participant id is mapped to a pair (inputs, labels). Where inputs is a 2d np.array of shape NxW and labels is a 1d np.array of size N. The nth row of the inputs matrix has label id of the nth element of the labels array.
    label_mapping : a list mapping ids to conditions.
    """
    conditions = set()
    for participant, part_data in nested_data.items():
        conditions |= set(part_data.keys())
    label_mapping = list(conditions)
#    print(f"label_mapping = {label_mapping}")
    label_mapping = sorted(label_mapping)
#    print(f"label_mapping = {label_mapping}")
    reverse_label_mapping = {c:i for i, c in enumerate(label_mapping)}
    # merged data
    merged_data = {}
    for participant, part_data in nested_data.items():
        #print(f"participant = {participant}")
        collected_data = []
        collected_labels = []
        for condition, pc_data in part_data.items():
            label = reverse_label_mapping[condition]
            collected_data.append(pc_data)
            n_datapoints = pc_data.shape[0]
            collected_labels.append(label*np.ones(n_datapoints))
        stacked_data = np.vstack(collected_data)
        stacked_labels = np.concatenate(collected_labels)
        merged_data[participant] = (stacked_data, stacked_labels)
#        print(f"stacked_data.shape = {stacked_data.shape}")
#        print(f"stacked_labels.shape = {stacked_labels.shape}")
    return merged_data, label_mapping
    
def merge_labelled_data_by_key(
        data_by_key, merge_keys=None):
    if merge_keys is None:
        merge_keys = list(data_by_key.keys())
    collected_data = []
    collected_labels = []
    for part_id in merge_keys:
        part_data, part_labels = data_by_key[part_id]
        collected_data.append(part_data)
        collected_labels.append(part_labels)
    stacked_data = np.vstack(collected_data)
    stacked_labels = np.concatenate(collected_labels)
    return stacked_data, stacked_labels

def window_all_participants_data(
        all_participants_data, conditions, channels, window_size, window_step,
        condition_fragile=True, channel_fragile=True, copy=True):
    """
    Creates dictionary of dictionaries of windowed data.
    
    parameters
    ----------
    all_participants_data :
    condition_fragile: 
        If True means function will raise error if any conditions are not 
        available either due to missing condition in ParticipantData object or
        insufficient data for a single row. If False then that condition will be
        silently exclude that condition
    channel_fragile: 
        If True means function will raise error if any channels arew not not 
        available in ParticipantData object. If False then all conditions
        will be silently excluded for that participant and the participant will
        be removed from the dictionary.
        
    returns
    -------
    all_windowed_data
        Outer dictionary is indexed by particpant ID, values are inner dictionaries
        Inner dictionary is indexed by condition, values are 2d arrays of
        participant-condition windowed data
        
    """
    all_windowed_data = {}
    for participant, participant_data in all_participants_data.items():
#        print(f"participant {participant} has type(participant_data) = {type(participant_data)}")
        participant_windowed_data = {}
        for condition in conditions:
            try:
                pc_windowed_data = participant_data.get_windowed_data_concat(
                    condition, channels, window_size, step=window_step,
                    copy=copy)
                participant_windowed_data[condition] = pc_windowed_data
            except ValueError:
                if channel_fragile:
                    raise
            except KeyError:
                if condition_fragile:
                    raise
        if len(participant_windowed_data) > 0:
            all_windowed_data[participant] = participant_windowed_data
    return all_windowed_data

def get_window_start_in_event_time(window_index, window_step, sample_rate):
    start_sample_index = window_index*window_step
    in_event_start_time = start_sample_index/sample_rate
    return in_event_start_time

def get_window_duration(window_size, sample_rate):
    return window_size/sample_rate

##
def correct_window_indices(participants, label_mapping, datadf):
    for participant in participants:
        for condition_id, condition_name in enumerate(label_mapping):
            filter_ = datadf['participant'] == participant
            filter_ &= datadf['condition'] == condition_id
            pc_indices = datadf[filter_].index
            num_elements = len(pc_indices)
            datadf.loc[pc_indices, 'window index'] = np.arange(num_elements)


def get_window_start_times_for_participant(
        participant_events, condition_id, label_mapping,
        datadf, window_step, sample_rate):
    """
    inputs
    ------
    participant_events [predicament.data.events.ParticipantEvents] -
        for participant to get starttimes for
    condition_id [int] - condition_id in label_mapping list
    datadf -
        pandas.DataFrame of data each row representing a timeseries window
        or features extracted from a time series window.
    """
    condition_name = label_mapping[condition_id]
    filter_ = datadf['participant']==participant_events.ID
    filter_ &=  datadf['condition']==condition_id
    pc_indices = datadf[filter_].index
    window_start_unixtimes = np.empty(pc_indices.shape[0])
    for w, window_index in enumerate(datadf.loc[pc_indices,'window index']):
        try:
            window_start_in_event_time = \
                get_window_start_in_event_time(
                    window_index, window_step, sample_rate)
        except:
            logger.info(f"window_index = {window_index}")
            logger.info(f"window_step = {window_step}")
            logger.info(f"sample_rate = {sample_rate}")
            raise
        window_start_unixtime = \
            participant_events.in_event_time_to_unixtime(
                condition_name,window_start_in_event_time)
        window_start_unixtimes[w] = window_start_unixtime
    return window_start_unixtimes

def remove_window_start_and_end_times(datadf):
    start_time_col = 'start time'
    end_time_col = 'end time'
    datadf.pop(start_time_col)
    datadf.pop(end_time_col)
    return datadf
    
def insert_window_start_and_end_times(
        all_participants_events, label_mapping, datadf,  window_size, window_step,
        sample_rate):

    window_duration = get_window_duration(window_size, sample_rate)
    start_time_col = 'start time'
    end_time_col = 'end time'
    if not start_time_col in datadf.columns:
        datadf.insert(3,start_time_col, None)
    if not end_time_col in datadf.columns:
        datadf.insert(4,end_time_col, None)
    participants = list(all_participants_events.keys())
    for participant, participant_events in all_participants_events.items():
        assert(participant == participant_events.ID)
        for condition_id, condition_name in enumerate(label_mapping):
            filter_ = datadf['participant'] == participant
            filter_ &= datadf['condition'] == condition_id
#            logger.debug(f"participant = {participant}, condition = {label_mapping[condition_id]}")
            pc_indices = datadf[filter_].index
#            print(f"len(pc_indices) = {len(pc_indices)}")
            try:
                start_times = get_window_start_times_for_participant(
                    participant_events, condition_id, label_mapping,
                    datadf, window_step, sample_rate)
                datadf.loc[pc_indices,start_time_col] = start_times
            except KeyError:
                pass
    datadf[end_time_col] = datadf['start time'] + window_duration
    
##TODO move to test framework and apply asserts
def testing_window_all_participants_data():
    # loading default values
    from predicament.utils.config import DREEM_EEG_CHANNELS
    from predicament.utils.config import EEG_CONDITIONS
    from predicament.utils.config import DEFAULT_WINDOW_SIZE
    from predicament.utils.config import DEFAULT_WINDOW_STEP
    #
    from predicament.data.timeseries import create_participant_data_edf_only
    all_participants_data = create_participant_data_edf_only()
    conditions = EEG_CONDITIONS
    channels = DREEM_EEG_CHANNELS
    window_size = DEFAULT_WINDOW_SIZE
    window_step = DEFAULT_WINDOW_SIZE//8
    all_windowed_data = window_all_participants_data(
            all_participants_data, conditions, channels, window_size, window_step,
            condition_fragile=False, channel_fragile=False, copy=False)
    row_counts = {}
    for participant, window_data_p in all_windowed_data.items():
        for condition, window_data_pc in window_data_p.items():
            print(f"{participant}-{condition}: {window_data_pc.shape[0]}x{window_data_pc.shape[1]}")
            row_count = row_counts.get(condition,0)
            row_counts[condition] = row_count + window_data_pc.shape[0]
    print(f"row counts per condition = {row_counts}")
    print(f"Total number of data rows: {np.sum(list(row_counts.values()))}")
    print(f"channels = {channels}")


if __name__ == '__main__':
    testing_window_all_participants_data()

