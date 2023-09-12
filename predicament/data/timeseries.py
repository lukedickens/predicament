# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 21:58:32 2022

@author: Zerui Mu
"""

import os
from typing import Dict
import mne
import numpy as np

from predicament.utils.config import STUDY_DATA_FOLDER
from predicament.utils.config import EDF_FILE_PATHS
from predicament.utils.config import DEFAULT_SAMPLE_RATE
#EEG_buffer
from predicament.utils.config import DREEM_EEG_CHANNELS
from predicament.utils.config import FULL_PARTICIPANT_LIST
from predicament.utils.config import TARGET_CONDITIONS
from predicament.utils.config import EEG_CONDITIONS

from predicament.data.events import load_event_info_from_csv

class ParticipantData(object):
    """
    Stores the time series data associated with a single participant.
    This can be loaded from EDF files or csv (not yet implemented).
    Data must correspond to contemporaneously recorded samples whose local
    time index and sample rate aligns with the event information.
    """
    def __init__(self, part_ID, event_details, sample_rate):
        self.ID = part_ID
        self.timeseries = {}
        self.event_details = event_details
        self.sample_rate = sample_rate

    def add_timeseries_data(
            self, data, data_channel_names, sample_rate,
            include_channels=None):
        # data is loaded as 2d np.array rows of time series with one row per
        # channel if include_channels is set then we want to load only those
        # rows
        if sample_rate != self.sample_rate:
            raise ValueError("Cannot add data with different sample rate")
        if include_channels is None:
            include_channels = data_channel_names
        channel_lookup = {
            ch_name:i for i, ch_name in enumerate(data_channel_names)}
        # participant ID
        for ch_name in include_channels:
            ch_idx = channel_lookup[ch_name]
            self.timeseries[ch_name] = data[ch_idx,:]

    def get_channels(self):
        return list(self.timeseries.keys())
    channels = property(fget=get_channels)

#    def _get_event_period_by_name(self, condition):
#        # return the period of time with 2 buffers (initially 30 sec)
#        exp_start_time = self.event_details.exp_start_time
#        event_start = self.event_details.events_info[condition]["start"]
#        event_end = self.event_details.events_info[condition]["end"]
#        return event_start - exp_start_time + EEG_buffer, event_end - exp_start_time - EEG_buffer

    def get_event_start_end_indices(self, condition):
        # start and end indices
        start, end = self.event_details.get_event_studytimes(condition)
        start_index = int(start * self.sample_rate)
        end_index = int(end * self.sample_rate)
        return start_index, end_index

    def get_event_timeseries_by_channel(self, condition, channel):
        start_index, end_index = self.get_event_start_end_indices(condition)
        return self.timeseries[channel][start_index: end_index]

#    def get_EEG_by_event(self, event_name, channel_list=None):
#        if channel_list is None:
#            channel_list = self.channels
#        event_data = []
#        for channel in channel_list:
#            channel_data = self.get_EEG_by_channel_and_event(channel, event_name)
#            event_data.append(channel_data)
#        return event_data

#    def get_event_duration(self, event_name):
#        if event_name not in self.event_details.events_info.keys():
#            return None
#        start = self.event_details.events_info[event_name]['start']
#        end = self.event_details.events_info[event_name]['end']
#        if start == None or end == None:
#            return None
#        return end - start
        
    def __repr__(self):
        output = f'ParticipantData(ID:{self.ID}, channels:{self.channels})'
        return output


    def get_windowed_data_concat(
            self, condition, channels, window_size, step=None, copy=True):
        """
        Gets all data for given participant-condition windowed by fixed size
        with given step length. If step-length is None, then assume
        non-overlapping windows.
        Concat refers to the fact that each row is a concatenation of the
        corresponding window data for all channels.
        """
        if step is None:
            step = window_size
        channel_matrices = []
        for channel in channels:
            ch_timeseries = self.get_event_timeseries_by_channel(
                condition, channel)
            ch_mtx = window_array(ch_timeseries, window_size, step, copy=copy)
            channel_matrices.append(ch_mtx)
        return np.hstack(channel_matrices)

# helper function may belong elsewhere
def window_array(array, window_size, step, copy=False):
    # see https://stackoverflow.com/questions/45730504/how-do-i-create-a-sliding-window-with-a-50-overlap-with-a-numpy-array
    # for an alternative way
    # the width of the window
    w = window_size
    # the integer difference between the index of the ith 
    # and (i+1)th window start points 
    st = step
    view = np.lib.stride_tricks.sliding_window_view(array, w)[::st]
    if copy:
        return view.copy()
    else:
        return view
            
## builder functions
def init_participants_data(
        all_participants_events,
        sample_rate=DEFAULT_SAMPLE_RATE):
    all_participants_data = {}
    for part_ID in all_participants_events.keys():
        event_details = all_participants_events[part_ID]
        participant_data = ParticipantData(part_ID, event_details, sample_rate)  
        all_participants_data[part_ID] = participant_data       
    return all_participants_data

def read_all_participants_edf_files(
        all_participants_data, edf_file_paths, 
        sample_rate=DEFAULT_SAMPLE_RATE):
    for part_ID in all_participants_data.keys():
        print(f"adding {part_ID}")
        edf_file_path = edf_file_paths[part_ID]
        print(f"\tfrom {edf_file_path}")
        participant_data = all_participants_data[part_ID]
        all_participants_data[part_ID] = read_edf_file_to_participant_data(
            participant_data, edf_file_path, sample_rate)
    return all_participants_data

def read_edf_file_to_participant_data(
        participant_data, edf_file_path, sample_rate,
        **kwargs):
    """
    
    parameters
    ----------
    part_ID - participant ID
    exclude_channels - channels to exclude from loading in from mne object.
    """
    print(f"edf_file_path = {edf_file_path}")
    mne_data = mne.io.read_raw_edf(edf_file_path)
    data_channel_names = mne_data.ch_names
    np_data = mne_data.get_data()
    print(f"mne_data.ch_names = {mne_data.ch_names}")
    participant_data.add_timeseries_data(
            np_data, data_channel_names, sample_rate, **kwargs)
    print("Successfully loaded EDF file for {} .".format(participant_data.ID))
    return participant_data

def create_participant_data_edf_only(
        participant_list=FULL_PARTICIPANT_LIST,
        edf_file_paths=EDF_FILE_PATHS):
    ## multi step participant data creation
    # get all participants event information
    all_participants_events = load_event_info_from_csv(participant_list)
    # create ParticipantData objects with event info but no data
    all_participants_data = init_participants_data(all_participants_events)
    # load data into the ParticipantData objects
    all_participants_data = read_all_participants_edf_files(
        all_participants_data, edf_file_paths)
    return all_participants_data

##TODO move to testing framework and apply asserts
# testing output
def test_windowed_data():
    from predicament.utils.config import DEFAULT_WINDOW_SIZE
    from predicament.utils.config import DEFAULT_WINDOW_STEP
    test_ID = 'VG_01'
    test_condition = 'wildlife_video'
    #test_channels = ['EEG Fpz-O1']
    #test_channels = ['EEG Fpz-O1', 'EEG Fpz-O2']
    test_channels = ['EEG Fpz-O1', 'EEG Fpz-O2', 'EEG F8-O2']
    window_size = DEFAULT_WINDOW_SIZE
    window_step = DEFAULT_WINDOW_STEP
    # get just one participants data
    all_participants_events = load_event_info_from_csv(
        participant_list=[test_ID])
    all_participants_data = init_participants_data(all_participants_events)
    edf_file_paths = EDF_FILE_PATHS
    all_participants_data = read_all_participants_edf_files(
        all_participants_data, edf_file_paths)
    participant_data = all_participants_data[test_ID]
    windowed_data = participant_data.get_windowed_data_concat(
        test_condition, test_channels, window_size, step=window_step, copy=True)
    print(f"windowed_data = {windowed_data}")
    print(f"windowed_data.shape = {windowed_data.shape}")
    
    
if __name__ == '__main__':
    # only load participants who have data for all target conditions
#    all_participants_events = load_event_info_from_csv(
#        participant_list=TARGET_CONDITIONS)
#    for part_ID, participant_events in all_participants_events.items():
#        print(f"{part_ID}: {participant_events}")
#    all_participants_data = init_participants_data(all_participants_events)
#    print(f"all_participants_data = {all_participants_data}")
#    edf_file_paths = EDF_FILE_PATHS
#    print(f"edf_file_paths = {edf_file_paths}")
#    all_participants_data = read_all_participants_edf_files(
#        all_participants_data, edf_file_paths)
#    print(f"all_participants_data = {all_participants_data}")
#    test_windowed_data()
    testing_window_all_participants_data()
