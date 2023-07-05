# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 21:58:32 2022

@author: Zerui Mu
"""

import os
from typing import Dict
import mne

from predicament.utils.config import STUDY_DATA_FOLDER, VG_file_paths, VG_Hz, EEG_buffer
from predicament.utils.config import DREEM_EEG_CHANNELS
from Ray.Event_details import Event_time_details

# these are the short names. I have tried to move to just using fully qualified names
#EEG_channels = {1:"EEG Fpz-O1", 2:"EEG Fpz-O2", 3:"EEG Fpz-F7", 4:"EEG F8-F7", 5:"EEG F7-01", 6:"EEG F8-O2", 7:"EEG Fpz-F8"}

class EEG_data_obj(object):
    def __init__(self, part_ID, data, data_channel_names, include_channels=DREEM_EEG_CHANNELS) -> None:
        channel_lookup = {ch_name:i for i, ch_name in enumerate(data_channel_names)}
        print(f"include_channels={include_channels}")
        print(f"type(data) = {type(data)}")
        print(f"data.shape = {data.shape}")
        # participant ID
        self.ID = part_ID
        # each channel uses the short name and is loaded from the full data object
#        self.EEG_data = {
#            ch_name: data[channel_lookup[ch_name]] \
#                for ch_name in include_channels}
        self.EEG_data = {}
        for ch_name in include_channels:
            print(f"ch_name = {ch_name}")
            print(f"channel_lookup[ch_name] = {channel_lookup[ch_name]}")
            ch_idx = channel_lookup[ch_name]
            self.EEG_data[ch_name] = data[ch_idx,:]
        self.event_details = None # need to be set after init

    def get_channels(self):
        return list(self.EEG_data.keys())
    channels = property(fget=get_channels)

    def set_event_details(self, event_details: Event_time_details):
        self.event_details = event_details # including date, start_time, events_info

    def _get_event_period_by_name(self, event):
        # return the period of time with 2 buffers (initially 30 sec)
        exp_start_time = self.event_details.exp_start_time
        event_start = self.event_details.events_info[event]["start"]
        event_end = self.event_details.events_info[event]["end"]
        return event_start - exp_start_time + EEG_buffer, event_end - exp_start_time - EEG_buffer

    def get_EEG_by_channel_and_event(self, channel, event_name):
        # channel can be the name or its index
        # print(channel, event_name)
        start, end = self._get_event_period_by_name(event_name)
        print(f"self.channels = {self.channels}")
        if channel in self.channels:
            return self.EEG_data[channel][int(start * VG_Hz): int(end * VG_Hz)]
            # self.channels is now a list not a dictionary.
#        elif channel in self.channels.values():
#            return self.EEG_data[channel][int(start * VG_Hz): int(end * VG_Hz)]
        else:
            raise ValueError("Unrecognised input channel: {}".format(channel))

    def get_EEG_by_event(self, event_name, channel_list=None):
        if channel_list is None:
            channel_list = self.channels
        # get all 7 EEG channels data
        event_data = []
        for channel in channel_list:
            channel_data = self.get_EEG_by_channel_and_event(channel, event_name)
            event_data.append(channel_data)
        return event_data

    def get_event_duration(self, event_name):
        if event_name not in self.event_details.events_info.keys():
            return None
        start = self.event_details.events_info[event_name]['start']
        end = self.event_details.events_info[event_name]['end']
        if start == None or end == None:
            return None
        return end - start
        
    def __repr__(self):
        output = f'EEG_data_obj(ID:{self.ID}, channels:{self.channels})'
        return output

def read_all_VG_files() -> Dict[str, EEG_data_obj]:
    return {part_ID: read_VG_file(part_ID) for part_ID in VG_file_paths.keys()}

def read_VG_file(part_ID, exclude_channels=None, include_channels=None) -> EEG_data_obj:
    """
    global variables
    ----------------
    VG_file_paths is from config file
    
    parameters
    ----------
    part_ID - participant ID
    exclude_channels - channels to exclude from loading in from mne object.
    """
    try:
        if part_ID not in VG_file_paths.keys():
            raise IndexError("The given part_ID ({}) is not included".format(part_ID))
    except RuntimeError as e:
        print("Error:", e)
    print(f"STUDY_DATA_FOLDER = {STUDY_DATA_FOLDER}")
    VG_file_path = os.path.join(STUDY_DATA_FOLDER, VG_file_paths[part_ID])
    print(f"VG_file_path = {VG_file_path}")
    if not exclude_channels is None:
        VG_file = mne.io.read_raw_edf(VG_file_path, exclude=exclude_channels)
    VG_file = mne.io.read_raw_edf(VG_file_path)
    data_channel_names = VG_file.ch_names
    data_channels = VG_file.get_data()
    print(f"VG_file.ch_names = {VG_file.ch_names}")
    if not include_channels is None:
        data = EEG_data_obj(
            part_ID, data_channels, data_channel_names, include_channels=include_channels)
    data = EEG_data_obj(part_ID, data_channels, data_channel_names)
    # print(VG_file.ch_names)
    print("Successfully loaded {} VG file (EEG data).".format(part_ID))
    return data

def read_all_VG_to_Raw():
    return {part_ID: read_VG_to_Raw(part_ID) for part_ID in VG_file_paths.keys()}

def read_VG_to_Raw(part_ID, exclude_channels=None):
    if exclude_channels is None:
        exclude_channels = ['Accelero Norm', 'Positiongram', 'PulseOxy Infrare', 'PulseOxy Red Hea', 'Respiration x', 'Respiration y', 'Respiration z']
    VG_file_path = os.path.join(STUDY_DATA_FOLDER, VG_file_paths[part_ID])
    raw = mne.io.read_raw_edf(VG_file_path, exclude=exclude_channels, preload = True)
    return raw
