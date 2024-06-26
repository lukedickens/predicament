# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 17:11:22 2022

@author: Zerui Mu
"""

import os
import pandas as pd
import random

from predicament.utils.config import EEG_buffer, VG_Hz
from predicament.utils.config import EVENT_DETAILS_PATH
from predicament.utils.config import EXP_INFO_PATH
from predicament.utils.file_utils import unix2local
from Ray.Event_details import Event_time_details
from Ray import EEG_data, E4_data


##TODO weird cyclic dependency where file is loaded and saved to same location
def save_info_to_csv(event_details_path=EVENT_DETAILS_PATH, exp_info_path=EXP_INFO_PATH):
    EEG_files = EEG_data.read_all_VG_files()
    event_details_dic = {
        'partID': [],
        'event': [],
        'start': [],
        'end': [],
        'valid': []
    }

    exp_info_dic = {
        'partID': [],
        'date': [],
        'exp_start': []
    }

    participant_ids = list(EEG_files.keys())
    for part in participant_ids:
        EEG_part = EEG_files[part]
        for Ev_name, EV_info in EEG_part.event_details.events_info.items():
            start_time = unix2local(EV_info['start']).split(' ')[1] if EV_info['start'] != None else 'None' # only need time from datetime
            end_time = unix2local(EV_info['end']).split(' ')[1] if EV_info['end'] != None else 'None' # only need time from datetime
            event_details_dic['partID'].append(part)
            event_details_dic['event'].append(Ev_name)
            event_details_dic['start'].append(start_time) 
            event_details_dic['end'].append(end_time)
            event_details_dic['valid'].append(EV_info['valid'])
        
        exp_info_dic['partID'].append(part)
        exp_info_dic['date'].append(EEG_part.event_details.exp_date)
        exp_info_dic['exp_start'].append(unix2local(EEG_part.event_details.exp_start_time).split(' ')[1])

    event_details_df = pd.DataFrame(event_details_dic)
    exp_info_df = pd.DataFrame(exp_info_dic)

    with open(event_details_path, 'w', newline='') as f:
        event_details_df.to_csv(f, index = False)
    with open(exp_info_path, 'w', newline='') as f:
        exp_info_df.to_csv(f, index = False)

def load_info_from_csv(event_details_path=EVENT_DETAILS_PATH, exp_info_path=EXP_INFO_PATH):
    """
    Event details
    """
    print(f"event_details_path = {event_details_path}")
    event_details_df = pd.read_csv(event_details_path, index_col=0)
    print(f"event_details_df.loc['VG_01'] = {event_details_df.loc['VG_01']}")
    exp_info_df = pd.read_csv(exp_info_path, index_col=0)

    part_list = list(exp_info_df.index)
    ev_details = {}

    for partID in part_list:

        ev_detail = Event_time_details(partID)
        ev_detail.set_exp_datetime(exp_info_df.loc[partID]['date'], exp_info_df.loc[partID]['exp_start'])    
        for event in event_details_df.loc[partID,].iterrows():
            print(f"event = {event}")
            print(f"event[1]['end'] = {event[1]['end']}")
            print(f"type(event[1]['end']) = {type(event[1]['end'])}")
#            start = str(event[1]['start']) if event[1]['start'] != 'None' else None
#            end = str(event[1]['end']) if event[1]['end'] != 'None' and not math.isnan(event[1]['end']) else None
            try:
                start = str(event[1]['start'])
            except ValueError:
                start = None
            try:
                end = str(event[1]['end'])
            except ValueError:
                end = None
            print(f"partID: end = {partID}:{end}")
            ev_detail.set_event(event[1]['event'], start, end, event[1]['valid'])
        ev_details[partID] = ev_detail
    return ev_details

def set_up(isEEG = True, isE4 = True):
    # isEEG: load EEG data or not
    # isE4: load E4 data or not
    EEG_files = EEG_data.read_all_VG_files() if isEEG == True else None
    E4_files = E4_data.read_all_E4_files() if isE4 == True else None
    ev_details = load_info_from_csv()

    if isEEG == True:
        for partID, EEG in EEG_files.items():
            EEG.set_event_details(ev_details[partID])
    if isE4 == True:
        for partID, E4 in E4_files.items():
            E4.set_event_details(ev_details[partID])

    return EEG_files, E4_files

def gen_EEG_traintest_to_csv_mix(event_list, multiply, windows = 1, train_ratio = 0.8, EEG_files = None):
    # mix all partcipants data together and separate train_set and test_set by train_ratio
    if EEG_files == None:
        EEG_files, _ = set_up(isE4 = False)
    #TODO what is fs(EEG sample rate)? And how can we multiply by it before it is assigned?
    # how is row size determined? this ends up being the width of the data matrix produced
    row_size, fs, interval = 4096, VG_Hz, fs*2 
    # channel_batch_size = {
    #     'EEG Fpz-O1': row_size // 7,
    #     'EEG Fpz-O2': row_size // 7,
    #     'EEG Fpz-F7': row_size // 7,
    #     'EEG F8-F7': row_size // 7,
    #     'EEG F7-01': row_size // 7,
    #     'EEG F8-O2': row_size // 7,
    #     'EEG Fpz-F8': row_size - (row_size // 7) * 6,
    # }
    channel_batch_size = {
        'EEG Fpz-O1': row_size // 4,
        'EEG Fpz-O2': row_size // 4,
        'EEG F8-F7': row_size // 4,
        'EEG Fpz-F8': row_size - (row_size // 4) * 3,
    }
    data_label_list = {'train': [], 'test':[]}
    label_dic = {label: i for i, label in enumerate(event_list)}
    # too many samples for exper_video and a little bit less for family_inter
    internal_ratio_for_each_event = {
        "familiar_music": 1.0,
        "wildlife_video": 1.0,
        "family_inter": 0.75,
        "tchaikovsky": 1.0,
        "exper_video": 1.5
    }

    participant_ids = list(EEG_files.keys())
    for part in participant_ids:
        for event_name in event_list:
            if EEG_files[part].event_details.check_has_event(event_name) and \
                    EEG_files[part].event_details.check_event_has_start_and_end(event_name):
                #TODO decypher
                # universal start end times for participant-condition event
                event_end = EEG_files[part].event_details.events_info[event_name]['end']
                event_start = EEG_files[part].event_details.events_info[event_name]['start']
                # duration of participant-condition event in samples
                # equal to sample rate(fs) times the buffered-duration in seconds
                data_length = (event_end - event_start - 2 * EEG_buffer) * fs
                test_length = round(data_length/fs * (1-train_ratio)) * fs
                event_data_all_chans = {chan: EEG_files[part].get_EEG_by_channel_and_event(channel=chan, event_name=event_name) \
                    for chan in channel_batch_size.keys()}

                # test data
                for start in range(0, int(test_length - row_size // len(channel_batch_size.keys())*0.5), \
                                    int(internal_ratio_for_each_event[event_name]*interval)):
                    row_data = []
                    for channel, size in channel_batch_size.items():
                        row_data.extend(event_data_all_chans[channel][start: start+size])
                    data_label_list['test'].append((row_data, label_dic[event_name]))

                # train data
                for start in range(int(test_length + row_size//len(channel_batch_size.keys())*0.5), \
                                    int(data_length - row_size//len(channel_batch_size.keys()))-1, \
                                    int(internal_ratio_for_each_event[event_name]*interval)):
                    row_data = []
                    for channel, size in channel_batch_size.items():
                        row_data.extend(event_data_all_chans[channel][start: start+size])
                    data_label_list['train'].append((row_data, label_dic[event_name]))

    random.shuffle(data_label_list['train'])
    random.shuffle(data_label_list['test'])
    train_data_df = pd.DataFrame([dat[0] for dat in data_label_list['train']]).multiply(multiply)
    test_data_df = pd.DataFrame([dat[0] for dat in data_label_list['test']]).multiply(multiply)
    train_label_df = pd.DataFrame([dat[1] for dat in data_label_list['train']])
    test_label_df = pd.DataFrame([dat[1] for dat in data_label_list['test']])
    print(train_data_df.shape)
    print(test_data_df.shape)
    print(train_label_df.shape)
    print(test_label_df.shape)
    return train_data_df, test_data_df, train_label_df, test_label_df

def gen_EEG_traintest_to_csv_part(event_list, multiply, windows = 1, train_ratio = 0.8, EEG_files = None):
    # separate train_set and test_set by participant and the train_ratio
    if EEG_files == None:
        EEG_files, _ = set_up(isE4 = False)
    # row size is total width of data matrix. So time window size varies depending on number of channels
    # at 4096 for 4 channels, thats 1024 smples per channel, 1024/250 \approx 4 seconds
    # fs is the sample frequency
    # interval is the step size of the windowing in sample steps (2*VG_Hz means 2 seconds worth)
    row_size, fs, interval = 4096, VG_Hz, VG_Hz*2
    train_num_per_class_limit, test_num_per_class_limit = 1000, 100
    # channel_batch_size = {
    #     'EEG Fpz-O1': row_size // 7,
    #     'EEG Fpz-O2': row_size // 7,
    #     'EEG Fpz-F7': row_size // 7,
    #     'EEG F8-F7': row_size // 7,
    #     'EEG F7-01': row_size // 7,
    #     'EEG F8-O2': row_size // 7,
    #     'EEG Fpz-F8': row_size - (row_size // 7) * 6,
    # }
    channel_batch_size = {
        'EEG Fpz-O1': row_size // 4,
        'EEG Fpz-O2': row_size // 4,
        'EEG F8-F7': row_size // 4,
        'EEG Fpz-F8': row_size - (row_size // 4) * 3,
    }
    label_dic = {label: i for i, label in enumerate(event_list)}
    # train and test participants list
    test_part_num = round(len(EEG_files.keys()) * (1-train_ratio))
    test_part_num = test_part_num if test_part_num != 0 else 1
    test_part_list = random.sample(list(EEG_files.keys()), test_part_num)
    train_part_list = [part for part in EEG_files.keys() if part not in test_part_list]

    # train data and labels
    train_data_label_dic = {}
    for part in train_part_list:
        for event_name in event_list:
            if label_dic[event_name] not in train_data_label_dic.keys():
                train_data_label_dic[label_dic[event_name]] = []
            if EEG_files[part].event_details.check_has_event(event_name) and EEG_files[part].event_details.check_event_has_start_and_end(event_name):
                data_length = (EEG_files[part].event_details.events_info[event_name]['end'] - \
                    EEG_files[part].event_details.events_info[event_name]['start'] - 2 * EEG_buffer) * fs
                for start in range(0, int(data_length - row_size // len(channel_batch_size.keys())) - 1, interval):
                    if len(train_data_label_dic[label_dic[event_name]]) >= train_num_per_class_limit:
                        break
                    row_data = []
                    for channel, size in channel_batch_size.items():
                        part_data = EEG_files[part].get_EEG_by_channel_and_event(channel=channel, event_name=event_name)
                        row_data.extend(part_data[start: start + size])
                    train_data_label_dic[label_dic[event_name]].append((row_data, label_dic[event_name]))

    train_data_label = []
    for l in train_data_label_dic.values():
        train_data_label.extend(l)
                    
    # test data and labels
    test_data_label_dic = {}
    for part in test_part_list:
        for event_name in event_list:
            if label_dic[event_name] not in test_data_label_dic.keys():
                test_data_label_dic[label_dic[event_name]] = []
            # events to use are present and have time-stamps
            if EEG_files[part].event_details.check_has_event(event_name) and EEG_files[part].event_details.check_event_has_start_and_end(event_name):
                data_length = (EEG_files[part].event_details.events_info[event_name]['end'] - \
                    EEG_files[part].event_details.events_info[event_name]['start'] - 2 * EEG_buffer) * fs
                for start in range(0, int(data_length - row_size // len(channel_batch_size.keys())) - 1, interval):
                    if len(test_data_label_dic[label_dic[event_name]]) >= test_num_per_class_limit:
                        break
                    row_data = []
                    for channel, size in channel_batch_size.items():
                        part_data = EEG_files[part].get_EEG_by_channel_and_event(channel=channel, event_name=event_name)
                        row_data.extend(part_data[start: start + size])
                    test_data_label_dic[label_dic[event_name]].append((row_data, label_dic[event_name]))

    test_data_label = []
    for l in test_data_label_dic.values():
        test_data_label.extend(l)

    random.shuffle(train_data_label)
    random.shuffle(test_data_label)
    train_data_df = pd.DataFrame([dat[0] for dat in train_data_label]).multiply(multiply)
    test_data_df = pd.DataFrame([dat[0] for dat in test_data_label]).multiply(multiply)
    train_label_df = pd.DataFrame([dat[1] for dat in train_data_label])
    test_label_df = pd.DataFrame([dat[1] for dat in test_data_label])
    print(train_data_df.shape)
    print(test_data_df.shape)
    print(train_label_df.shape)
    print(test_label_df.shape)
    return train_data_df, test_data_df, train_label_df, test_label_df
