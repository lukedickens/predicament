# -*- coding: utf-8 -*-
"""
"""

import json
import pandas as pd
from predicament.utils.file_utils import local2unix
from predicament.utils.config import EEG_buffer
from predicament.utils.config import EEG_CONDITIONS

from predicament.utils.config import EVENT_DETAILS_PATH
from predicament.utils.config import EXP_INFO_PATH
from predicament.utils.config import BUFFER_INTERVAL

import logging
logger = logging.getLogger(__name__)

class ParticipantEvents(object):
    """
    Stores the event data for a given participant. This includes the
    conditions for the event, the condition start time and the end time as well
    as the overall participants start time. This can be used to appropriately reference 
    timeseries data which is associated with these events.
    """
    def __init__(self, participant, buffer_interval=BUFFER_INTERVAL) -> None:
        # Note: self.exp_start_time is for EEG file, since different E4 files have different start_timestamp
        self.ID = participant
        self.events_info = {} # with the actual event name and its start and end time (without buffer)
        self.exp_date = None # need to be set after init
        self.exp_start_time = None # need to be set after init
        self.buffer_interval = buffer_interval
        
    def set_exp_datetime(self, date, time_hms):
        """
        Get the global start datetime of the experimental data
        """
        self.exp_date = date
        self.exp_start_time = local2unix(date + " " + time_hms)

    def set_event(
            self, condition, start, end, validation = True, verbose=False):
        # validation: if False, means the time is estimated, without verification
        if condition not in EEG_CONDITIONS:
            if verbose:
                print("The event ({}) is not in the event list. Skipping!".format(condition))
            return
        try:
            start_timestamp = local2unix(self.exp_date + " " + str(start))
        except:
            state_timestamp = None
        try:
            end_timestamp = local2unix(self.exp_date + " " + str(end))
        except:
            end_timestamp = None
        logger.debug(f"(start_timestamp, end_timestamp) = {(start_timestamp, end_timestamp)}")
        if validation and (start_timestamp is None or end_timestamp is None):
            logger.warn(f"{self.ID}:{condition} is valid but timestamps have issues.") 
        self.events_info[condition] = {
            "start": start_timestamp,
            "end": end_timestamp,
            'valid': validation}

    def get_conditions(self):
        return set(self.events_info.keys())
        
    def get_valid_conditions(self):
        return {k for k in self.get_conditions() if self.events_info[k]['valid']}

    def get_conditions_with_both_timestamps(self):
        return {k for k in self.get_conditions() \
            if self.check_event_has_start_and_end(condition)}

    def get_good_conditions(self):
#        valid_conditions = self.get_valid_conditions()
#        conditions_with_timestamps= self.get_conditions_with_both_timestamps()
#        return valid_conditions.intersection(conditions_with_timestamps)
        #TODO at time of writing it is unclear what the valid field in the 
        # csv file is. The dissertation has durations for participant-conditions
        # with valid=False 
        return self.get_conditions_with_both_timestamps()

    def check_has_event(self, condition) -> bool:
        return condition in self.events_info.keys()

    def check_event_has_start_and_end(self, condition) -> bool:
        return (self.events_info[condition]['start'] != None and self.events_info[condition]['end'] != None)

    def get_event_start_and_end(self, condition):
        """
        Global start and end time of condition (unix time)
        """
        uni_start = self.events_info[condition]["start"]
        uni_end = self.events_info[condition]["end"]
        return uni_start, uni_end

    def unixtime_to_studytime(self, unixtime):
        return unixtime - self.exp_start_time

    def studytime_to_unixtime(self, studytime):
        return studytime + self.exp_start_time

    def get_event_studytimes_raw(self, condition):
        """
        Relative raw start and end time of condition as compared to the study
        participant's start time (does not include buffer time)
        """
        uni_start, uni_end = self.get_event_start_and_end(condition)
        #print(f"{self.ID}:{condition}: {uni_start}--{uni_end}")
        try:
            return self.unixtime_to_studytime(uni_start), self.unixtime_to_studytime(uni_end)
        except TypeError:
            raise ValueError(f'Missing condition {condition} for participant {self.ID}')

    def get_event_studytimes(self, condition):
        raw_start, raw_end = self.get_event_studytimes_raw(condition)
        return raw_start + self.buffer_interval, raw_end - self.buffer_interval

    def get_event_duration(self,condition):        
        start, end = self.get_event_studytimes(condition)
        return end-start

    def in_event_time_to_studytime(self,condition,in_event_time):
        """
        in_event_time indicates how long (in seconds) after the event is considered
        to have started (this includes the buffer interval).
        """
        if in_event_time > self.get_event_duration(condition):
            print(f"participant = {self.ID}")
            print(f"condition = {condition}")
            print(f"in_event_time = {in_event_time}")
            print(f"self.get_event_duration(condition) = {self.get_event_duration(condition)}")
            raise ValueError('Invalid event time, longer than condition duration')
        condition_start_studytime, _ = self.get_event_studytimes(condition)
        return in_event_time + condition_start_studytime

    def in_event_time_to_unixtime(self,condition,in_event_time):
        """
        in_event_time indicates how long (in seconds) after the event is considered
        to have started (this includes the buffer interval).
        """
        in_studytime = self.in_event_time_to_studytime(condition,in_event_time)
        return self.studytime_to_unixtime(in_studytime)

    def save_event_window_to_json(self, condition, json_path, windows = 1):
        exp_start = self.exp_start_time
        start = int(self.events_info[condition]['start'] - exp_start)
        end = int(self.events_info[condition]['end'] - exp_start)
        spindles = [{'start': this_start*1.0, 'end': this_start + 1.0} \
            for this_start in range(start+buffer_interval, end-buffer_interval, windows)]
        with open(json_path, 'w') as f:
            json.dump(spindles ,f)
            
    def __str__(self):
        out = f"ParticipantEvents({self.events_info})"
        return out

def load_event_info_from_csv(
        participant_list,
        exp_info_path=EXP_INFO_PATH, event_details_path=EVENT_DETAILS_PATH,
        verbose=False):
    """
    Loads Event details
    Previously part of data_load_save.load_info_from_csv 
    """
    if verbose:
        print(f"event_details_path = {event_details_path}")
    event_details_df = pd.read_csv(event_details_path, index_col=0)
    if verbose:
        print(f"event_details_df.loc['VG_01'] = {event_details_df.loc['VG_01']}")
    exp_info_df = pd.read_csv(exp_info_path, index_col=0)
    if participant_list is None:
        participant_list = list(exp_info_df.index)
    if verbose:
        print(f"participant_list = {participant_list}")
    all_participants_events = {}

    for partID in participant_list:
        participant_events = build_participant_events_object(
            partID, event_details_df, exp_info_df)
        all_participants_events[partID] = participant_events
    return all_participants_events

def build_participant_events_object(partID, event_details_df, exp_info_df):
        participant_events = ParticipantEvents(partID)
        participant_events.set_exp_datetime(
            exp_info_df.loc[partID]['date'],
            exp_info_df.loc[partID]['exp_start'])    
        for event in event_details_df.loc[partID,].iterrows():
            start = event[1]['start']
            end = event[1]['end']
            condition = event[1]['event'] 
            is_valid = event[1]['valid']
            participant_events.set_event(condition, start, end, is_valid)
        return participant_events


def load_and_show_event_times(participant_list):
    all_participants_events = load_event_info_from_csv(participant_list)
    conditions_of_interest = set([
        "exper_video", "wildlife_video", "familiar_music",
        "tchaikovsky", "break", "family_inter"])
    condition_durations = {}
    condition_avg_durations = {}
    condition_counts = {}
    for condition in conditions_of_interest:
        condition_duration = 0
        condition_count = 0
        print(f"Condition: {condition}")
        for participant, participant_events in all_participants_events.items():
            try:
                start, end = participant_events.get_event_studytimes(condition)
                duration = participant_events.get_event_duration(condition)
                condition_duration += duration
                condition_count += 1
                print(f"{participant}: {start} -- {end} [{duration} secs]")
            except KeyError:
                pass
            except ValueError:
                pass
        condition_counts[condition] = condition_count
        condition_durations[condition] = condition_duration
        condition_avg_durations[condition] = condition_duration/condition_count
        
        print()            
    print()
    print(f"condition_counts = {condition_counts}")
    print(f"condition_durations = {condition_durations}")
    print(f"condition_avg_durations = {condition_avg_durations}")
    
def load_and_show_valid_events(participant_list):
    all_participants_events = load_event_info_from_csv(participant_list)
    conditions_of_interest = set([
        "exper_video", "wildlife_video", "familiar_music",
        "tchaikovsky", "break", "family_inter"])
#    conditions_of_interest = set([
#        "exper_video", "wildlife_video", "familiar_music",
#        "tchaikovsky", "family_inter"])
##    all_good = []
##    for participant, participant_events in all_participants_events.items():
##        these_conditions = participant_events.get_good_conditions()
##        print(f"{participant}: {these_conditions}")
##        if not these_conditions.issuperset(conditions_of_interest):
##            print(f"\tmissing: {conditions_of_interest.difference(these_conditions)}")
##        else:
##            all_good.append(participant)
##    print(f"all_good = {all_good}")
    raise NotImplementedError('Yet')
    
if __name__ == '__main__':
    load_and_show_event_times()
