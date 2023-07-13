# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 23:59:34 2022

@author: Zerui Mu

"""
import os
BASE_DATA_FOLDER = os.environ.get('PREDICAMENT_DATA_DIR', 'data')
STUDY_DATA_FOLDER = os.path.join(BASE_DATA_FOLDER, r'CARE_HOME_DATA') +'/'
EVENT_DETAILS_PATH = os.path.join(STUDY_DATA_FOLDER, './event_details.csv')
EXP_INFO_PATH = os.path.join(STUDY_DATA_FOLDER, './exp_info.csv')

EVALUATION_BASE_SUBDIR = 'evaluation_data'
EVALUATION_BASE_PATH = os.path.join(BASE_DATA_FOLDER)
if not os.path.exists(EVALUATION_BASE_PATH):
    print(
        f"Creating train test base dir at:\n\t{EVALUATION_BASE_PATH}")
    os.makedirs(EVALUATION_BASE_PATH)


#TODO needs renaming
E4_file_paths = {
    'VG_01': r'./VG01/E4_8921_15_44/',
    # 'VG_02': None,
    'VG_03': r'./VG03/E4_9921_12_16/',
    'VG_05': r'./VG05/E4_9921_13_24/',
    'VG_06': r'./VG06/E4_51021_13_33/',
    'VG_07': r'./VG07/E4_51021_15_39/',
    'VG_08': r'./VG08/E4_71021_10_42/',
    'VG_09': r'./VG09/E4_11221_14_46/',
    'VG_10': r'./VG10/E4_31221_11_17/',
    'VH_01': r'./VH01/E4_61021_11_03/',
    'VH_02': r'./VH02/E4_61021_13_59/',
    'VH_03': r'./VH03/E4_11221_11_22/'
}

EDF_FILE_SUBPATHS = {
    'VG_01': r'./VG01/8921_15_52.edf',
    'VG_02': r'./VG02/7921_15_18.edf',
    'VG_03': r'./VG03/90921_12_20.edf',
    'VG_05': r'./VG05/90921_13_27.edf',
    'VG_06': r'./VG06/51021_13_40.edf',
    'VG_07': r'./VG07/51021_15_43.edf',
    'VG_08': r'./VG08/71021_10_44.edf',
    'VG_09': r'./VG09/11221_14_59.edf',
    'VG_10': r'./VG10/31221_11_24.edf',
    'VH_01': r'./VH01/61021_11_17.edf',
    'VH_02': r'./VH02/61021_14_3.edf',
    'VH_03': r'./VH03/11221_11_21.edf'
}

EDF_FILE_PATHS = {
    part_ID : os.path.join(STUDY_DATA_FOLDER, subpath) \
        for part_ID, subpath in EDF_FILE_SUBPATHS.items()}

## get the channel id from the channel name
ALL_DREEM_CHANNELS = [ # 1&5, 2&6, 3&4 similar, EEG data (1-7)
    'Accelero Norm', # 0
    'EEG Fpz-O1', # 1
    'EEG Fpz-O2', # 2
    'EEG Fpz-F7', # 3
    'EEG F8-F7', # 4
    'EEG F7-01', # 5
    'EEG F8-O2', # 6
    'EEG Fpz-F8', # 7
    # Positiongram does not exist in VG_01, VG_02, VG_03, VG_05
    # in these files, old code changes index (9-13) to (8-12)
    'Positiongram',  # 8
    'PulseOxy Infrare', # 9
    'PulseOxy Red Hea', # 10
    'Respiration x', # 11
    'Respiration y', # 12
    'Respiration z' # 13
    ]

EEG_CONDITIONS = {
    "setup",
    "baseline",
    "exper_video",
    "wildlife_video",
    "familiar_music",
    "tchaikovsky",
    "break",
    "family_inter",
    "takeoff_EEG",
    # these do not refer to events where there is an EEG signal
    # "break2",
    # "setup_BIS",
    # "baseline_BIS",
    # "familiar_music_2",
    # "family_inter_2",
    # "takeoff_BIS_E4"
}


# get the channel id from the channel name
ALL_DREEM_CHANNELS_LOOKUP = {v:k for k,v in enumerate(ALL_DREEM_CHANNELS) }

DREEM_EEG_CHANNELS = [
    'EEG Fpz-O1', 'EEG Fpz-O2', 'EEG Fpz-F7', 'EEG F8-F7', 'EEG F7-01',
    'EEG F8-O2', 'EEG Fpz-F8']
# Analysis suggests that there may be extreme correlation in some pairs of
# channels so the following minimal set is suggested
DREEM_MINIMAL_CHANNELS = [
    'EEG Fpz-O1', 'EEG Fpz-O2', 'EEG Fpz-F7', 'EEG Fpz-F8']
DREEM_CHANNELS_EXCLUDED = [ ch_name for ch_name in ALL_DREEM_CHANNELS_LOOKUP 
    if ch_name not in DREEM_EEG_CHANNELS]
# not used any more (previously the first 4 characters were removed
EEG_CHANNEL_SHORTNAMES = {ch_name:ch_name[4:] for ch_name in DREEM_EEG_CHANNELS}

FULL_PARTICIPANT_LIST = list(EDF_FILE_PATHS.keys())
# participants will all valid events in the target set of conditions
TARGET_CONDITIONS =   [
        "exper_video", "wildlife_video", "familiar_music",
        "tchaikovsky", "family_inter"]

    # participant missing exp video, break and family
E4_participant_ids = list(E4_file_paths.keys())

# A uniform buffer to eliminate the noise at the start of each event, in seconds (S).
BUFFER_INTERVAL = 30
DEFAULT_SAMPLE_RATE = 250 # EEG sample rate in Hz
# approx 4 seconds at 250 Hz
DEFAULT_WINDOW_SIZE = 1024
# approx 2 second overlap
DEFAULT_WINDOW_STEP = DEFAULT_WINDOW_SIZE//2






## old names
EEG_buffer = BUFFER_INTERVAL 
E4_buffer = BUFFER_INTERVAL
VG_Hz = DEFAULT_SAMPLE_RATE

VG_file_paths = EDF_FILE_SUBPATHS 
VG_participant_ids = FULL_PARTICIPANT_LIST

MOTOR_MOVEMENT_DATA_FOLDER = os.path.join(BASE_DATA_FOLDER, r'EEG-Motor-Movement-Imagery-Dataset') +'/'

