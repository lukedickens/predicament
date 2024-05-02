import json
import configparser


STRING_ELEMENTS = {}
STRING_ELEMENTS['LOAD'] = [
    'data_format'

    ]
STRING_ELEMENTS['WINDOWED'] = [
    'group_col', 'target_col'
    ]
STRING_ELEMENTS['FEATURED'] = [
    ]
INT_ELEMENTS = {}
INT_ELEMENTS['LOAD'] = [
    "n_channels",
    "sample_rate",
    "window_size",
    "window_step",
    ]
INT_ELEMENTS['WINDOWED'] = [
    ]
INT_ELEMENTS['FEATURED'] = [
    ]

COMPLEX_ELEMENTS = {}
COMPLEX_ELEMENTS['LOAD'] = [
    "participant_list",
    "conditions",
    "channels", 
    "label_mapping",
    "label_groups"]
COMPLEX_ELEMENTS['WINDOWED'] = [
    'label_cols']
COMPLEX_ELEMENTS['FEATURED'] = [
    'feature_set', 'feature_names'
    ]

def config_to_dict(config):
    if type(config) is dict:
        return config
    dict_config = {}
    for part in config.keys():
        subconfig = config[part]
        subdict = {}
        for element in subconfig.keys():
            if element in STRING_ELEMENTS[part]:
                subdict[element] = subconfig[element]
            elif element in INT_ELEMENTS[part]:
                subdict[element] = int(subconfig[element])
            elif element in COMPLEX_ELEMENTS[part]:
                subdict[element] = json.loads(subconfig[element].replace("'",'"'))
            else:
                raise ValueError(
                    f"Unrecognised config element part {part}, element {element}")
        dict_config[part] = subdict
    return dict_config        
                
def dict_to_config(dict_):
    config = configparser.ConfigParser()
    for key1, subdict in dict_.items():
        config[key1] = {}
        for key2, data in subdict.items():
            config[key1][key2] = str(data).replace("'",'"')
    return config
