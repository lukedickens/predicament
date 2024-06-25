import numpy as np
import scipy as sp
import pandas as pd
import scipy.stats
import copy
from strenum import StrEnum
from itertools import combinations
import math
from math import log
import scipy.signal # using butter, lfilter
from scipy.fft import fft, ifft
# see https://github.com/Mottl/hurst
from hurst import compute_Hc
import logging
logger = logging.getLogger(__name__)
import re
re_nonalpha = re.compile('[^a-zA-Z]')


from predicament.utils.dataframe_utils import drop_nan_cols
from predicament.utils.dataframe_utils import drop_inf_cols

from predicament.measures.lempel_ziv import lempel_ziv_entropy
from predicament.measures.lempel_ziv import lempel_ziv_casali_flow
from predicament.measures.lempel_ziv import lempel_ziv_casali_loop

# for Auto Regression (arCoeff) features
from statsmodels.tsa.ar_model import AutoReg 

Feature = StrEnum(
    'Feature',
    ['Mean', 'SD', 'MAD', 'Max', 'Min', 'SMA', 'Energy', 'IQR', 'Correlation',
    'SampleEntropy', 'arCoeff',  'Hurst', 'LyapunovExponent', 'LempelZivEntropy', 
    'LempelZivComplexity',
    'MaxFreqInd', 'MeanFreq', 'FreqSkewness', 'FreqKurtosis']) # , 'EnergyBands'])
Features = set([f for f in Feature])
MAXIMAL_FEATURE_GROUP = set([str(f) for f in Features])
STATS_FEATURE_GROUP = set(
    ['Mean', 'SD', 'MAD', 'Max', 'Min', 'IQR', 'Correlation'])
    # 'Energy' excluded as very highly correlated with SD
# Timings for 4 sec window on Care home data
# SampleEntropy 5.5 hours (1.5 secs/it)
# LempelZivEntropy ?? hours (3.25 secs/it)
# LempelZivComplexity (flow) ?? hours (3.25 secs/it)
# LempelZivComplexity (alt) ?? hours (3.25 secs/it)
# arCoeff 5.5 minutes
# Timings for 10 sec window on Care home data
# SampleEntropy  hours ( secs/it)
# LempelZivEntropy 21 hours (7.7 secs/it)
# LempelZivComplexity ?? hours (2.13 it/secs)
INFO_FEATURE_GROUP = set(
    ['arCoeff',  'Hurst', 'LyapunovExponent', 'LempelZivEntropy', 'LempelZivComplexity','SampleEntropy']) # 
FREQ_FEATURE_GROUP = set(
    ['MaxFreqInd', 'MeanFreq', 'FreqSkewness', 'FreqKurtosis']) #, 'EnergyBands'])
SUPPORTED_FEATURE_GROUP = set(
    list(STATS_FEATURE_GROUP)+list(INFO_FEATURE_GROUP)+list(FREQ_FEATURE_GROUP)
    + ['HRVRMSSD'])
PROBLEMATIC_FEATURE_GROUP = set([
#    'SampleEntropy',
    'LempelZivEntropy', # LempelZivEntropy expensive and may be imperfectly implemented
    'Energy', # Energy highly correlated with SD 
    'FreqSkewness']) # FreqSkewness highly correlated with MeanFreq
IDEAL_FEATURE_GROUP = SUPPORTED_FEATURE_GROUP - PROBLEMATIC_FEATURE_GROUP


def get_feature_group(group_name, data_format, allow_unsafe_features=False):
    if group_name  == 'stats':
        feature_group = STATS_FEATURE_GROUP
    elif group_name  == 'info':
        feature_group = INFO_FEATURE_GROUP
    elif group_name  == 'freq':
        feature_group = FREQ_FEATURE_GROUP
    else:
        raise ValueError(f"Unrecognised group name {group_name}")
    if not allow_unsafe_features:
        feature_group = IDEAL_FEATURE_GROUP.intersection(feature_group)
    return feature_group

def filter_and_group_feature_names(
        group_names, feature_names, data_format,
        allow_unsafe_features=False):
    """
    
    """
    group_type_mapping = {}
    group_feature_mapping = {}
    for group_name in group_names:
        group_types = get_feature_group(
            group_name, data_format, allow_unsafe_features)
        group_type_mapping[group_name] = group_types
        group_features = filter_features(
            feature_names, group_types)
        group_feature_mapping[group_name] = group_features
#        print(f"{group_name}_features:\n\t{group_features}")
    return group_type_mapping, group_feature_mapping

def remove_features_with_inf_values(featured_df, config):
    """
    """
    config = copy.deepcopy(config)
    feature_names = config['FEATURED']['feature_names']
    feature_names.sort()

    # Filter columns with infinite values and record removed columns
    # makes strong assumption that only feature columns are infinite
    featured_df, removed_columns = drop_inf_cols(featured_df)
    if not np.all([rem in feature_names for rem in removed_columns]):
        raise ValueError(
            "Trying to remove infinite features but non-features removed")
    #    
    # outputting for debug
    logger.debug("Before removal")
    logger.debug(
        f"feature_names= {feature_names} (len = {len(feature_names)})")
    feature_names = [ f for f in feature_names if not f in removed_columns]
    feature_names.sort()
    config['FEATURED']['feature_names'] = feature_names
    #    
    # outputting for debug
    logger.debug("After removal")
    logger.debug(
        f"feature_names= {feature_names} (len = {len(feature_names)})")
    #
    return featured_df, config


def robustify_feature_name(feature_name):
    # Using regular expression to find digits at the end of the feature_name
    match = re.search(r'(\d+)$', feature_name)
    if match:
        # If digits found, replace them with digits inside square brackets
        transformed_feature_name = re.sub(r'(\d+)$', r'[\1]', feature_name)
        return transformed_feature_name
    else:
        # If no digits found, just append the original feature_name
        return feature_name

def replace_channel_id_with_channel_name(feature_name, id_to_name_mapping):
    feature_name = robustify_feature_name(feature_name)
    # Using regular expression to find digits inside square brackets
    # at the end of the feature_name
    match = re.search(r'(\[\d+\])$', feature_name)
    if match:
        # Extract the digits found inside square brackets
        digits_in_brackets = match.group(1)
        # Extract digits from inside the brackets
        digits = re.search(r'(\d+)', digits_in_brackets).group(1)
        # Check if the digits exist in the id_to_name_mapping
        if digits in id_to_name_mapping:
            # Replace the digits inside square brackets with the corresponding feature_name from the id_to_name_mapping
            replaced_feature_name = re.sub(r'\[\d+\]$', f'[{id_to_name_mapping[digits]}]', feature_name)
            return replaced_feature_name
        else:
            # If digits not found in id_to_name_mapping, return the original feature_name
            return feature_name
    else:
        # If no digits inside square brackets found, return the original feature_name
        return feature_name
        
def robustify_feature_names(feature_names):
    transformed_feature_names = []
    for feature_name in feature_names:
        transformed_feature_names.append(robustify_feature_name(feature_name))
    return transformed_feature_names
    
def replace_channel_ids_with_channel_names(
        feature_names, id_to_name_mapping):
    transformed_feature_names = []
    for feature_name in feature_names:
        transformed_feature_names.append(
            replace_channel_id_with_channel_name(
                feature_name, id_to_name_mapping))
    return transformed_feature_names

def construct_feature_groups(config, group_by):
    feature_names = config['FEATURED']['feature_names']
    if group_by == 'type':
        # feature_cond_corr_df.index
        base_feature_types = config['FEATURED']['feature_set']
        base_feature_types.sort()
        feature_groups = {
            t:[f  for f in feature_names if f.startswith(t) ] for t in base_feature_types}
    if group_by == 'channel':
        channels = config['LOAD']['channels']
        id_to_name_mapping = {
            str(ch): channel for ch, channel in enumerate(channels)}
        feature_names = config['FEATURED']['feature_names']
        excludes = ['Correlation', 'arCoeff']
        for exclude in excludes:
            feature_names = [
                f for f in feature_names if not f.startswith(exclude) ]
        clean_feature_names = replace_channel_ids_with_channel_names(
            feature_names, id_to_name_mapping)
#        print(f"feature_names = {feature_names}") 
#        print(f"clean_feature_names = {clean_feature_names}") 
        feature_groups = {}
        for channel in channels:
            channel_features = []
            for f, clean in zip(feature_names, clean_feature_names):
                if clean.endswith(f'[{channel}]'):
                    channel_features.append(f)
            if len(channel_features) > 0:
                feature_groups[channel] = channel_features
    # clean up empty groups
    keys = list(feature_groups.keys())
    for key in keys:
        if len(feature_groups[key]) == 0:
            feature_groups.pop(key)
    return feature_groups
    
def filter_and_group_featured_df(
        featured_df, config, group_names,
        allow_unsafe_features=False, remove_overlapping_windows=False):
    """
    filters out undesired names and groups these for easier analysis.
    also removes any features with infinite values
    """
    config = copy.deepcopy(config)
    data_format = config['LOAD']['data_format']
    #
    featured_df, config = remove_features_with_inf_values(
        featured_df, config)
    #
    feature_names = config['FEATURED']['feature_names']
    tuple_ = filter_and_group_feature_names(
            group_names, feature_names, data_format,
            allow_unsafe_features)
    group_type_mapping, group_feature_mapping = tuple_
    # extract feature names
    feature_names= [
        f for group_name in group_names \
            for f in group_feature_mapping[group_name]  ]
    # extract all feature types
    feature_types= [
        t for group_name in group_names \
            for t in group_type_mapping[group_name]  ]
    
    config['FEATURED']['feature_names'] = feature_names
    config['FEATURED']['feature_set'] = feature_types
    # now filter dataframe again to exclude non-grouped features
    label_cols = config['WINDOWED']['label_cols']
    all_cols = list(label_cols) + feature_names
    featured_df = featured_df.loc[:,all_cols]
    # remove dataframe rows to ensure non-overlapping windows?
    if remove_overlapping_windows:
        window_size = config['WINDOWED']['window_size']
        window_step = config['WINDOWED']['window_step']
        logger.debug(
            f"Initially there are {len(featured_df.index)} windows")
        logger.info("Removing overlapping windows")
        overlap_factor = window_size // window_step
        row_filter = (featured_df['window index'].values % overlap_factor) == 0
        featured_df = featured_df.loc[row_filter,:]
        logger.debug(
            f"Subsequently we have {len(featured_df.index)} windows")
    return featured_df, config, group_feature_mapping
    
#TODO better in a timeseries module?
def quantize_timeseries(seq, amp_min, amp_max, bits_per=8, binarize=True):
    bins = np.linspace(amp_min, amp_max, 2**bits_per+1)
    if bits_per <= 8:
        quantized = np.empty(seq.size, dtype=np.uint8)
    else:
        raise ValueError(f'Invalid bits per {bits_per}')
    for i in range(2**bits_per):
        quantized[(seq >= bins[i]) & (seq < bins[i+1]) ] = i
    if binarize:
        unpacked = np.unpackbits(quantized)
        if bits_per < 8:
            # by default there are 8 bits per unpacked uint8.
            # We make this shorter  by removing the big end of each encoding.
            unpacked =  unpacked.reshape((-1,8))[:,-bits_per:].flatten()
        return unpacked
    else:
        return quantized


def quantize_multichannel(X, amp_mins, amp_maxes, bits_per=8, binarize=True):
    bins = np.linspace(amp_mins, amp_maxes, 2**bits_per+1, axis=1)
#     print(f"bins = {bins}")
    if bits_per <= 8:
        quantized = np.empty(X.shape, dtype=np.uint8)
    else:
        raise ValueError(f'Invalid bits per {bits_per}')
    for i in range(2**bits_per):
        quantized[(X >= bins[:,i:i+1]) & (X < bins[:,i+1:i+2]) ] = i
    if binarize:
        n_channels = quantized.shape[0]
        unpacked = np.unpackbits(quantized, axis=1)
        if bits_per < 8:
            # by default there are 8 bits per packed quantized vector.
            # We make this shorter
            unpacked =  unpacked.reshape((n_channels, -1,8))[:,:,-bits_per:].reshape((n_channels, -1))
        return unpacked
    else:
        return quantized


#
## sample entropy from Wikipedia
##def sample_entropy(timeseries_data:list, embed_dim:int, tol:float):
##    B = get_matches(construct_templates(timeseries_data, embed_dim), tol)
##    A = get_matches(construct_templates(timeseries_data, embed_dim+1), tol)
##    return -log(A/B)

##def construct_templates(timeseries_data:list, m:int=2):
##    num_windows = len(timeseries_data) - m + 1
##    return [timeseries_data[x:x+m] for x in range(0, num_windows)]

##def get_matches(templates:list, tol:float):
##    return len(list(filter(lambda x: is_match(x[0], x[1], tol), combinations(templates, 2))))

##def is_match(template_1:list, template_2:list, tol:float):
##    return all([abs(x - y) < tol for (x, y) in zip(template_1, template_2)])
## sample entropy from Wikipedia
#

# rewritten sample entropy using numpy arrays
def sample_entropy(timeseries_data, embed_dim, tol):
    templates = construct_templates(timeseries_data, embed_dim)
    B = get_matches(templates, tol)
    
    templates_m1 = construct_templates(timeseries_data, embed_dim + 1)
    A = get_matches(templates_m1, tol)
    
    return -log(A / B)

def construct_templates(timeseries_data, m=2):
    return np.lib.stride_tricks.sliding_window_view(timeseries_data, (m,))

def get_matches(templates, tol):
    combs = np.array(list(combinations(templates, 2)))
    matches = np.sum(np.all(np.abs(combs[:, 0] - combs[:, 1]) < tol, axis=1))
    return matches

## Faster approach using 2d numpy arrays
def sample_entropy_stack(timeseries_data, embed_dim, tols):
    templates = construct_templates_stack(timeseries_data, embed_dim)
    B = get_matches_stack(templates, tols)
    
    templates_m1 = construct_templates_stack(timeseries_data, embed_dim + 1)
    A = get_matches_stack(templates_m1, tols)
    
    return -np.log(A / B)

def construct_templates_stack(timeseries_data, m=2):
    window_shape = (timeseries_data.shape[0], m)
    return np.squeeze(np.lib.stride_tricks.sliding_window_view(
        timeseries_data, window_shape))

def get_matches_stack(templates, tols):
    num_rows, _, _ = templates.shape
    indices = np.arange(num_rows)
    combs = np.array(list(combinations(indices, 2)))
    matches = np.sum(np.all(
            np.abs(templates[combs[:, 0]] - templates[combs[:, 1]]) < tols[np.newaxis, :, np.newaxis],
            axis=2),
        axis=0)
    return matches
    

#
## butterworth bandpass filter from stackoverflow
# https://stackoverflow.com/questions/30659579/calculate-energy-for-each-frequency-band-around-frequency-f-of-interest-in-pytho
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return b, a    
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scipy.signal.lfilter(b, a, data)
    return y
## butterworth bandpass filter from stackoverflow
#



# just takes the bottom triangular part of the correlations matrix
def correlations(X):
    return np.array([ e for i,r in enumerate(np.corrcoef(X)) for e in r[i+1:]])
    
def convert_timeseries_to_features(
        X, feature_set,
        entropy_embed_dim=2, entropy_tol=None, entropy_tols=None,
        auto_reg_order=3, bits_per=1, lzc_type=None, lzc_imp=None,
        lze_imp=None, sament_imp=None,
        amp_means=None, amp_maxes=None, amp_mins=None,
        hurst_kind='price', hurst_simplified=False):
    """
    parameters
    ----------
    X - time-series.    Assumes X is 2d np.array CxT, where C is channels and T is time-points 
    feature_set - set of features to use. Can be set of strings or set of Feature enum types.
    bits_per - when quantizing the data for LempelZiv calculations, we set how many bits to encode each per channel sample. 
    lzc_imp - which implementation of LempelZivComplexity to use
    sament_imp - which implementation of SampleEntroy to use
    """
#    print(f"feature_set = {feature_set}")
#    print(f"type(feature_set) = {type(feature_set)}")
    for f in feature_set:
        if f not in Features:
            raise ValueError(f"Unrecognised feature {f}")
    C, T = X.shape
    features = np.zeros(0)
    feature_names = []
    # | mean	|	Mean value |
    if Feature.Mean in feature_set:
        means = np.mean(X,axis=1)
        features = np.concatenate((features, means))
        feature_names += [f'Mean{d}' for d in range(means.size)]
    # | std	|	Standard deviation |
    if Feature.SD in feature_set:    
        sds = np.std(X,axis=1)
        features = np.concatenate((features, sds))
        feature_names += [f'SD{d}' for d in range(sds.size)]
    # | mad	|	Median absolute value/deviation |
    if Feature.MAD in feature_set:
        mads = scipy.stats.median_abs_deviation(X,axis=1)
        features = np.concatenate(
            [features, mads])
        feature_names += [f'MAD{d}' for d in range(mads.size)]
    # | max	|	Largest values in array |
    if Feature.Max in feature_set:
        maxes = np.max(X,axis=1)
        features = np.concatenate([features, maxes])
        feature_names += [f'Max{d}' for d in range(maxes.size)]
    # | min	|	Smallest value in array |
    if Feature.Min in feature_set:
        mins = np.min(X,axis=1)
        features = np.concatenate([features, mins])
        feature_names += [f'Min{d}' for d in range(mins.size)]
    # | sma	|	Signal magnitude area |
    if Feature.SMA in feature_set:
        raise NotImplementedError('Not sure whether this is distinct from MAD')
        # not sure how useful for general signals
        # see: https://en.wikipedia.org/wiki/Signal_magnitude_area
        # seems similar to median absolute deviation
    # | energy	|	Average sum of the squares |
    if Feature.Energy in feature_set:
        energies = np.mean(X**2,axis=1)
        features = np.concatenate([features, energies])
        feature_names += [f'Energy{d}' for d in range(energies.size)]
    # | iqr	|	Interquartile range |
    if Feature.IQR in feature_set:
        iqrs = scipy.stats.iqr(X,axis=1)
        features = np.concatenate([features, iqrs])
        feature_names += [f'IQR{d}' for d in range(iqrs.size)]
    # | correlation	|	Correlation coefficient |
    if Feature.Correlation in feature_set:
        these_correlations = correlations(X)
        features = np.concatenate([features, these_correlations])
        feature_names += [f'Correlation{d}' for d in range(these_correlations.size)]
    # | entropy	|	Signal Entropy |
    if Feature.SampleEntropy in feature_set:
        # assuming sample entropy
        # see: https://academic.oup.com/biomethods/article/4/1/bpz016/5634143
        # see: https://www.mdpi.com/1099-4300/20/10/764
        # entropy_tol guidance is to use 0.05*(np.mean(np.std(X,axis=1)))
        if entropy_tols is None:
            entropy_tols = np.ones(X.shape[0])*entropy_tol
##        entropies = [
##            sample_entropy(list(ts), entropy_embed_dim, e_tol) for e_tol, ts in zip(entropy_tols,X)]
        entropies = sample_entropy_stack(X, entropy_embed_dim, entropy_tols)
        features = np.concatenate([features, entropies])
        feature_names += [f'SampleEntropy[m={entropy_embed_dim}][{d}]' for d in range(len(entropies))]
    # | arCoeff	|	Autorregresion coefficients |
    if Feature.arCoeff in feature_set:
        # using code from here:
        # https://www.statsmodels.org/dev/examples/notebooks/generated/autoregressions.html
        arCoeffs = np.empty((auto_reg_order+1, C))
        #TODO dimension wise ar model. Does this need revisiting?
        for c in range(C):
            mod = AutoReg(X[c,:], auto_reg_order, old_names=False)
            res = mod.fit()
            arCoeffs[:,c] = res.params
        arCoeffs = arCoeffs.flatten()
        features = np.concatenate([features, arCoeffs])
        feature_names += [f'arCoeff{d}' for d in range(arCoeffs.size)]
    if Feature.Hurst in feature_set:
        hurst_vals = [
            compute_Hc(series, kind=hurst_kind, simplified=hurst_simplified)[:2] 
                for series in X ]
        hurst_Hs, hurst_Cs = zip(*hurst_vals)
        features = np.concatenate([features, hurst_Hs])
        feature_names += [f'Hurst_H{d}' for d in range(len(hurst_Hs))]
        features = np.concatenate([features, hurst_Cs])
        feature_names += [f'Hurst_C{d}' for d in range(len(hurst_Cs))]
    if Feature.LyapunovExponent in feature_set:
        # from: https://en.wikipedia.org/wiki/Lyapunov_exponent    
        # but we consider this channel wise only
        # a solution for univariate series is found here:
        # https://stackoverflow.com/questions/37908104/lyapunov-exponent-python-implementation
        # but this leads to instabilities if the sequence contains duplicates
        # we follow the advice to remove duplicates and use this recipe:
        # https://stackoverflow.com/questions/37839928/remove-consecutive-duplicates-in-a-numpy-array
        # difficult to do without for loop as duplicate removal may lead to 
        # ragged array
        lyapunovs = np.empty(C)
        for c, xchan in enumerate(X):
            # remove duplicates
            xchan = xchan[np.insert(np.diff(xchan).astype(bool), 0, True)]
            # calculate exponent
            lyapunovs[c] = np.mean(np.log(np.abs(np.diff(xchan))))
        features = np.concatenate([features, lyapunovs])
        feature_names += [f'LyapunovExponent{d}' for d in range(lyapunovs.size)]
    Q = None
    if Feature.LempelZivEntropy in feature_set: 
        # from
        Q = quantize_multichannel(
            X, amp_mins, amp_maxes, bits_per=bits_per, binarize=True)
        lempel_ziv_entropies = np.empty(C)
        if (lze_imp is None) or (lze_imp == 'seq'):
            for c, ts in enumerate(Q):
                lempel_ziv_entropies[c] = lempel_ziv_entropy(ts)
        else:
            raise ValueError(f"Unrecognised LempelZivEntropy implementation {lze_imp}")
        features = np.concatenate([features, lempel_ziv_entropies])
        feature_names += [f'LempelZivEntropy[b={bits_per}][{d}]' for d in range(lempel_ziv_entropies.size)]
    if Feature.LempelZivComplexity in feature_set: 
        if Q is None:
            Q = quantize_multichannel(
                X, amp_mins, amp_maxes, bits_per=bits_per, binarize=True)
#        quantized = quantize_data(X)
        if lzc_type == 'casali': 
            if lzc_imp == 'flow':
                lzc = lempel_ziv_casali_flow(Q.T)
            elif lzc_imp == 'loop':
                lzc = lempel_ziv_casali_loop(Q.T)
            feature_names += [f'LempelZivComplexity[b={bits_per}]' ] 
            features = np.concatenate([features, [lzc]])
        else:
            raise ValueError(f'Unrecognised LempelZivComplexity type {lzc_type}')
    ##
    # check for non-empty intersection of the sets freq_features and feature_set
    freq_features = set([
        Feature.MaxFreqInd, Feature.MeanFreq, Feature.FreqSkewness,
        Feature.FreqKurtosis]) # ,TODO add Feature.EnergyBands
    #TODO The following frequency calculations are based on intensities
    #TODO as the probability mass estimates. This needs supporting by 
    #TODO literature, or otherwise validating
    if freq_features & feature_set:
        # convert signal to frequency domain
        # here frequency bands are given in multiples of the base frequency in dB
        # if variable length time-series are to be supported we need
        # to ensure that true frequency bands are used
        Xfreq = fft(X)
        intensities = np.abs(Xfreq)**2
        #print(f"X.shape = {X.shape}")
        #print(f"Xfreq.shape = {Xfreq.shape}")
        freqBands = np.arange(Xfreq.shape[1]).reshape((1,-1))
        XFreqMeans = np.sum(freqBands*intensities,axis=1)/np.sum(intensities, axis=1)
        XFreqMeans = XFreqMeans.reshape((-1,1))
        XFreqM2 = np.sum(intensities*(freqBands - XFreqMeans)**2,axis=1)/np.sum(intensities, axis=1)
        #print(f"freqBands = {freqBands}")
        # | maxFreqInd	|	Largest frequency component |
        if Feature.MaxFreqInd in feature_set:
            max_freq_indices = np.argmax(intensities, axis=1)
            features = np.concatenate((features, max_freq_indices))
            feature_names += [f'MaxFreqInd{d}' for d in range(max_freq_indices.size)]
        # | meanFreq	|	Frequency signal weighted average |
        if Feature.MeanFreq in feature_set:
            features = np.concatenate((features,  XFreqMeans.flatten()))
            feature_names += [f'MeanFreq{d}' for d in range(XFreqMeans.size)]
            #print(f"XmeanFreq = {XmeanFreq}")
            #TODO an alternative mean frequency is to use p=1 (manhattan norm)
            #XmeanFreqAlt = np.sum(freqBands*np.abs(Xfreq),axis=1)/np.sum(np.abs(Xfreq), axis=1)
            #print(f"XmeanFreqAlt = {XmeanFreqAlt}")
         # | skewness	|	Frequency signal Skewness |
        if Feature.FreqSkewness in feature_set:
            freq_skew_bias = False
            # using:
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html
            # if bias is TueThe sample skewness is computed as the
            # Fisher-Pearson coefficient of skewness, i.e. g_1 = m_3/m_2^{3/2}
            # Where the i-th moment: m_i = 1/N \sum_{n=1}^N (x[n] - \bar{x})^i
            # For the unbiased version use:
            # G_1 = \sqrt{N(N-1)}/(N-2) . g_1
            # But we don't have a sample instead we have a histogram weighted by intensities
            # and we don't have a value N. Instead our values are converted from time-domain
            # signal.
            # So we can use the following instead for the ith moment
            # m_i = \sum_{k=1}^K y[k](x[k] - \bar{x})^i /\sum_{k=1}^K y[k]
            # where x[k] is the kth frequency (we use the index) and
            # y[k] is the intensity at that band.
#            XmeanFreq = np.sum(freqBands*intensities,axis=1)/np.sum(intensities, axis=1)
#            XFreqSkews =  np.concatenate((features,  scipy.stats.skew(XFreq, axis=1,))
            XFreqM3 = np.sum(intensities*(freqBands - XFreqMeans)**3,axis=1)/np.sum(intensities, axis=1)
            XFreqSkews = XFreqM3/XFreqM2**(3/2)
            features = np.concatenate((features, XFreqSkews))
            feature_names += [f'FreqSkewness{d}' for d in range(XFreqSkews.size)]
         # | kurtosis	|	Frequency signal Kurtosis |
        if Feature.FreqKurtosis in feature_set:
            #TODO verify
            # From: https://en.wikipedia.org/wiki/Kurtosis
            # Kurtosis is calculated as:Kurt[X] = E[((X-\mu)/\sigma)^4] = \mu_4/\sigma^4
            # where \mu_4 is the 4th standard moment and \sigma is the sd
            # for populations we use: \mu_4 = m_4 and \sigma^2 = m_2
            XFreqM4 = np.sum(intensities*(freqBands - XFreqMeans)**4,axis=1)/np.sum(intensities, axis=1)
            XFreqKurtoses = XFreqM4/XFreqM2**2
            features = np.concatenate((features, XFreqKurtoses))
            feature_names += [f'FreqKurtosis{d}' for d in range(XFreqKurtoses.size)]
         # | energyBand	|	Energy of a frequency interval |
         # TODO: how to do energy bands. May relate to Welch's method see:
         # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
         # Information came from guidance on processing EEG signals here:
         # https://raphaelvallat.com/bandpower.html
         # | angle	|	Angle between two vectors |
    return features, feature_names
    
    
def add_features_to_dataframe(
        existing_df, new_df, label_cols):
    merge_on = label_cols
    left_cols = existing_df.columns
    right_cols = new_df.columns
    # new dataframe will overwrite any columns that are not labels and have
    # matching column names
    preserved_left_cols = [
        c for c in left_cols if (c in merge_on) or (not c in right_cols) ]
    # hide the columns that will be overridden
    existing_df = existing_df.loc[:,preserved_left_cols]
#    print(f"len(existing_df.index) = {len(existing_df.index)}")
#    print(f"len(new_df.index) = {len(new_df.index)}")
#    print(f"merge_on = {merge_on}")
#    for col in merge_on:
#        print(f"existing_df[col].dtype = {existing_df[col].dtype}")    
#        print(f"new_df[col].dtype = {new_df[col].dtype}")    
#        print(f"existing_df[col].unique() = {existing_df[col].unique()}")    
#        print(f"new_df[col].unique() = {new_df[col].unique()}")    
    result_df = pd.merge(existing_df, new_df, how='inner', on=merge_on)
#    print(f"len(result_df.index) = {len(result_df.index)}")
    if len(existing_df.index) != len(result_df.index):
        existing_df.to_csv('/tmp/existing_df.csv')
        new_df.to_csv('/tmp/new_df.csv')
        raise ValueError('Cannot merge as rows do not coincide, saving to /tmp')
    return result_df


# re_nonalpha = re.compile('[^a-zA-Z]')
# feature_types = [(name,re_nonalpha.sub('', name)) for name in feature_names]
# print(f"feature_types = {feature_types}")

def filter_features(feature_names, type_set):
#    return [
#        name for name in feature_names if re_nonalpha.sub('', name) in type_set]
    filtered = [
        name for name in feature_names if re_nonalpha.split(name)[0] in type_set]
    filtered.sort()
    return filtered
    #
#    for name in feature_names:
#        stem  name.split('[')[0].split('_')[0]
            
def derive_feature_types(feature_names, base_feature_types, label_cols=None):
    if label_cols is None:
        label_cols  =  ["participant", "condition", "window index"]
    derived_feature_names = []
    derived_feature_types = set([])
    for f in feature_names:
        if f in label_cols:
            continue
        elif (f[-1] == ']'):
            if (f[:-1].rstrip('0123456789')[-1] == '['):
                f = f[:-1].rstrip('0123456789')[:-1]
        else:
            f = f.rstrip('0123456789')
        for type_ in base_feature_types:
            if f.startswith(type_):
                derived_feature_types.add(f)
                break
    derived_feature_types = list(derived_feature_types)
    derived_feature_types.sort()
    return derived_feature_types
