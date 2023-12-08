import numpy as np
import scipy as sp
import scipy.stats
from strenum import StrEnum
from itertools import combinations
from math import log
import scipy.signal # using butter, lfilter
from scipy.fft import fft, ifft
# see https://github.com/Mottl/hurst
from hurst import compute_Hc




from statsmodels.tsa.ar_model import AutoReg # for Auto Regression (arCoeff) features

Feature = StrEnum(
    'Feature',
    ['Mean', 'SD', 'MAD', 'Max', 'Min', 'SMA', 'Energy', 'IQR', 'Correlation',
    'SampleEntropy', 'arCoeff',  'Hurst', 'LyapunovExponent',
    'MaxFreqInd', 'MeanFreq', 'FreqSkewness', 'FreqKurtosis']) # , 'EnergyBands'])
Features = set([f for f in Feature])
MAXIMAL_FEATURE_GROUP = set([str(f) for f in Features])
STATS_FEATURE_GROUP = set(
    ['Mean', 'SD', 'MAD', 'Max', 'Min', 'Energy', 'IQR', 'Correlation'])
# Timings for 2 sec window on Care home data
# SampleEntropy 120 hours
# arCoeff 5.5 minutes
INFO_FEATURE_GROUP = set(
    ['SampleEntropy', 'arCoeff',  'Hurst', 'LyapunovExponent']) # 'LempelZivEntropy'
FREQ_FEATURE_GROUP = set(
    ['MaxFreqInd', 'MeanFreq', 'FreqSkewness', 'FreqKurtosis']) #, 'EnergyBands'])


#
## sample entropy from Wikipedia
def sample_entropy(timeseries_data:list, window_size:int, r:float):
    B = get_matches(construct_templates(timeseries_data, window_size), r)
    A = get_matches(construct_templates(timeseries_data, window_size+1), r)
    return -log(A/B)

def construct_templates(timeseries_data:list, m:int=2):
    num_windows = len(timeseries_data) - m + 1
    return [timeseries_data[x:x+m] for x in range(0, num_windows)]

def get_matches(templates:list, r:float):
    return len(list(filter(lambda x: is_match(x[0], x[1], r), combinations(templates, 2))))

def is_match(template_1:list, template_2:list, r:float):
    return all([abs(x - y) < r for (x, y) in zip(template_1, template_2)])

## sample entropy from Wikipedia
#

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

## Lempel-Ziv entropy estimator
# adapted from:
# https://stackoverflow.com/questions/46296891/entropy-estimator-based-on-the-lempel-ziv-algorithm-using-python
import math
def contains(small, big):
    try:
        big.tobytes().index(small.tobytes())//big.itemsize
        return True
    except ValueError:
        return False

def lempel_ziv_entropy(l):
    """
    input
    -----
    l - a 1d numpy timeseries array 
    """
    n = len(l)
    sum_gamma = 0

    for i in range(1, n):
        sequence = l[:i]

        for j in range(i+1, n+1):
            s = l[i:j]
            if contains(s, sequence) != True:
                sum_gamma += len(s)
                break

    ae = 1 / (sum_gamma / n) * math.log(n)
    return ae
## End of Lempel-Ziv entropy estimator


# just takes the bottom triangular part of the correlations matrix
def correlations(X):
    return np.array([ e for i,r in enumerate(np.corrcoef(X)) for e in r[i+1:]])
    
def convert_timeseries_to_features(
        X, feature_set,
        entropy_window_size=10, entropy_tol=0.6,
        auto_reg_order=3,
        hurst_kind='price', hurst_simplified=True):
    """
    parameters
    ----------
    X - time-series.    Assumes X is 2d np.array CxT, where C is channels and T is time-points 
    feature_set - set of features to use. Can be set of strings or set of Feature enum types.
    """
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
        entropies = [
            sample_entropy(list(ts), entropy_window_size, entropy_tol) for ts in X]
        features = np.concatenate([features, entropies])
        feature_names += [f'SampleEntropy{d}' for d in range(len(entropies))]
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
            xchan = xchan[np.insert(np.diff(xchan).astype(np.bool), 0, True)]
            # calculate exponent
            lyapunovs[c] = np.mean(np.log(np.abs(np.diff(xchan))))
        features = np.concatenate([features, lyapunovs])
        feature_names += [f'LyapunovExponent{d}' for d in range(lyapunovs.size)]
    if Feature.LempelZivEntropy in feature_set: 
        # from
        lempel_ziv_entropies = np.empty(C)
        for c, xchan in enumerate(X):
            lempel_ziv_entropies[c] = lempel_ziv_entropy(xchan)
        features = np.concatenate([features, lempel_ziv_entropies])
        feature_names += [f'LempelZivEntropy{d}' for d in range(lempel_ziv_entropies.size)]
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
            XFreqM2 = np.sum(intensities*(freqBands - XFreqMeans)**2,axis=1)/np.sum(intensities, axis=1)
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
