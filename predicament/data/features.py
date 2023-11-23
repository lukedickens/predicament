import numpy as np
import scipy as sp
import scipy.stats
from strenum import StrEnum
from itertools import combinations
from math import log
import scipy.signal # using butter, lfilter
from scipy.fft import fft, ifft

from statsmodels.tsa.ar_model import AutoReg # for Auto Regression (arCoeff) features

Feature = StrEnum(
    'Feature',
    ['Mean', 'SD', 'MAD', 'Max', 'Min', 'SMA', 'Energy', 'IQR', 'Entropy',
    'arCoeff', 'Correlation', 'MaxFreqInd', 'MeanFreq', 'FreqSkewness',
    'FreqKurtosis', 'EnergyBands'])
Features = set([f for f in Feature])


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

# just takes the bottom triangular part of the correlations matrix
def correlations(X):
    return np.array([ e for i,r in enumerate(np.corrcoef(X)) for e in r[i+1:]])
    
def convert_timeseries_to_features(
        X, feature_set,
        entropy_window_size=10, entropy_tol=0.6,
        auto_reg_order=3):
    """
    parameters
    ----------
    X - time-series.    Assumes X is 2d np.array CxT, where C is channels and T is time-points 
    feature_set - set of features to use. Can be set of strings or set of Feature enum types.
    """
    C, T = X.shape
    features = np.zeros(0)
    
    # | mean	|	Mean value |
    if Feature.Mean in feature_set:
        features = np.concatenate((features, np.mean(X,axis=1)))
    # | std	|	Standard deviation |
    if Feature.SD in feature_set:    
        features = np.concatenate((features, np.std(X,axis=1)))
    # | mad	|	Median absolute value/deviation |
    if Feature.MAD in feature_set:
        features = np.concatenate(
            [features, scipy.stats.median_abs_deviation(X,axis=1)])
    # | max	|	Largest values in array |
    if Feature.Max in feature_set:
        features = np.concatenate([features, np.max(X,axis=1)])
    # | min	|	Smallest value in array |
    if Feature.Min in feature_set:
        features = np.concatenate([features, np.min(X,axis=1)])
    # | sma	|	Signal magnitude area |
    if Feature.SMA in feature_set:
        raise NotImplementedError('Not sure whether this is distinct from MAD')
        # not sure how useful for general signals
        # see: https://en.wikipedia.org/wiki/Signal_magnitude_area
        # seems similar to median absolute deviation
    # | energy	|	Average sum of the squares |
    if Feature.Energy in feature_set:
        features = np.concatenate([features, np.mean(X**2,axis=1)])
    # | iqr	|	Interquartile range |
    if Feature.Energy in feature_set:
        features = np.concatenate([features, scipy.stats.iqr(X,axis=1)])
    # | entropy	|	Signal Entropy |
    if Feature.Entropy in feature_set:
        # assuming sample entropy
        # see: https://academic.oup.com/biomethods/article/4/1/bpz016/5634143
        # see: https://www.mdpi.com/1099-4300/20/10/764
        # entropy_tol guidance is to use 0.05*(np.mean(np.std(X,axis=1)))
        entropies = [
            sample_entropy(list(ts), entropy_window_size, entropy_tol) for ts in X]
        features = np.concatenate([features, entropies])
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
        features = np.concatenate([features, arCoeffs.flatten()])
        # | correlation	|	Correlation coefficient |
        features = np.concatenate([features, correlations(X)])
    freq_features = set([
        Feature.MaxFreqInd, Feature.MeanFreq, Feature.FreqSkewness,
        Feature.FreqKurtosis, Feature.EnergyBands])
    # check for non-empty intersection of the sets
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
            features = np.concatenate((features, np.argmax(intensities, axis=1)))
        # | meanFreq	|	Frequency signal weighted average |
        if Feature.MeanFreq in feature_set:
            features = np.concatenate((features,  XFreqMeans.flatten()))
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
         # | energyBand	|	Energy of a frequency interval |
         # | angle	|	Angle between two vectors |
    return features
