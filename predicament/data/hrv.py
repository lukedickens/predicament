import pandas as pd
import numpy as np
import scipy.stats
import os
import logging
logger = logging.getLogger(__name__)

from predicament.utils.config import E4_FULL_DIRPATHS
from predicament.utils.config import E4_IBI_FILE


## To move to an appropriate module
# get ibi file for the participant and read+preprocess dataframe
def read_and_prepare_ibi_file(participant):
    ibi_path = os.path.join(E4_FULL_DIRPATHS[participant], E4_IBI_FILE)
    logger.info(f"reading ibi_path = {ibi_path}")
    ibi_df = pd.read_csv(ibi_path, header=None)
    df_t = ibi_df.T
    first_row = df_t.pop(0)
    file_start_time = first_row[0]
    ibi_df = df_t.T
    mapper={0:'local time', 1:'ibi'}
    ibi_df = ibi_df.rename(columns=mapper)
    # the universal times of the beats are the local times
    # plus the universal start time for the file
    ibi_df['unix time'] = ibi_df['local time'] + file_start_time
    return ibi_df


def getrmssds_for_participant_condition(
        start_end_times, successive_differences, sd_unix_times,
        default_sd, normed_rmssd):
    """
    inputs
    ------
    start_end_times - start end times of window
    successive_differences - of ibis 
    sd_unix_times - start times for the successive differences 
    default_sd - default successive difference for imputation
    norm_result [bool] - should rmssds be normalised by the default_sd    
    """
    # reshape successive differences
    successive_differences = successive_differences.reshape(-1,1)
    # now we create a matrix whose rows are ibi entries
    # and columns are window start and end times, true for cell
    # (i,j) if ibi row i is in window j
    sd_unix_times = sd_unix_times.reshape(-1,1)
    filter_stack = (sd_unix_times > start_end_times.T[0,:]) & (sd_unix_times <= start_end_times.T[1,:])
    filter_stack = filter_stack.astype(int)

    # the filter is used to pull out only successive differences from the 
    # window in question (with marginal overlap to calculate the final succesive difference)
    sum_ssds = np.sum(((successive_differences**2)*filter_stack), axis=0)
    # we want to calculate a denominator for each row based on how many 
    # true elements are in that column of the filter stack
    denominator = np.sum(filter_stack,axis=0).reshape(1,-1)
    # the mean squared successive difference for each window is then the normalised sum
    # Note: we avoid dividing by zero by forcing empty columns in filterstack to 
    # be counted as having sum 1
    mssds = sum_ssds/np.maximum(denominator,1)
    # square root the result
    rmssds = np.sqrt(mssds)
    # now impute rmssd for the columns in filter_stack with too few values
    # from a brief back of the envelope investigation, I make this 2 or less
    rmssds[denominator<=2] = default_sd
    # finally, the sampling rate at 64 Hz means that very small
    # variations are missed. For very rare instances where 3 or 
    # more successive differences are exactly equal, we
    # replace with a default value (default_sd/e seems conservative)
    # This allows us to consider logging the values
    rmssds[rmssds==0] = default_sd*np.exp(-1)
    rmssds = rmssds.flatten()
    if not normed_rmssd:
        return rmssds
    # normalisation then facilitates generalisation across participants
    # normed_rmssds = rmssds/default_sd
    normed_rmssds = rmssds/default_sd
    return normed_rmssds

def update_df_with_rmssds(
        datadf, label_mapping, normed_rmssd=True):
    participants = np.unique(datadf['participant'])
    col_name = 'HRVRMSSD'
    if normed_rmssd:
        col_name = col_name + '[Normed]'
    datadf[col_name] = None
    for participant in participants:
        ibi_df = read_and_prepare_ibi_file(participant)
        # get array of unix times from ibi file
        ibi_unix_times = ibi_df['unix time'].to_numpy()
        ibis = ibi_df['ibi'].to_numpy().astype(float)
        # successive differences are the differences between inter-beat intervals (ibis)
        successive_differences = np.diff(ibis)
        sd_unix_times = ibi_unix_times[:-1]
        # we want to impute the successive difference if there are no
        # data within a given time window. We do this with the median
        # as default value. This also serves as a normaliser if
        # normalisation is required  there may be other options
        default_sd = np.sqrt(np.median(successive_differences**2))
        participant_filter = datadf['participant'] == participant 
        for condition_id, condition_name in enumerate(label_mapping):
            pc_filter = participant_filter & (datadf['condition'] == condition_id)
            #print(f"datadf.columns = {datadf.columns}")
            start_end_times = datadf.loc[pc_filter, ['start time','end time']].to_numpy()
#             print(f"start_end_times.shape = {start_end_times.shape}")
            rmssds = getrmssds_for_participant_condition(
                start_end_times, successive_differences,
                sd_unix_times, default_sd, normed_rmssd)
            if np.any(np.isnan(rmssds)):
                raise ValueError(
                    f"rmssds contains nan results, count {np.sum(np.isnan(rmssds))}, default_sd is {default_sd}")
            # now load back into the matrix via the filter
            datadf.loc[pc_filter,(col_name,)] = rmssds

