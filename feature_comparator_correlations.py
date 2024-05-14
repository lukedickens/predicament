import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import itertools
import json
import math
import scipy.stats
import seaborn as sns
import os

import logging
import logging.handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
#
# always write everything to the rotating log files
if not os.path.exists('logs'): os.mkdir('logs')
log_file_handler = logging.handlers.TimedRotatingFileHandler(
    'logs/args.log', when='M', interval=2)
log_file_handler.setFormatter( logging.Formatter(
    '%(asctime)s [%(levelname)s](%(name)s:%(funcName)s:%(lineno)d): %(message)s') )
log_file_handler.setLevel(logging.DEBUG)
logger.addHandler(log_file_handler)
#
# also log to the console at a level determined by the --verbose flag
console_handler = logging.StreamHandler() # sys.stderr
console_handler.setLevel(logging.CRITICAL) # set later by set_log_level_from_verbose() in interactive sessions
console_handler.setFormatter( logging.Formatter('[%(levelname)s](%(name)s): %(message)s') )
logger.addHandler(console_handler)

#
from predicament.utils.config import FEATURED_BASE_PATH
from predicament.utils.config import RESULTS_BASE_PATH
from predicament.utils.config import establish_path
from predicament.utils.file_utils import load_dataframe_and_config
from predicament.utils.file_utils import load_featured_dataframe_and_config

from predicament.data.features import IDEAL_FEATURE_GROUP
from predicament.data.features import STATS_FEATURE_GROUP
from predicament.data.features import INFO_FEATURE_GROUP
from predicament.data.features import FREQ_FEATURE_GROUP
from predicament.data.features import filter_features
from predicament.data.features import get_feature_group
from predicament.data.features import filter_and_group_feature_names
from predicament.data.features import derive_feature_types
from predicament.data.features import filter_and_group_featured_df
from predicament.data.features import robustify_feature_names
from predicament.data.features import replace_channel_ids_with_channel_names
from predicament.data.features import construct_feature_groups

from predicament.data.features import derive_feature_types
from predicament.analysis.correlations import scatter_plot_features
from predicament.analysis.correlations import get_feature_feature_correlations
from predicament.analysis.correlations import plot_feature_feature_correlations
from predicament.analysis.correlations import plot_density_of_correlated_features
from predicament.analysis.correlations import pval_from_pearsonsr
from predicament.analysis.correlations import analyse_feature_vs_comparator

from predicament.utils.dataframe_utils import compute_and_add_aggregating_column
from predicament.utils.dataframe_utils import aggregate_rows_by_grouped_index
from predicament.utils.dataframe_utils import split_column_into_independent_columns

def main(subdirs=None, **kwargs):
    if subdirs is None:
        subdirs = [
            'binary_E4_4secs', 'binary_E4_10secs',
            'binary_dreem_4secs', 'binary_dreem_10secs',
            'E4_4secs', 'E4_10secs',
            'dreem_4secs', 'dreem_10secs']

    for subdir in subdirs:
        load_and_analyse_feature_comparator_correlations(
            subdir, **kwargs)
            
def load_and_analyse_feature_comparator_correlations(
        subdir,
        group_names=None, comparators=None, group_bys=None,
        allow_unsafe_features=False,
        allow_overlapping_windows=False,
        add_agg_column = False,
        aggregate_rows = True,
        remove_rows_with_nans = False,
        row_agg_element='cell',
        row_agg_mode = 'abs_max',
        annot=False, # whether to give r value in cell
        row_height = 0.3,
        col_width = 0.3,
        granularity = 40,
        **kwargs):
    if group_names is None:
        group_names = ['stats', 'info', 'freq']
    remove_overlapping_windows = not allow_overlapping_windows

    if comparators is None:
        comparators = ['condition', 'participant']
    if group_bys is None:
        group_bys = ['type', 'channel']

    # load the data
    featured_data_dir = os.path.join(FEATURED_BASE_PATH,subdir)
    featured_df, config = load_dataframe_and_config(
        featured_data_dir, 'featured.csv')
    # filter out unwanted features and cells
    featured_df, config, group_feature_mapping = filter_and_group_featured_df(
        featured_df, config, group_names,
        allow_unsafe_features, remove_overlapping_windows)

    results_dir = establish_path(RESULTS_BASE_PATH, subdir)
    logger.info(f"results_dir  = {results_dir}")
    for comparator in comparators:
        for group_by in group_bys:
            correlation_mtx = analyse_feature_vs_comparator(
                featured_df, config, results_dir, comparator, group_by,
                add_agg_column, aggregate_rows, remove_rows_with_nans,
                row_agg_element, row_agg_mode, 
                annot, row_height, col_width, granularity)    
    plot_pearsons_r_versus_p_value(
        featured_df, correlation_mtx, results_dir)
    
def plot_pearsons_r_versus_p_value(
        featured_df, correlation_mtx, results_dir, max_abs_pval=0.1):
    n = len(featured_df.index)
#    max_abs_pval = np.nanmax(np.abs(correlation_mtx.values))
    rhos = np.linspace(-max_abs_pval,max_abs_pval,101)
    plt.figure()
    pvals = pval_from_pearsonsr(rhos,n)
    plt.plot(rhos, pvals)
    plt.xlabel("pearsons r")
    plt.ylabel("p-value")
    plt.yscale('log')
    _ = plt.title("Significance of pearsons r values")    
    pngfname = f"pearsons_r_versus_p_value.png"
    pngfpath = os.path.join(results_dir, pngfname)
    print(f"saving pearsons_r vs p_value results to {pngfpath}")
    plt.savefig(pngfpath)
    
def create_parser():
    import argparse
    description= """
        Loads study data, then windows and (typically) partitions it according
        to the configuration parameters then saves down to file."""
    parser = argparse.ArgumentParser(
        description=description,
        epilog='See git repository readme for more details.')
    # partition arguments
    # general        
    parser.add_argument('-V', '--version', action="version", version="%(prog)s 0.1")
    parser.add_argument(
        '-v', '--verbose', action="count",
        help="verbose level... repeat up to three times.")
    return parser

def set_log_level_from_verbose(args):
    if not args.verbose:
        console_handler.setLevel('ERROR')
    elif args.verbose == 1:
        console_handler.setLevel('WARNING')
    elif args.verbose == 2:
        console_handler.setLevel('INFO')
    elif args.verbose >= 3:
        console_handler.setLevel('DEBUG')
    else:
        logger.critical("UNEXPLAINED NEGATIVE COUNT!")
        
if __name__ == '__main__':
    args = create_parser().parse_args()
    set_log_level_from_verbose(args)
    kwargs = vars(args)
    kwargs.pop('verbose')
    main(**kwargs)
    
