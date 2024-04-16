import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy as sp
import scipy.stats
from tqdm import tqdm
import itertools
import re
res_digit = r'[0-9]'


from sklearn.metrics import classification_report

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn import preprocessing 


from sklearn.model_selection import RandomizedSearchCV 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from skopt import BayesSearchCV
# parameter ranges are specified by one of below
from skopt.space import Real, Categorical, Integer

# local imports
from predicament.utils.file_utils import load_dataframe_and_config

from predicament.utils.config import FEATURED_BASE_PATH
from predicament.data.features import IDEAL_FEATURE_GROUP

from predicament.evaluation.balancing import get_group_label_counts
from predicament.evaluation.balancing import balance_data
from predicament.evaluation.grouping import get_group_assignments
from predicament.evaluation.staging import get_design_matrix
from predicament.evaluation.staging import get_estimator
from predicament.evaluation.results import output_model_best_from_results
from predicament.evaluation.results import save_results_df_to_file

from predicament.evaluation.hyperparameters import get_param_scopes
from predicament.evaluation.hyperparameters import get_param_search_object

from predicament.models.mlp_wrappers import ThreeHiddenLayerClassifier


def main(**kwargs):
    # high level choices
    subdir = 'dreem_10secs' # dataset
    held_out = 'participant'
    is_balanced = True # balance data set
    use_only_ideal_features = True # restrict to preferred ideal features
    standardise_data = True
    n_iter = 50 # number of iterations for your search
    new_search = True # restarts the search
    random_state = 43
    estimator = get_estimator()
    
    featured_data_dir = os.path.join(FEATURED_BASE_PATH,subdir)
    featured_df, featured_config = load_dataframe_and_config(
        featured_data_dir, 'featured.csv')
    featured_config = config_to_dict(featured_config)


def create_parser():
    import argparse
    description= """
        Loads study data, then windows and (typically) partitions it according
        to the configuration parameters then saves down to file."""
    parser = argparse.ArgumentParser(
        description=description,
        epilog='See git repository readme for more details.')
    # partition arguments
    # specify within or between
    # other parameters to control loaded data
    parser.add_argument(
        '--model', '-m',
        help="""The model type to use""")
    parser.add_argument(
        '--subdir',
        help="""Data subdirectory to use. TYpically a datetime string""")
    parser.add_argument(
        '--feature-group', default='supported',
        help="""Predefined group of features to use""")
    parser.add_argument(
        '--feature-set',
        help="""Specify feature types as comma separated string""")
        
    # general        
    parser.add_argument('-V', '--version', action="version", version="%(prog)s 0.1")
    parser.add_argument('-v', '--verbose', action="count", help="verbose level... repeat up to three times.")
        
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

