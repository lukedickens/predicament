import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# logging
import logging
import logging.handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
#
# always write everything to the rotating log files
if not os.path.exists('logs'): os.mkdir('logs')
log_file_handler = logging.handlers.TimedRotatingFileHandler('logs/args.log', when='M', interval=2)
log_file_handler.setFormatter( logging.Formatter('%(asctime)s [%(levelname)s](%(name)s:%(funcName)s:%(lineno)d): %(message)s') )
log_file_handler.setLevel(logging.DEBUG)
logger.addHandler(log_file_handler)
#
# also log to the console at a level determined by the --verbose flag
console_handler = logging.StreamHandler() # sys.stderr
console_handler.setLevel(logging.CRITICAL) # set later by set_log_level_from_verbose() in interactive sessions
console_handler.setFormatter( logging.Formatter('[%(levelname)s](%(name)s): %(message)s') )
logger.addHandler(console_handler)


from sklearn.metrics import classification_report

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn import preprocessing 


from sklearn.model_selection import RandomizedSearchCV 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from skopt import BayesSearchCV
# parameter ranges are specified by one of below
from skopt.space import Real, Categorical, Integer


#from predicament.utils.file_utils import load_dataframe_and_config
from predicament.utils.file_utils import load_featured_dataframe_and_config
from predicament.utils.config_parser import config_to_dict

from predicament.utils.config import FEATURED_BASE_PATH
from predicament.data.features import IDEAL_FEATURE_GROUP

from predicament.evaluation.balancing import get_group_label_counts
from predicament.evaluation.balancing import balance_data
from predicament.evaluation.grouping import get_group_assignments
from predicament.evaluation.staging import get_design_matrix
from predicament.evaluation.results import output_model_best_from_results
from predicament.evaluation.results import get_model_best_from_results
from predicament.evaluation.results import save_results_df_to_file
from predicament.evaluation.results import save_results_plot_to_file

from predicament.evaluation.hyperparameters import get_param_scopes
from predicament.evaluation.hyperparameters import get_param_search_object
from predicament.evaluation.hyperparameters import get_estimator_and_hyperparamter_excludes

from predicament.models.mlp_wrappers import ThreeHiddenLayerClassifier

    
def main(
        subdir=None,
        held_out=None,
        keep_classes_unbalanced=False, 
        use_unsafe_features=False,
        standardise_data = None,
        estimator=None,
        max_iter_opt = None,
        n_iter=50,
        retrieve_old_search=False,
        random_state = None,
        use_callback = False,
        hyperparameter_overrides=None,
        hyperparameter_excludes=None,
        scoring="f1_macro",
        search_type=None):

    logger.info('Running featured prediction hyperparamter search on:')        
    logger.info(f'\tsubdir: {subdir}')        
    featured_df, featured_config = load_featured_dataframe_and_config(
        subdir)
#    featured_data_dir = os.path.join(FEATURED_BASE_PATH,subdir)
#    featured_df, featured_config = load_dataframe_and_config(
#        featured_data_dir, 'featured.csv')

    if type(random_state) is str:
        random_state = int(random_state)
    random_state = np.random.RandomState(seed=random_state)

    # configuration
    n_channels = featured_config['LOAD']['n_channels']
    label_mapping = featured_config['LOAD']['label_mapping']
    data_format = featured_config['LOAD']['data_format']
    channels = featured_config['LOAD']['channels']
    participant_list = featured_config['LOAD']['participant_list']
    sample_rate = featured_config['LOAD']['sample_rate']
    Fs = sample_rate
    window_size = featured_config['WINDOWED']['window_size']
    window_step = featured_config['WINDOWED']['window_step']
    time = window_size/sample_rate
    logger.info(f"sample_rate: {sample_rate}, n_samples = {window_size}, time: {time}s, n_channels: {n_channels}")
    
    ## balancing
    participant_condition_counts = get_group_label_counts(
        featured_df, 'participant', 'condition')
    # balancing data
    logger.debug(f"keep_classes_unbalanced = {keep_classes_unbalanced}")
    if not keep_classes_unbalanced:
        # balance featured data
        logger.info(f"before balancing: participant_condition_counts = {participant_condition_counts}")
        featured_df = balance_data(featured_df, group_col='participant', label_col='condition')
        participant_condition_counts = get_group_label_counts(featured_df, 'participant', 'condition')
        logger.info(f"after balancing: participant_condition_counts = {participant_condition_counts}")
        is_balanced = True
    else:
        is_balanced = False
    #
    condition_counts = np.sum(participant_condition_counts, axis=0)
    logger.info(
        f"condition window counts {list(zip(label_mapping,condition_counts))}")
    participant_counts = np.sum(participant_condition_counts, axis=1)
    logger.info(
        f"participant window counts {list(zip(participant_list,participant_counts))}")
    n_windows = np.sum(participant_condition_counts)
    logger.info(
        f"total window counts {n_windows}")
    logger.info(f"is_balanced = {is_balanced}")
        
    # define model and hyperparameter search
    if hyperparameter_overrides is None:
        hyperparameter_overrides = dict()
    else:
        raise NotImplementedError(
            "Would need to implement method to set override search spaces from string")
    if hyperparameter_excludes is None:
        hyperparameter_excludes = list()
    else:
        hyperparameter_excludes = hyperparameter_excludes.split(',')
    # the base model to tune
    # store the name
    estimator_name = estimator
    estimator, hyperparameter_excludes = get_estimator_and_hyperparamter_excludes(
        estimator_name, hyperparameter_excludes, max_iter_opt=None)
    logger.info(f"estimator_name = {estimator_name}, estimator = {estimator}, type(estimator) = {type(estimator)}")
    #
    # now create the parameter search object and run the hyperparameter search
    param_scopes = get_param_scopes(
        search_type, estimator, excludes=hyperparameter_excludes, **hyperparameter_overrides)
    logger.info(f"param_scopes = {param_scopes}")

    ## define data properties and data-split
    feature_set = featured_config['FEATURED']['feature_set']
    if not use_unsafe_features:
        feature_set = list(IDEAL_FEATURE_GROUP.intersection(feature_set))   
    logger.info(f"feature_set = {feature_set}")

    # extract input data
    feature_types, feature_names, designmtx = get_design_matrix(
        featured_df, feature_set)
    # extract labels
    labels = featured_df['condition'].values.astype(int)
    #
    if standardise_data:
        scaler = preprocessing.StandardScaler().fit(designmtx)
        designmtx = scaler.transform(designmtx)
    #
    # prepare Hold one group out cross validation
    held_out, groups, group_assignments = get_group_assignments(featured_df)
    n_groups = len(groups)
    # cross validation splits    
    group_kfold = GroupKFold(n_splits=n_groups)
    
    ## Define and Execute hyperparameter search strategy 
    np.int = int
    param_search  = get_param_search_object(
        search_type, estimator, param_scopes=param_scopes, 
        n_iter = n_iter, cv=group_kfold,
        verbose=2, random_state=random_state, n_jobs=-1,
        fit_params={'X': designmtx, 'y': labels, 'callbacks': None},
        scoring=scoring,
        refit=False  # Ensure that the search does not refit the model with the best parameters found so far
    )

    # apply the hyperparameter search
    if use_callback:
        if retrieve_old_search:
            raise NotImplementedError('Would need to load old results')
            param_search.fit_params['search_results'] = intermediate_results
        else:
            on_step, intermediate_results = create_callback_and_storage(param_search)
        try:
            # Fit the random search model
            _ = param_search.fit(X=designmtx, y=labels, groups=group_assignments, callback=on_step)
        except Exception as e:
            logger.info("Exception occurred:", str(e))
        finally:
            # Print or process intermediate results even if an error occurs
            logger.info("Intermediate Results:")
            for i, (scores, params) in enumerate(intermediate_results):
                logger.info(f"Iteration {i + 1}: Scores - {scores}, Params - {params}")
    else:
        _ = param_search.fit(X=designmtx, y=labels, groups=group_assignments)

    ## Saving and outputing results
    # n_splits is determined by the data for hold out one group cross validation
    n_splits = param_search.get_params()['cv'].get_n_splits()
    print(f"In main: estimator_name: {estimator_name}")
    results_df = compute_analyse_and_store_results(
        param_search, subdir, estimator_name, data_format, held_out,
        is_balanced, scoring, n_splits, feature_types, window_size, window_step,
        n_windows, search_type)

    # outputting for screen, maybe better intergrated into
    # output_model_best_to_screen or similar
    label_cols = featured_config['WINDOWED']['label_cols']
    feature_types = list(feature_set)
    feature_types.sort()
    print(f"# Feature Set:\n{feature_types}")
    derived_feature_names = []
    derived_feature_types = set([])
    for f in featured_df.columns:
        if f in label_cols:
            continue
        elif (f[-1] == ']'):
            if (f[:-1].rstrip('0123456789')[-1] == '['):
                f = f[:-1].rstrip('0123456789')[:-1]
        else:
            f = f.rstrip('0123456789')
        for type_ in feature_set:
            if f.startswith(type_):
                derived_feature_types.add(f)
                break
    derived_feature_types = list(derived_feature_types)
    derived_feature_types.sort()
    output = ';'.join(derived_feature_types)
    print(f"Derived Feature Types:\n{output}")
    print()
    print("Data properties:")
    print(
        f"\tsample rate {sample_rate}")
    print(
        f"\twindow_size {window_size}")
    print(
        f"\twindow step {window_step}")
    print(
        f"\ttotal window counts {int(n_windows)}")
    print(
        f"\tcondition window counts:\n\t\t{list(zip(label_mapping,condition_counts))}")
    print(
        f"\tparticipant window counts:\n\t\t{list(zip(participant_list,participant_counts))}")
    
    output_model_best_from_results(results_df)
    best_score, best_score_std = get_model_best_from_results(results_df, estimator_name)
    print(f"Headline performance: best test score: {best_score} [std {best_score_std}]")
    output_model_best_from_results(results_df)

def compute_analyse_and_store_results(param_search, subdir,
        estimator_name, data_format, held_out, is_balanced, scoring, n_splits, feature_types,
        window_size, window_step, n_windows, search_type):
    print(f"In compute_analyse_and_store_results: estimator_name: {estimator_name}")
    results_df = pd.DataFrame(param_search.cv_results_)
    i = 0
    results_df.insert(i, 'subdir', subdir)
    i +=1
    results_df.insert(i, 'estimator', estimator_name)
    i +=1
    results_df.insert(i, 'data format', data_format)
    i +=1
    results_df.insert(i, 'held out', held_out)
    i +=1
    results_df.insert(i, 'balanced', is_balanced)
    i +=1
    results_df.insert(i, 'scoring', scoring)
    i +=1
    results_df.insert(i, 'n_splits', n_splits)
    i +=1
    results_df.insert(i, 'feature set', str(feature_types))
    i +=1
    results_df.insert(i, 'window size', window_size)
    i +=1
    results_df.insert(i, 'window step', window_step)
    i +=1
    results_df.insert(i, 'n_windows', n_windows)
    results_fname = f'{search_type}_{subdir}_{estimator_name}'
    logger.info(f"Saving to {results_fname}")
    save_results_df_to_file(results_df, results_fname, subdir)

    logger.info("Plotting proportion of fits above various thresholds...")
    # plot the proportion of results with performance above threshold
    thresholds = np.linspace(0,0.3,51)
    N = len(results_df)
    props = np.empty(thresholds.size)
    for t, thresh in enumerate(thresholds):
        count = np.sum(results_df['mean_test_score'] > thresh)
        props[t] = count/N
    plt.plot(thresholds, props)
    plt.xlabel("mean test score")
    plt.ylabel("proportion greater than")
    thresholds_fname = f'{search_type}_{estimator_name}_thresholds.png'
#    logger.info(f"Saving thresholds plot to {thresholds_fname}")
#    plt.savefig(thresholds_fname)
    save_results_plot_to_file(plt.gcf(), thresholds_fname, subdir)
    return results_df


def create_callback_and_storage(param_search):
    intermediate_results = []
    # Define a custom callback function to store intermediate results
    def on_step(optim_result):
        # Store the current state of the optimization process
        intermediate_results.append(
            (optim_result.func_vals[-1], optim_result.x_iters[-1]))

        # Print out the best score and best parameters found so far
        best_score = -optim_result.fun
        logger.info("Best score: %s" % best_score)
        logger.info("Best parameters: %s" % optim_result.x)
    return on_step, intermediate_results

def create_parser():
    import argparse
    description= """
        Loads study data, then windows and (typically) partitions it according
        to the configuration parameters then saves down to file."""
    parser = argparse.ArgumentParser(
        description=description,
        epilog='See git repository readme for more details.')
    parser.add_argument(
        '-e', '--estimator', default='TwoHiddenLayerClassifier',
        help="""Name of estimator to use.""")
    parser.add_argument(
        '--subdir', default='dreem_4secs',
        help="""Data subdirectory to use.""")
    parser.add_argument(
        '--use-unsafe-features', action='store_true',
        help="""Use all features in data including non-ideal features""")
    parser.add_argument(
        '--held-out', default='participant',
        help="""Data subdirectory to use.""")
    parser.add_argument(
        '--keep-classes-unbalanced', action='store_true',
        help="""Do not apply class balancing.""")
    parser.add_argument(
        '--standardise-data', default='store_true',
        help="""Standardise data before training.""")
    parser.add_argument(
        '--max-iter-opt', default=200, type=int,
        help="""Maximum number of optimisation iterations.""")
    parser.add_argument(
        '--n-iter', default=50, type=int,
        help="""Hyperparameter search iterations.""")
    parser.add_argument(
        '--retrieve-old-search', default="store_true",
        help="""Load old hyperparameter search results.""")
    parser.add_argument(
        '--random-state',
        help="""Set the random state of random number generator.""")
    parser.add_argument(
        '--use-callback', action='store_true',
        help="""Use a callback function to output intermediate results.""")
    parser.add_argument(
        '--hyperparameter-overrides',
        help="""Override some hyperparameter choices.""")
    parser.add_argument(
        '--hyperparameter-excludes',
        help="""Exclude some hyperparameters from the parameter search.""")
    parser.add_argument(
        '--search-type', default='bayesian_optimization',
        help="""What hyperparameter search type to perform.""")
    parser.add_argument(
        '--scoring', default='f1_macro',
        help="""What score to use when optimising hyperparameters.""")
            
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

