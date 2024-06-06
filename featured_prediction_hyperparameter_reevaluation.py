import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

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
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from sklearn.model_selection import RandomizedSearchCV 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from skopt import BayesSearchCV
# parameter ranges are specified by one of below
from skopt.space import Real, Categorical, Integer


from predicament.utils.file_utils import load_dataframe_and_config
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

from predicament.evaluation.hyperparameters import get_estimator
from predicament.evaluation.hyperparameters import get_param_scopes
#from predicament.evaluation.hyperparameters import get_param_search_object

from predicament.models.mlp_wrappers import ThreeHiddenLayerClassifier
    
def main(
        subdir=None, resfile=None,
        standardise_data = None,
#        held_out=None,
        keep_classes_unbalanced=False, 
#        standardise_data = None,
#        estimator=None,
#        max_iter_opt = None,
#        n_iter=50,
#        random_state = None,
        perm_scoring='accuracy',
        perm_trials=100000,
        fragile=True):

    logger.info('Running featured prediction hyperparamter revaluation with:')        
    logger.info(f'\tsubdir: {subdir}')  
    logger.info(f'\tresfile: {resfile}')
    logger.info(f'\tsubdir: {subdir}')        
    featured_df, featured_config = load_featured_dataframe_and_config(
        subdir)

    # configuration
    n_channels = featured_config['LOAD']['n_channels']
    label_mapping = featured_config['LOAD']['label_mapping']
    data_format = featured_config['LOAD']['data_format']
    channels = featured_config['LOAD']['channels']
    participant_list = featured_config['LOAD']['participant_list']
    sample_rate = featured_config['LOAD']['sample_rate']
    Fs = sample_rate
    window_size = featured_config['LOAD']['window_size']
    window_step = featured_config['LOAD']['window_step']
    time = window_size/sample_rate
    logger.info(f"sample_rate: {sample_rate}, n_samples = {window_size}, time: {time}s, n_channels: {n_channels}")
    resfname = resfile.split(os.sep)[-1]
    if resfname.startswith('bayesian_optimization'):
        reeval_path = resfile.replace('bayesian_optimization', 'reevaluation')
    else:
        raise ValueError(f"results path {resfile} has file name {resfname}, this must start with 'bayesian_optimization'")
    print(f"reeval_path = {reeval_path}")
    results_df = pd.read_csv(resfile, index_col=0, header=0)
#    param_names = [ "_".join(col.split("_")[1:]) \
#        for col in results_df.columns
#            if col.startswith("param_")]
    print(f"results_df=\n{results_df}")
    first_rank_1_row = results_df[results_df['rank_test_score'] == 1].iloc[0]
    best_params = {}
    for col in results_df.columns:
        if col.startswith("param_"):
            param_name = "_".join(col.split("_")[1:])
            print(f"Storing for param {param_name}")
            best_params[param_name] = first_rank_1_row[col]
    print(f"best_params = {best_params}")
    estimator_name = first_rank_1_row['estimator']
    res_data_format = first_rank_1_row['data format']
    res_held_out = first_rank_1_row['held out']
    res_balanced = first_rank_1_row['balanced']
    res_n_splits = first_rank_1_row['n_splits']
    res_window_size = first_rank_1_row['window size']
    res_window_step = first_rank_1_row['window step']
    res_n_windows = first_rank_1_row['n_windows']

    logger.info(f"data_format = {data_format}, res_data_format = {res_data_format}")
    logger.info(f"window_size = {window_size}, res_window_size = {res_window_size}")
    if fragile:
        if data_format != res_data_format:
            raise ValueError("Data formats don't match")
        if window_size != res_window_size:
            raise ValueError("Window sizes don't match")

    estimator = get_estimator(estimator_name)
    estimator.set_params(**best_params)
    print(f"estimator.get_params(): {estimator.get_params()}")

    ## balancing
    participant_condition_counts = get_group_label_counts(
        featured_df, 'participant', 'condition')
    # balancing data
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
    

    ## define data properties and data-split
    feature_set = json.loads(first_rank_1_row['feature set'].replace("'",'"'))
    print(f"type(feature_set) = {type(feature_set)}")
    logger.info(f"feature_set = {feature_set}")
    # extract input data
    feature_types, feature_names, designmtx = get_design_matrix(
        featured_df, feature_set)
    # extract labels
    labels = featured_df['condition'].values.astype(int)
    X = designmtx
    y = labels
    # 
    if standardise_data:
        scaler = preprocessing.StandardScaler().fit(designmtx)
        designmtx = scaler.transform(designmtx)
    #
    if first_rank_1_row['held out'] != 'participant':
        raise ValueError("Don't yet know how to deal with other ways to hold out")
    # prepare Hold one group out cross validation
    held_out, groups, group_assignments = get_group_assignments(featured_df)
    n_groups = len(groups)
    # cross validation splits    
    group_kfold = GroupKFold(n_splits=n_groups)
    
    ## reevaluate    
    # Collect metrics and confusion matrices for each fold
    n_trains = []
    n_vals = []
    accuracies = []
    macro_f1_scores = []
    micro_f1_scores = []
    confusion_matrices = []    #
    # Assuming X is your feature matrix and y is your target vector
    for train_index, test_index in group_kfold.split(X, y, groups=group_assignments):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Train the model on the training set
        estimator.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = estimator.predict(X_test)
        
        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        micro_f1 = f1_score(y_test, y_pred, average='micro')
        cm = confusion_matrix(y_test, y_pred)        

        # Append metrics and confusion matrix to the lists
        n_trains.append(y_train.size)
        n_vals.append(y_test.size)
        accuracies.append(acc)
        macro_f1_scores.append(macro_f1)
        micro_f1_scores.append(micro_f1)
        confusion_matrices.append(cm)

    # construct reevaluation df and save to file    
    columns = [
        'estimator', 'held out', 'balanced', 'params', 'label mapping', 'fold',
        'n_train', 'n_val', 'perm_p']
    cm_cols = [ 
        'cm[%d,%d]' % (i,j) \
            for i in range(cm.shape[0]) \
                for j in range(cm.shape[1]) ]
    columns.extend(cm_cols)
    print(f"columns for dataframe {columns}")    
    print(f"len(columns) {len(columns)}")
    print(f"len(cm_cols) {len(cm_cols)}")
    # Print metrics for each fold
    lol = []
    for fold in range(len(accuracies)):
        n_train = n_trains[fold]
        n_val = n_vals[fold]
        accuracy = accuracies[fold]
        macro_f1_score = macro_f1_scores[fold]
        micro_f1_score = micro_f1_scores[fold]
        cm = confusion_matrices[fold]
        perm_p = permutation_test_confusion_matrix(
            cm, scoring=perm_scoring, trials=perm_trials)
        cmrow = [ cm[i,j] \
            for i in range(cm.shape[0]) \
                for j in range(cm.shape[1]) ]
        dfrow = [
            estimator_name,first_rank_1_row['held out'], is_balanced, 
            first_rank_1_row['params'], label_mapping,
            fold, n_train, n_val, perm_p]
        dfrow.extend(cmrow)
        print(f"len(dfrow) {len(dfrow)}")
        print(f"len(cmrow) {len(cmrow)}")
        lol.append(dfrow)
        print(f"Fold {fold} Accuracy: {accuracy}")
        print(f"Fold {fold} Macro F1-Score: {macro_f1_score}")
        print(f"Fold {fold} Micro F1-Score: {micro_f1_score}")
        print(f"Fold {fold} Confusion Matrix:\n{cm}")
        print(f"perm_p = {perm_p}")
        print()
    # Calculate and print aggregate metrics
    mean_accuracy = np.mean(accuracies)
    mean_macro_f1 = np.mean(macro_f1_scores)
    mean_micro_f1 = np.mean(micro_f1_scores)
    print(f"Mean Accuracy: {mean_accuracy}")
    print(f"Mean Macro F1-Score: {mean_macro_f1}")
    print(f"Mean Micro F1-Score: {mean_micro_f1}")        
    reeval_df = pd.DataFrame(lol, columns=columns)
    print(f"Saving reevaluation to file {reeval_path}")
    reeval_df.to_csv(reeval_path)


def permutation_test_confusion_matrix(cm, scoring='accuracy', trials=10000):
    score = score_from_cm(cm, scoring)
    beat_count = 0
    for i in range(trials):
        permutation = permute_cm(cm)
        sample_score = score_from_cm(permutation, scoring)
        if sample_score > score:
            beat_count += 1
    return beat_count/trials

def permute_cm(cm):
    pvals = cm.sum(axis=0)/cm.sum()
    sample = np.array(
        [ np.random.multinomial(n,pvals) for n in cm.sum(axis=1) ])
    return sample

def score_from_cm(cm, scoring):
    if scoring == 'accuracy':
        return np.sum(cm.diagonal())/cm.sum()
    else:
        raise ValueError(f"Unrecognised scoring {scoring}")

def create_parser():
    import argparse
    description= """
        Loads study data, then windows and (typically) partitions it according
        to the configuration parameters then saves down to file."""
    parser = argparse.ArgumentParser(
        description=description,
        epilog='See git repository readme for more details.')
#    parser.add_argument(
#        '-e', '--estimator', default='TwoHiddenLayerClassifier',
#        help="""Name of estimator to use.""")
    parser.add_argument(
        '--subdir', default='dreem_4secs',
        help="""Data subdirectory to use.""")
    parser.add_argument(
        '--resfile',
        help="""Path to results csv file""")
#    parser.add_argument(
#        '--use-unsafe-features', action='store_true',
#        help="""Use all features in data including non-ideal features""")
#    parser.add_argument(
#        '--held-out', default='participant',
#        help="""Data subdirectory to use.""")
    parser.add_argument(
        '--keep-classes-unbalanced', action='store_true',
        help="""Do not apply class balancing.""")
    parser.add_argument(
        '--standardise-data', default='store_true',
        help="""Standardise data before training.""")
#    parser.add_argument(
#        '--max-iter-opt', default=200, type=int,
#        help="""Maximum number of optimisation iterations.""")
#    parser.add_argument(
#        '--n-iter', default=50, type=int,
#        help="""Hyperparameter search iterations.""")
#    parser.add_argument(
#        '--retrieve-old-search', default="store_true",
#        help="""Load old hyperparameter search results.""")
#    parser.add_argument(
#        '--random-state',
#        help="""Set the random state of random number generator.""")
#    parser.add_argument(
#        '--use-callback', action='store_true',
#        help="""Use a callback function to output intermediate results.""")
#    parser.add_argument(
#        '--hyperparameter-overrides',
#        help="""Override some hyperparameter choices.""")
#    parser.add_argument(
#        '--hyperparameter-excludes',
#        help="""Exclude some hyperparameters from the parameter search.""")
#    parser.add_argument(
#        '--search-type', default='bayesian_optimization',
#        help="""What hyperparameter search type to perform.""")
            
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

