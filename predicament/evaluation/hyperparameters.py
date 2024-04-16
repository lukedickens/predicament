import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from predicament.models.mlp_wrappers import ThreeHiddenLayerClassifier


# parameter ranges are specified by one of below
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer


def get_bayesopt_search_spaces(estimator):
    search_spaces = BAYESOPT_SEARCH_SPACES[type(estimator)]
    return search_spaces
    
def get_randsearch_distributions(estimator):
    param_distributions = RANDSEARCH_DISTRIBUTIONS[type(estimator)]
    return param_distributions

def get_param_scopes(search_type, estimator, excludes=None, **overrides):
    if search_type == 'random_search':
        param_scopes = get_randsearch_distributions(estimator)
    elif search_type == 'bayesian_optimization':
            param_scopes = get_bayesopt_search_spaces(estimator)
    else:
        raise ValueError(f"Unrecognised search_type: {search_type}")
    if not excludes is None:
        for paramname in excludes:
            param_scopes.pop(paramname)
    param_scopes.update(overrides)
    return param_scopes
    
def get_param_search_object(
        search_type, estimator, param_scopes=None,
        excludes=None, overrides=None, **kwargs):
    """
    returns - parameter search object,
        for random search this is over param_distributions
        for bayesian optimization this over search_spaces.
    """
    if overrides is None:
        overrides = dict()
    if param_scopes is None:
        param_scopes = get_randsearch_distributions(
            estimator, excludes=excludes, **overrides)
    if search_type == 'random_search':
        # Random search of parameters, using 3 fold cross validation, 
        # search across 100 different combinations, and use all available cores
        param_search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_scopes,
            **kwargs)
        return param_search
    elif search_type == 'bayesian_optimization':
        param_search = BayesSearchCV(
            estimator=estimator,
            search_spaces=param_scopes,
            **kwargs)
        return param_search
    else:
        raise ValueError(f"Unrecognised search_type: {search_type}")


BAYESOPT_SEARCH_SPACES = {}
# SVC
BAYESOPT_SEARCH_SPACES[type(SVC())] = dict(
        # regularization - soft-margin
        C = Real(1e-1, 1e4, prior='log-uniform'),
        # Kernel
        kernel = Categorical(['rbf', 'sigmoid']),
        # Kernel scale parameter
        gamma = Real(1e-9, 1, prior='log-uniform'),
        # Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
        coef0 = Real(-1, 1),
        # Whether to use the shrinking heuristic. See the User Guide.
        shrinking = Categorical([True, False])
        ## fixed default parameters
        # probability = False
        # tol = 1e-3
        # class_weight = None
        # Create the random grid
    )
# Random Forest Classifer
BAYESOPT_SEARCH_SPACES[type(RandomForestClassifier())] = dict(
        # Number of trees in random forest
        n_estimators = Integer(10,1000, prior='log-uniform'),
        # Number of features to consider at every split
        max_features = Categorical(['log2', 'sqrt']),
        # function to measure the quality of a split
        criterion = Categorical(['gini', 'entropy', 'log_loss']),
        # Maximum number of levels in tree
        max_depth = Integer(1, 50,  prior='log-uniform'),
        # Minimum number of samples required at each leaf node
        min_samples_leaf = Real(1e-4, 1e-2, prior='log-uniform'),
        # Method of selecting samples for training each tree
        bootstrap = Categorical([True, False])
    )
# Baseline MLPClassifier, hidden_layer_sizes not supported
BAYESOPT_SEARCH_SPACES[type(MLPClassifier())] =  {
#        'hidden_layer_sizes': Categorical([ (n,) for n in range(10,200,10)]),
        'activation': Categorical(['tanh', 'relu']),
        'solver': Categorical(['sgd', 'adam']),
        'alpha': Real(1e-6, 1e-2, prior='log-uniform'),
        'learning_rate': Categorical(['constant','adaptive']), # ,'invscaling'
        'learning_rate_init': Real(1e-6, 1e+1, prior='log-uniform'),
     }

# up-to three layer MLP
BAYESOPT_SEARCH_SPACES[type(ThreeHiddenLayerClassifier())] =  {
        'layer1': Integer(10, 100),
        'layer2': Integer(10, 100),
        'layer3': Integer(10, 100),
        'activation': Categorical(['tanh', 'relu']),
        'solver': Categorical(['sgd', 'adam']),
        'alpha': Real(1e-6, 1e-2, prior='log-uniform'),
        'learning_rate': Categorical(['constant','adaptive']), # ,'invscaling'
        'learning_rate_init': Real(1e-6, 1e+1, prior='log-uniform'),
     }


BAYESOPT_SEARCH_SPACES[type(GradientBoostingClassifier())] =  {
    'learning_rate': Real(0.01, 1.0, prior='log-uniform'),
    'n_estimators': Integer(50, 500),
    'max_depth': Integer(3, 10),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 20),
    'subsample': Real(0.5, 1.0, prior='uniform'),
    # alternative for max features
    #'max_features': Categorical(['sqrt', 'log2', None]),
    'max_features': Real(1e-3, 1.0, prior='log-uniform'),
    # exponential loss only good for binary
#    'loss': Categorical(['log_loss', 'exponential']),
}

RANDSEARCH_DISTRIBUTIONS  = {}
RANDSEARCH_DISTRIBUTIONS[type(SVC())] = dict(
        # regularization - soft-margin
        C = np.logspace(-1, 4, 9),
        # Kernel
        kernel = ['rbf', 'sigmoid'],
        # gamma is kernel scale parameter - doesn't apply to linear kernel
        gamma = np.logspace(-1,1,11),
        # coef0 is an independent term in kernel function.
        # It is only significant in ‘poly’ and ‘sigmoid’.
        coef0 = np.linspace(-1, 1, 11),
        # Whether to use the shrinking heuristic. See the User Guide.
        shrinking = [True, False]
        ## fixed default parameters
        # probability = False
        # tol = 1e-3
        # class_weight = None
        # Create the random grid
    )
RANDSEARCH_DISTRIBUTIONS[type(RandomForestClassifier())] = dict(
    # Number of trees in random forest
    n_estimators = np.linspace(start=10, stop =200, num=10, dtype=int),
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt'],
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)] + [None],
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10],
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4],
    # Method of selecting samples for training each tree
    bootstrap = [True, False])



