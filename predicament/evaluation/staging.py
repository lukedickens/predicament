import numpy as np
import pandas as pd
import re

from predicament.data.features import SUPPORTED_FEATURE_GROUP

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from predicament.models.mlp_wrappers import ThreeHiddenLayerClassifier

SKLEARN_MODEL_CONSTRUCTORS = dict(
    RandomForestClassifier=RandomForestClassifier,
    MLPClassifier=MLPClassifier,
    SVC=SVC,
    ThreeHiddenLayerClassifier=ThreeHiddenLayerClassifier)


def get_design_matrix(data_df, feature_types=None, fragile=True):
    # you can choose a subset of the feature types to use
    if feature_types is None:
        feature_types = SUPPORTED_FEATURE_GROUP
    # this constructs a list of the featured data columns based on the above feature types
    feature_type_pairs = [ (f,re.split('[^a-zA-Z]', f)[0]) for f in data_df.columns ]
#    print(f"feature_type_pairs = {feature_type_pairs}")
    feature_names = [ f for f,t in feature_type_pairs if t in feature_types ]
#    print(f"feature_names = {feature_names}")
    if fragile:
        # check feature_type all in design matrix
        recovered_feature_types = [ t for f,t in feature_type_pairs ]
        recovered_feature_types = list(np.unique(recovered_feature_types))
#        print(f"feature_type_pairs = {feature_type_pairs}")
        for feature_type in feature_types:
            if not feature_type in recovered_feature_types:
                raise ValueError(
                    f"You have requested feature type {feature_type} but it is not found in the data.")
    designmtx = data_df[feature_names].to_numpy()
    return feature_types, feature_names, designmtx
    
def get_estimator(model_name, **modelargs):
    return SKLEARN_MODEL_CONSTRUCTORS[model_name](**model_args)
