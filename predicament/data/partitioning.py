# -*- coding: utf-8 -*-
"""
"""

import os
import numpy as np

from predicament.data.windowed import merge_condition_data
from predicament.data.windowed import merge_labelled_data_by_key

def between_subject_cv_partition_from_nested(nested_data):
    """
    Takes nested dictionary of participant->condition->data
    and iterative yields folds of data
    """
    merged_data, label_mapping = merge_condition_data(nested_data)
    for fold in between_subject_cv_partition(merged_data):
        train_data, train_labels, test_data, test_labels, held_out_ID = fold
        yield train_data, train_labels, test_data, test_labels, held_out_ID, label_mapping
        
    
def between_subject_cv_partition(labelled_data_by_participant):
    #
    participants = list(labelled_data_by_participant.keys())
    for held_out_ID in participants:
        merge_keys = [ part_ID \
            for part_ID in participants \
                if part_ID != held_out_ID]
        train_data, train_labels = merge_labelled_data_by_key(
            labelled_data_by_participant, merge_keys=merge_keys)
        test_data, test_labels = labelled_data_by_participant[held_out_ID]
        yield train_data, train_labels, test_data, test_labels, held_out_ID


if __name__ == '__main__':
    pass
