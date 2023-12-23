import numpy as np
import pandas as pd


# balancing data
## 
def get_group_label_counts(df, group_col, label_col):
    """
    get the existing counts for each group, label pair
    """
    groups = np.unique(df[group_col])
    n_groups = groups.size
    labels = np.unique(df[label_col])
    n_labels = labels.size
    group_label_counts = np.zeros((groups.size,labels.size))

    for g, group in enumerate(groups): 
        group_df = df[df[group_col] == group]
        for l, label in enumerate(labels):
            count = len(group_df[group_df[label_col] == label])
            group_label_counts[g,l] = count
    return group_label_counts
    


def propose_balanced_group_label_counts(
        df, group_col, label_col):
    """
    propose new counts for each group, label pair such that
        * there is sufficient data to cover these
        * these are balanced across labels
        * as balanced as possible within group
    """
    group_label_counts = get_group_label_counts(df, group_col, label_col)
    n_groups, n_labels = group_label_counts.shape

    label_counts = np.sum(group_label_counts, axis=0)
    desired_label_count = np.min(label_counts)
    proposed_group_label_counts = \
        np.ones(group_label_counts.shape)*desired_label_count//n_groups
    proposed_group_label_counts = \
        np.minimum(group_label_counts, proposed_group_label_counts)
    to_allocate = desired_label_count - np.sum(proposed_group_label_counts, axis=0)
    while np.any(to_allocate >= n_groups):
        proposed_group_label_counts += (to_allocate//n_groups).reshape((1,-1))
        proposed_group_label_counts = \
            np.minimum(group_label_counts,proposed_group_label_counts)
        to_allocate = \
            desired_label_count - np.sum(proposed_group_label_counts, axis=0)
    proposed_label_counts = np.sum(proposed_group_label_counts, axis=0)
    proposed_class_balance = proposed_label_counts/np.sum(proposed_label_counts)
    while np.any(to_allocate > 0):
        filter_ = (group_label_counts > proposed_group_label_counts) \
            & (to_allocate.reshape((1,-1)) > 0)
        proposed_group_label_counts[filter_] += 1
        # recalcualte what remains to allocate
        to_allocate = desired_label_count \
            - np.sum(proposed_group_label_counts, axis=0)
    return proposed_group_label_counts.astype(int)

def subsample_proposed_group_label_counts(
        df, proposed_gl_counts, group_col, label_col):
    groups = np.unique(df[group_col])
    n_groups = groups.size
    labels = np.unique(df[label_col])
    n_labels = labels.size
    new_df = pd.DataFrame(columns=df.columns)
    for g, group in enumerate(groups):
        for l, label in enumerate(labels):
            gl_df = df[(df[group_col] == group) & (df[label_col] == label)]
            gl_sample = gl_df.sample(proposed_gl_counts[g,l], replace=False)
            new_df = pd.concat(
                (new_df, gl_sample))
    return new_df
    
    
def balance_data(df, group_col='participant', label_col='condition'):
    # balancing the data
    proposed_group_label_counts = propose_balanced_group_label_counts(
        df, group_col, label_col)
    proposed_label_counts = np.sum(proposed_group_label_counts, axis=0)
    proposed_class_balance = proposed_label_counts/np.sum(proposed_label_counts)
    df = subsample_proposed_group_label_counts(
        df, proposed_group_label_counts, group_col, label_col)
    return df

