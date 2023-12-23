import numpy as np

def get_group_assignments(df, held_out='participant'):
    groups = np.unique(df[held_out])
    n_groups = len(groups)
    group_assignments = np.empty(len(df), dtype=int)
    for g, group in enumerate(groups):
        group_assignments[df['participant']==group] = g
    return held_out, groups, group_assignments
    
    
    

