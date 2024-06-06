import numpy as np
import pandas as pd
import os


from predicament.utils.config import RESULTS_BASE_PATH

def get_model_best_from_results(df, model):
    best_score = df[df['estimator'] == model]['mean_test_score'].max()
    filter_ = (df['estimator'] == model) \
        & (df['mean_test_score']==best_score)
    best_score_std = df[filter_]['std_test_score'].to_numpy()[0]
    return best_score, best_score_std

def output_model_best_from_results(df):
    for model in np.unique(df['estimator']):
#        best_score = df[df['estimator'] == model]['mean_test_score'].max()
#        best_score_std = df[(df['estimator'] == model) & (df['mean_test_score']==best_score)]['std_test_score'].to_numpy()[0]
        best_score, best_score_std = get_model_best_from_results(df, model)
        print(f"{model}")
        print(f"\tmax_test_score= {best_score}, max_std_test_score= {best_score_std}")
        d = df[df['mean_test_score'] == best_score]['params']
        for k,v in d.items():
            model_best_params = v
            print(f"best params: {v}")
            print(';'.join([k for k in model_best_params.keys()]))
            print(';'.join([str(v) for v in model_best_params.values()]))
        print()
        
        
def get_results_dir(subdir):
    return os.path.join(RESULTS_BASE_PATH, subdir)
        
def save_results_df_to_file(
        df, shortname, subdir, timestamp=True):
    import datetime
    nowstr = datetime.datetime.now().replace(microsecond=0).isoformat()
    if timestamp:
        name = shortname + '_' + nowstr
    else:
        name = shortname
    fname = f'{name}.csv'
    results_dir = get_results_dir(subdir)
    fpath = os.path.join(results_dir, fname)
    print(f"saving to {fpath}")
    df.to_csv(fpath)
    
def save_results_plot_to_file(
        fig, fname, subdir):
    results_dir = get_results_dir(subdir)
    fpath = os.path.join(results_dir, fname)
    print(f"saving to {fpath}")
    fig.savefig(fpath)


def test():
    pass

