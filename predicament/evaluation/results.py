import numpy as np
import pandas as pd
import os


from predicament.utils.config import RESULTS_BASE_PATH

def output_model_best_from_results(df):
    for model in np.unique(df['model']):
        model_max_test_score = df[df['model'] == model]['mean_test_score'].max()
        model_max_std_test_score = df[(df['model'] == model) & (df['mean_test_score']==model_max_test_score)]['std_test_score'].to_numpy()[0]
        print(f"{model}: max_test_score= {model_max_test_score}, max_std_test_score= {model_max_std_test_score}")
        d = df[df['mean_test_score'] == model_max_test_score]['params']
        for k,v in d.items():
            model_best_params = v
            print(f"best params: {v}")
        print()
        
        
        
def save_results_df_to_file(df, shortname, timestamp=True, results_dir=RESULTS_BASE_PATH):
    import datetime
    nowstr = datetime.datetime.now().replace(microsecond=0).isoformat()
    if timestamp:
        name = shortname + '_' + nowstr
    else:
        name = shortname
    fname = f'{name}.csv'
    fpath = os.path.join(results_dir, fname)
    print(f"saving to {fpath}")
    df.to_csv(fpath)
    
def test():
    pass

