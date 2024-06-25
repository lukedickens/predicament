import numpy as np 
import pandas as pd 
import seaborn as sns
import scipy.stats
import math

import matplotlib.pyplot as plt
import os

from predicament.utils.dataframe_utils import compute_and_add_aggregating_column
from predicament.utils.dataframe_utils import aggregate_rows_by_grouped_index
from predicament.utils.dataframe_utils import split_column_into_independent_columns

from predicament.data.features import construct_feature_groups


def get_feature_feature_correlations(df, feature_names):
    corr_mtx = df[feature_names].corr()
    return corr_mtx

def plot_feature_feature_correlations(
        corr_mtx, fstem, results_dir, high_corr_threshold=0.85):
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_mtx, cmap='coolwarm', annot=False)
    plt.title(f"Feature Correlations {fstem}")
    fname = f"correlations_f2f_{fstem}.png"
    fpath = os.path.join(results_dir, fname)
    print(f"saving feature correlations heatmap to:\n\t{fpath}")
    plt.savefig(fpath)
    
    plt.figure(figsize=(10,8))
    sns.heatmap(
        np.abs(corr_mtx) >= high_corr_threshold, cmap='coolwarm', annot=False)
    plt.title(f"Highly Correlated features")
    fname = f"highly_correlated_f2f_{fstem}.png"
    fpath = os.path.join(results_dir, fname)
    print(f"saving higly correlated features heatmap to:\n\t{fpath}")
    plt.savefig(fpath)


def scatter_plot_features(
        df, feature_names, results_dir, fstem, with_jitter=False, jitter=1e-5):
    feature_data = df[feature_names].to_numpy()

    if with_jitter:
        col_means = np.mean(feature_data, axis=0)
        noise = (np.random.random(feature_data.shape)*2-1)*jitter*col_means.reshape((1,-1))
        feature_data = feature_data + noise
    N,F = feature_data.shape
    fig, axgrid = plt.subplots(F,F, figsize=(10,8))
    for i in range(F):
        for j in range(F):
            if i != j:
                axgrid[i,j].scatter(feature_data[:,j],feature_data[:,i], marker='.', s=0.5)
            else:
                try:
                    axgrid[i,i].hist(feature_data[:,i])
                except:
                    print(f"Failed to histogram {i}th feature {feature_names[i]}")
                    inf_val_positions = ~np.isfinite(feature_data[:,i])
                    print(f"inf_val_positions = {inf_val_positions}")
                    axgrid[i,i].hist(feature_data[~inf_val_positions,i])
            axgrid[i,j].set_xticks([])
            axgrid[i,j].set_yticks([])

            if j == 0:
                axgrid[i,j].set_ylabel(feature_names[i], rotation='horizontal', ha='right')
            if i == F-1:
                axgrid[i,j].set_xlabel(feature_names[j], rotation='vertical')
    plt.tight_layout()
    fname = f"scatter_plot_features_{fstem}.png"
    fpath = os.path.join(results_dir, fname)
    print(f"Saving scatter plot features to {fpath}")
    plt.savefig(fpath)

def analyse_feature_vs_comparator(
        featured_df, config, results_dir,
        comparator = "condition",
        group_by='channel',
        add_agg_column = False,
        aggregate_rows = True,
        remove_rows_with_nans = False,
        row_agg_element='cell',
        row_agg_mode = 'abs_max',
        annot=False, # whether to give r value in cell
        row_height = 0.3,
        col_width = 0.3,
        granularity = 40):
    """
    parameters
    ----------
    comparator: "condition" or "participant"
    aggregate_rows[bool] : should rows be aggregated by group
    group_by: 'channel' or 'type', how to aggregate rows
    add_agg_column: should a column with aggregated values be inlcuded
    remove_rows_with_nans [bool]
    row_agg_element: row aggregation done by 'row' or 'cell'
    row_agg_mode: operation to aggregate rows 'abs_max', 'max' or 'min'
    annot [bool]: whether to give r value in cell
    row_height: approx for plot 0.3
    col_width: approx for plot 0.3
    granularity: granularity of the colourbar per unit 40
    """
    feature_names = config['FEATURED']['feature_names']
    label_mapping = config['LOAD']['label_mapping']
    if comparator == 'condition':
        value_mapping = dict(enumerate(label_mapping))
        values = list(value_mapping.values())
    elif comparator == 'participant':
        participants = np.unique(featured_df['participant'])
        value_mapping = dict(zip(participants,participants))
        values = list(value_mapping.values())
    else:
        raise ValueError(f"Unrecognised comparator {comparator}")
    featured_df = split_column_into_independent_columns(
        featured_df, value_mapping, col_to_split=comparator)
    correlation_mtx = featured_df[values+feature_names].corr()
    correlation_mtx = correlation_mtx.loc[feature_names,values]

    if add_agg_column:
        correlation_mtx = compute_and_add_aggregating_column(
            correlation_mtx)
    labels = None
    if aggregate_rows:
        if group_by == 'both':
            feature_groups1 = construct_feature_groups(
                config, 'channel')
            correlation_mtx1 = aggregate_rows_by_grouped_index(
                correlation_mtx, feature_groups1,
                agg_element=row_agg_element,
                agg_mode=row_agg_mode )
            feature_groups2 = construct_feature_groups(
                config, 'type')
            correlation_mtx2 = aggregate_rows_by_grouped_index(
                correlation_mtx, feature_groups2,
                agg_element=row_agg_element,
                agg_mode=row_agg_mode )
            gap_row = pd.DataFrame(
                [[np.nan] * correlation_mtx2.shape[1]],
                columns=correlation_mtx2.columns, index=[""])
            correlation_mtx =  pd.concat(
                [correlation_mtx1, gap_row, correlation_mtx2])
#            labels = correlation_mtx1.tolist() + [""] + correlation_mtx2.tolist()
            fstem = f"feature_vs_{comparator}_both_correlations"
        else:
            feature_groups = construct_feature_groups(
                config, group_by)
            correlation_mtx = aggregate_rows_by_grouped_index(
                correlation_mtx, feature_groups,
                agg_element=row_agg_element,
                agg_mode=row_agg_mode )
            fstem = f"feature_vs_{comparator}_{group_by}_correlations"
    else:
        fstem = f"feature_vs_{comparator}_correlations"
#    if labels is None:
#        labels = correlation_mtx.tolist()


    csvfname = f"{fstem}.csv"
    pngfname = f"{fstem}.png"
    csvfpath = os.path.join(results_dir, csvfname)
    pngfpath = os.path.join(results_dir, pngfname)
    print(f"saving to {csvfpath}")
    correlation_mtx.to_csv(csvfpath)
    ## find bounds for colorbar
    ## get a good set of limits for the colorbar
    max_abs_val = np.nanmax(np.abs(correlation_mtx))
    print(f"Maximum absolute value in array is {max_abs_val}")
    max_abs_val = math.ceil(max_abs_val*granularity)/40
    # plotting heatmap
    fig_height = row_height*len(correlation_mtx.index)
    fig_width = col_width*(len(correlation_mtx.columns)+6)
    plt.figure(figsize=(8,fig_height))
    ax = sns.heatmap(correlation_mtx, cmap='coolwarm', annot=annot)

    ticklabels = [ (tick+0.5, label) \
        for tick, label in enumerate(correlation_mtx.index) if label != ""   ]
    ticks, labels = zip(*ticklabels)
    plt.yticks(ticks=ticks, labels=labels, rotation=0)
    
    ax.collections[0].set_clim(-max_abs_val,max_abs_val) 
    plt.title(f"Feature vs {comparator} correlations")
    _ = plt.xticks(rotation=90) 
    plt.tight_layout()
    print(f"saving plot to {pngfpath}")
    plt.savefig(pngfpath)
    return correlation_mtx

## Density of correlations    
def plot_density_of_correlated_features(
        df, results_dir, group_name_and_features):
    thresholds = np.linspace(0,1,21)
    for group_name, tmp_features  in  group_name_and_features:
        corr_mtx = get_feature_feature_correlations(
            df, tmp_features)
        num_more_correlated = np.empty(thresholds.size, dtype=int)
        for t,thresh in enumerate(thresholds):
        #     print(f"thresh = {thresh}")
            pairs = [(i,j) for i, j in zip(*np.where(np.abs(corr_mtx) >= thresh)) if i != j]
            num_more_correlated[t] = len(pairs)
        prop_more_correlated = num_more_correlated /num_more_correlated[0]

        plt.plot(thresholds, prop_more_correlated, label=group_name)

    plt.legend()
    plt.xlabel('threshold')
    plt.ylabel('prop more correlated')
    plt.title("Proportion of non-reflexive pairs more correlated than some threshold")
    fname = 'density_of_correlated_features.png'
    fpath = os.path.join(results_dir, fname)
    print(f"plotting density of correlated features to {fpath}")
    plt.savefig(fpath)

def pval_from_pearsonsr(rho,n):
    sigma_rho = np.sqrt((1-rho**2)/(n-2))
    t = rho/sigma_rho
    pval = scipy.stats.t.sf(np.abs(t), n-2)*2 
    return pval

