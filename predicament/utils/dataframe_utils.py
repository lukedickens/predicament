import numpy as np
import pandas as pd

def drop_inf_cols(df):
    removed_columns = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if np.any(np.isinf(df[col])):
            removed_columns.append(col)
            del df[col]
    return df, removed_columns

def drop_nan_cols(df):
    removed_columns = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if np.any(np.isnan(df[col])):
            removed_columns.append(col)
            del df[col]
    return df, removed_columns


def compute_and_add_aggregating_column(df, agg_mode='abs_max'):
    if agg_mode == 'abs_max':
        df[agg_mode] = np.nanmax(np.abs(all_corrs),axis=1)
    elif agg_mode == 'max':
        df[agg_mode] = np.nanmax(all_corrs,axis=1)
    elif agg_mode == 'min':
        df[agg_mode] = np.nanmin(all_corrs,axis=1)
    else:
        raise ValueError(
            f"Unrecognised aggregation operation {abs_aggregate_with}")
    return df


def aggregate_rows_by_grouped_index(
        df, index_groups, agg_element='row', agg_mode = 'abs_max'):
    feature_filters = {
        k: [i in group for i in df.index ] 
            for k, group in index_groups.items() }
    agg_values = np.empty((len(feature_filters), df.shape[1]))
    for r, key in enumerate(feature_filters.keys()):
        filter_ = feature_filters[key]
        tmp_df  = df.loc[filter_,:]
        if agg_element == 'row':
            if agg_mode == 'abs_max':
                try:
                    row_maxes = np.max(np.abs(tmp_df.values), axis=1)
                    best_row = np.nanargmax(row_maxes)
                except:
                    best_row = 0
            else:
                raise ValueError(
                    f"Unrecognised agg_mode = {agg_mode}")
#             print(f"best_row= {best_row}")
#             print(f"tmp_df.values.shape = {tmp_df.values.shape}")
#             print(f"agg_values = {agg_values}")
            agg_values[r,:] = tmp_df.values[best_row,:]        
        elif agg_element == 'cell':
            if agg_mode == 'abs_max':
                try:
                    best_row_per_col = np.nanargmax(np.abs(tmp_df.values), axis=0)
                except:
                    print(f"Failed to compute nanargmax for {key}")
                    best_row_per_col = np.nan(tmp_df.values.shape[1])
            else:
                raise ValueError(
                    f"Unrecognised agg_mode = {agg_mode}")
            agg_values[r,:] = tmp_df.values[best_row_per_col, np.arange(best_row_per_col.size)]
        else:
            raise ValueError(f"Unrecognised agg_element = {agg_element}")

    agg_df = pd.DataFrame(
        data=agg_values, index=list(feature_filters.keys()), columns=df.columns)
    return agg_df


def split_column_into_independent_columns(
        df, value_mapping=None, col_to_split='condition'):
    if value_mapping is None:
        value_mapping = {
            v:v for v in np.unique(df[col_to_split])}
    for v, value in value_mapping.items():
        indep_condition = str(value)
        df[indep_condition] = df[col_to_split] == v
    return df

