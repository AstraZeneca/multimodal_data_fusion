import pandas as pd
import os
from typing import Tuple


def load_data(datapath: str) -> pd.DataFrame:
    """
    Parameters
    ----------
    datapath: STRING, required
        Path to data
    
    Returns
    -------
    Dataframe
    """
    if ".csv" in datapath:
        data = pd.read_csv(datapath)
    elif ".parquet" in datapath:
        data = pd.read_parquet(datapath, engine='pyarrow')
    else:
        raise ValueError(f"Unsupported file format: {datapath}")
    # Drop Unnamed columns
    data = data.loc[:, ~data.columns.str.startswith('Unnamed')]
    # data = data.drop('Unnamed: 0', axis=1)
    return data


def split_data(data: pd.DataFrame, cohorts: pd.DataFrame, attr: list) -> pd.DataFrame:
    """
    Parameters
    ----------
    data: DATAFRAME, required
        data to be filtered
    cohorts: DATAFRAME, required
        cohorts with split information
    attr: LIST, required
        Atrribute/s to split data upon
    Returns
    -------
    Dataframe split based on condition
    """
    df = pd.DataFrame()
    if len(attr) > 1:
        # Combine the attributes into a single column
        # https://stackoverflow.com/questions/33282119/pandas-filter-dataframe-by-another-dataframe-by-row-elements/33282617
        cohorts['UID'] = cohorts[attr].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        data['UID'] = data[attr].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)  
        df = data[data.UID.isin(cohorts.UID)]
    else:
        uid = attr[0]
        df = data[data[uid].isin(cohorts[uid])]
    return df


def drop_missingdata(data: pd.DataFrame, cols: dict, mode: list) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Parameters
    ----------
    data: DATAFRAME, required
        data to be handled
    cols: DICTIONARY, required
        dictionary of list of columns to use for finding missing data
    mode: LIST, required
        handling mode for each modality. Options being drop rows with missing data, impute with 0 and impute with median
    Returns
    -------
    Dataframe with no missing data
    """
    df = data.copy(deep=True)
    i = 0
    for key, columns in cols.items():
        if mode[i] == "drop":
            df = data.dropna(axis=0, subset=columns, inplace=False)
        i += 1
    return df


def handle_missingdata(data: pd.DataFrame, cols: dict, mode: list, column_vals: dict = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Parameters
    ----------
    data: DATAFRAME, required
        data to be handled
    cols: DICTIONARY, required
        dictionary of list of columns to use for finding missing data
    mode: LIST, required
        handling mode for each modality. Options being drop rows with missing data, impute with 0 and impute with median
    column_vals: DICTIONARY, optional
        PANDAS SERIES of values to use to impute if impute with mean/median [when data is val/test]
    Returns
    -------
    Dataframe with no missing data
    """
    df = data.copy(deep=True)
    i = 0
    if column_vals is None:
        column_vals = {}
    for key, columns in cols.items():
        if key not in column_vals:
            column_vals[key] = []
        if mode[i] == "drop":
            i += 1
            continue
        elif mode[i] == "zeroes":
            df[columns].fillna(value=0, inplace=True)
        elif mode[i] == "median":
            if len(column_vals[key]) == 0:
                column_vals[key] = df[columns].median()
            df[columns] = df[columns].fillna(column_vals[key])
        elif mode[i] == "mean":
            if len(column_vals[key]) == 0:
                column_vals[key] = df[columns].mean()
            df[columns] = df[columns].fillna(column_vals[key])
        elif mode[i] == "mode":
            if len(column_vals[key]) == 0:
                column_vals[key] = df[columns].mode().iloc[0]
            df[columns] = df[columns].fillna(column_vals[key])
        elif mode[i] is None:
            i += 1
            continue
        else:
            raise ValueError("Defined mode does not exist")
        i += 1
    return df, column_vals