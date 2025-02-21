"""
File parser file. Contains file input and output functions
Author: Hamish Beath
Date: 25/09/2024

"""
import pandas as pd
import pyam 

# filepaths
INPUT_FP = 'src/data/input/'
PROCESSED_FP = 'src/data/processed/'
OUTPUT_FP = 'src/data/output/'
DATABASE_FP = 'src/data/database/'


# # set up the meta data for the global database
# META_DATA = pd.read_csv(INPUT_FP + 'meta_data.csv')
# # rename columns to match pyam requirements
# META_DATA = META_DATA.rename(columns={'Model': 'model', 'Scenario': 'scenario'})

# # set index to model and scenario
# META_DATA = META_DATA.set_index(['model', 'scenario'])

# # read in the global database, add meta data and set index to model and scenario
# GLOBAL_DATABASE_PYDF = pyam.IamDataFrame(data=INPUT_FP + 'AR6_Scenarios_Database_World_v1.1.csv', 
#                                          meta=META_DATA, index=['model', 'scenario'])

# REGIONAL_DATABASE_PYDF = pyam.IamDataFrame(data=INPUT_FP + 'AR6_Scenarios_Database_R10_regions_v1.1.csv',
#                                              meta=META_DATA, index=['model', 'scenario'])

def read_csv(file_name):
    """
    Function to read in a csv file

    Inputs:
    - file_name (str): name of the file to read in

    Outputs:
    - dataframe
    """
    df = pd.read_csv(file_name + '.csv')
    return df


def save_dataframe_csv(df, file_name):

    """
    Function to save a dataframe to a csv file

    Inputs:
    - df (dataframe): dataframe to save
    - file_name (str): name of the file to save to
    """
    df.to_csv(file_name + '.csv', index=False)


def read_pyam_df(file_name):
    """
    Function to read in a pyam dataframe from a csv file

    Inputs:
    - file_name (str): name of the file to read in

    Outputs:
    - dataframe
    """
    df = pyam.IamDataFrame(data=file_name + '.csv')
    return df


def read_pyam_add_metadata(file_name, meta_data) -> pyam.IamDataFrame:
    """
    Function to read in a pyam dataframe from a csv file and add all meta data

    Inputs:
    - file_name (str): name of the file to read in

    Outputs:
    - dataframe
    """
    df = pyam.IamDataFrame(data=file_name + '.csv', meta=meta_data)
    return df


def read_pyam_add_metacols(file_name, meta_data, meta_cols=list):
    """
    Function to read in a pyam dataframe from a csv file and add meta columns

    Inputs:
    - file_name (str): name of the file to read in
    - meta_cols (list): list of meta columns to add

    Outputs:
    - dataframe
    """
    # Check if meta_cols are in meta_data
    for col in meta_cols:
        if col not in meta_data.columns:
            raise KeyError(f"Meta column '{col}' not found in 'meta_data' DataFrame.")
    
    # subset the meta data
    meta = meta_data[meta_cols]
    
    # combine meta data to the pyam df
    df = pyam.IamDataFrame(data=file_name + '.csv', meta=meta)
    return df

# # read in the regional R10 database, add meta data and set index to model and scenario
# REGIONAL_DATABASE_PYDF = pyam.IamDataFrame(data=INPUT_FP + 'AR6_Scenarios_Database_R10_regions_v1.1.csv',
#                                            meta=META_DATA, index=['model', 'scenario'])

