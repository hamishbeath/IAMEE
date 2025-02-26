import numpy as np
import pyam
import pandas as pd
import country_converter as coco
import pickle as pkl
from constants import *
cc = coco.CountryConverter()



"""
This is a set of utils that are used by different scripts in the framework analysis
"""






# function that takes as an input a list of mandatory variables and regional coverage and 
# provides a list of scenarios that report on all of the mandatory variables for the given region
def mandatory_variables_scenarios(categories, regional, variables, database,
                                    subset=None):

    """
    Function that takes as an input a list of mandatory variables and regional coverage and 
    provides a list of scenarios that report on all of the mandatory variables for the given region

    Inputs:
    - database (AR6 database)
    - temperature scenarios
    - regions
    - variables

    Outputs:
    - list of scenarios that report on all of the mandatory variables for the given region

    """

    if regional == True:
        region = R10_CODES[2:3] 
    else:
        region = ['World']

    # ensure filtering by temperature category (subset or not)
    if subset:
        cat_df = database.filter(Category_subset=categories)        
    else:
        cat_df = database.filter(Category=categories)
    # else:
    #     raise ValueError('Subset must be a boolean')
    # cat_df = df.filter(Category_subset=categories)
    # if call_sub == None:
        
    #     if save_data == True:
    # # except:

    # Get the list of model scenario pairs reporting on all of the mandatory variables
    output_df = pd.DataFrame()
    region_df = cat_df.filter(region=region, variable=variables)
    
    # Group by model and scenario, then filter groups that have all mandatory variables
    grouped = region_df.data.groupby(['model', 'scenario'])
    valid_groups = grouped.filter(lambda x: x['variable'].nunique() == len(variables))
    
    # Extract unique model and scenario pairs
    model_scenario_pairs = valid_groups[['model', 'scenario']].drop_duplicates()
    
    output_df['model'] = model_scenario_pairs['model'].values
    output_df['scenario'] = model_scenario_pairs['scenario'].values
            

    return output_df



# function that takes as an input a list of variables and regions coverage and
# provides a list of scenarios that report on each variable variables for the given region
def create_variable_scenario_count(self, df, variables, regions, categories):

    # make outout dataframe
    output_df = pd.DataFrame()

    df = df.filter(year=2040, Category=categories)

    # add the variables to the output dataframe
    output_df['variable'] = variables

    # loop through each region
    for region in regions:

        region_df = df.filter(region=region)

        # count the number of scenarios for each variable
        scenario_count_list = region_df.data.groupby('variable')['scenario'].nunique().reindex(variables, fill_value=0).tolist()

        output_df[region] = scenario_count_list

    print(output_df)
    return output_df


    
def data_download(variables, models, scenarios, region, categories,
                    end_year, file_name=str):

    connAr6 = pyam.iiasa.Connection(name='ar6-public', 
                creds=None, 
                auth_url='https://api.manager.ece.iiasa.ac.at')    

    df = connAr6.query(model=models, scenario=scenarios,
        variable=variables, region=region,
        year=range(2020, end_year+1))
    
    print(df)
    df = df.filter(Category=categories)

    df.to_csv(file_name + '.csv')


def data_download_sub(variables, models, scenarios, categories, region, end_year):

    connAr6 = pyam.iiasa.Connection(name='ar6-public', 
                creds=None, 
                auth_url='https://api.manager.ece.iiasa.ac.at')    

    df = connAr6.query(model=models, scenario=scenarios, Category=categories,
        variable=variables, region=region, year=range(2020, end_year+1)
        )


    return df


def map_countries_to_regions(country_groups, country_data):

    """
    Function to map countries to regions based on country groupings. 
    The function takes as an input a dataframe of country groupings and a dataframe of country data.
    The function returns a dataframe with the countries mapped to their respective regions.

    Inputs: country_groups - dataframe of country groupings
            country_data - dataframe of country data

    Outputs: output_df - dataframe with countries mapped to regions
    
    """
    output_dict = {}
    # get the list groups of countries
    country_groups_list = country_groups['group'].unique().tolist()
    country_data_countries = country_data['country'].unique().tolist()
    
    output_df = pd.DataFrame()

    # Convert country names to ISO3 codes in both dataframes
    country_groups['countries'] = country_groups['countries'].str.split(', | and ').apply(lambda x: coco.convert(names=x, to='ISO3'))
    country_data['country'] = coco.convert(names=country_data['country'], to='ISO3')

    # Explode the lists of countries into separate rows
    country_groups = country_groups.explode('countries')

    # Merge the dataframes on the ISO3 country codes
    output_df = country_data.merge(country_groups, left_on='country', right_on='countries', how='left')

    # Drop the redundant 'countries' column
    output_df = output_df.drop(columns=['countries'])
    
    return output_df

