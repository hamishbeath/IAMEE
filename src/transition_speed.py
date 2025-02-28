import numpy as np
import pyam
import pandas as pd
from constants import *
from utils.file_parser import *


"""
transition_speed.py

File that contains the calculation of the transition speed metrics. This dimension is designed
to consider whether or not a scenario is likely to see public acceptance challenges based on
how quickly it transitions to a low-carbon economy. The metrics used are:
- Final demand reductions (max decadal reduction)
- Share of final energy from electricity (max decadal increase)
- Share of food demand from crops (max decadal increase)

Author: Hamish Beath
License: MIT

"""


def main(run_regional=None, pyamdf=None, categories=None, scenarios=None, meta=None) -> None:


    if run_regional is None:
        run_regional = True

    if categories is None:
        categories = CATEGORIES_DEFAULT

    if meta is None:
        meta = read_meta_data(META_FILE)
    
    if pyamdf is None:
        pyamdf = read_pyam_add_metadata(PROCESSED_DIR + 'Framework_pyam' + str(categories) + '.csv', meta)

    if scenarios is None:
        scenarios = read_csv(PROCESSED_DIR + 'Framework_scenarios' + str(categories))

    calculate_transition_speed_metrics(pyamdf, categories, scenarios, 2100)

    # if run_regional:
        
    #     output_df = pd.DataFrame()
    #     for region in R10_CODES:
    #         print('Running regional transition speed analysis for ' + region)
    #         to_append = calculate_transition_speed_metrics(pyamdf, categories, scenarios, 2100, regional=region)
    #         output_df = pd.concat([output_df, to_append], ignore_index=True, axis=0)
        
    #     output_df.to_csv(OUTPUT_DIR + 'transition_speed_metrics_regional' + str(categories) + '.csv', index=False)
    

# Function that calculates the necessary metrics for the transition speed analysis
def calculate_transition_speed_metrics(pyamdf, categories, scenarios, end_year, regional=None):
    
    
    """
    Function for calculating the necessary metrics for the transition speed analysis. 
    This includes the following:
    - Final demand reductions (max decadal reduction)
    - Share of final energy from electricity (max decadal increase)
    - Share of food demand from crops (max decadal increase)
    
    Inputs:
    - pyamdf: The pyam dataframe database snippet ready for the analysis
    - categories: The category(ies) that will be compared in the analysis
    - scenarios: The scenario list for scenarios that have all the mandatory variables
    - end_year: The end year of the analysis
    - regional: Whether the analysis is regional or global

    Outputs:
    - a .csv file with the max decadal values for each indicator for every scenario

    
    """
     # Check if a regional filter is applied
    if regional is not None:
        region = regional
    else:
        region = 'World'

    # Filter the pyam dataframe for the necessary variables
    df = pyamdf.filter(scenario=scenarios['scenario'], model=scenarios['model'], year=range(2020, end_year+1), region=region,
                       variable=TRANSITION_SPEED_VARIABLES)
    # Get the data from the pyam dataframe
    df = df.data

    # Calculate the max decadal values for each indicator 
    df = df.pivot_table(index=['model', 'scenario', 'year'], columns='variable', values='value').reset_index()
    
    # share of final energy from electricity
    df['Final energy share electricity'] = df['Final Energy|Electricity'] / df['Final Energy']

    # share of food demand from crops
    df['Share of food demand from crops'] = df['Food Demand|Crops'] / (df['Food Demand|Crops'] + df['Food Demand|Livestock'])

    # df = df.to_csv(OUTPUT_DIR + 'transition_speed_data' + str(categories) + '.csv', index=False)

    final_demand_reductions = []
    electrification_increases = []
    crop_share_increases = []

    df = df.reset_index(drop=True)  

    # remove years that are not decadal
    df = df[df['year'] % 10 == 0]

    for scenario, model in zip(scenarios['scenario'], scenarios['model']):

        scenario_df = df[df['scenario'] == scenario]
        scenario_model_df = scenario_df[scenario_df['model'] == model]

        scenario_model_df = df[(df['model'] == model) & (df['scenario'] == scenario)]

        # final demand reductions
        final_demand_reductions.append(scenario_model_df['Final Energy'].diff().groupby((scenario_model_df['year'] // 10) * 10).sum().min())

        # share of final energy from electricity
        electrification_increases.append(scenario_model_df['Final energy share electricity'].diff().groupby((scenario_model_df['year'] // 10) * 10).sum().max())

        # share of food demand from crops
        crop_share_increases.append(scenario_model_df['Share of food demand from crops'].diff().groupby((scenario_model_df['year'] // 10) * 10).sum().max())

    output_df = pd.DataFrame({'model': scenarios['model'], 'scenario': scenarios['scenario'],
                              'Final demand reductions': final_demand_reductions,
                              'Share of final energy from electricity': electrification_increases,
                              'Share of food demand from crops': crop_share_increases})

    if regional is not None:
        output_df['region'] = region
        return output_df

    else:
        # Save the results to a .csv file
        output_df.to_csv(OUTPUT_DIR + 'transition_speed_metrics' + str(categories) + '.csv', index=False)

