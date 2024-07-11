import numpy as np
import pyam
import pandas as pd
from utils import Data
from pygini import gini
from robust import run_regional_carbon_budgets
from robust import Robust
class Fairness:


    regional_carbon_budget_shares = pd.read_csv('inputs/R10_carbon_budget_shares.csv')
    try:
        regional_budget_results = pd.read_csv('outputs/carbon_budget_shares_regional' + 
                                              str(Data.categories) + '.csv')
    except FileNotFoundError:
        print('Attempting run of regional carbon budgets')
        run_regional_carbon_budgets()
        regional_budget_results = pd.read_csv('outputs/carbon_budget_shares_regional' + 
                                              str(Data.categories) + '.csv')
        
def main() -> None:


    between_region_gini(Data.regional_dimensions_pyamdf, 
                        Data.model_scenarios, 2100, Data.categories)
    carbon_budget_fairness(Data.model_scenarios, 
                           Fairness.regional_carbon_budget_shares, 
                           Fairness.regional_budget_results)


# Function that calculates the Gini coefficient between R10 regions
def between_region_gini(pyam_df, scenario_model_list, end_year, categories):

    # filter for the variables needed
    df = pyam_df.filter(variable=['GDP|MER'],
                        year=range(2020, end_year+1),
                        scenario=scenario_model_list['scenario'], 
                        model=scenario_model_list['model'])

    ginis = []
    # iterate over the scenarios
    for scenario, model in zip(scenario_model_list['scenario'], scenario_model_list['model']):
        
        # Filter out the data for the required scenario
        scenario_model_df = df.filter(model=model, scenario=scenario)
        region_gdps = np.array([])
        for region in Data.R10:
            if region == 'World':
                continue
        
            # filter for the regions
            region_df = pd.Series(scenario_model_df.filter(region=region).data['value'].values,
                                  index=scenario_model_df.filter(region=region).data['year'])
            cumulative_gdp = pyam.timeseries.cumulative(region_df, 2020, end_year)
            region_gdps = np.append(region_gdps, cumulative_gdp)

        # Calculate the Gini coefficient for the region using gini package
        ginis.append(gini(region_gdps))

    # Create a dataframe with the ginis
    between_region_gini_df = pd.DataFrame({'model': scenario_model_list['model'], 
                                           'scenario': scenario_model_list['scenario'], 
                                           'between_region_gini': ginis})
    between_region_gini_df.to_csv(
        'outputs/between_region_gini' + str(Data.categories) + '.csv', index=False)


# Function that gives a score based on the relative shares of the carbon budget used
# in the global north, and the global south by region
def carbon_budget_fairness(scenario_model_list, 
                           carbon_budgets, regional_scenario_results):
    """
    Function that gives a score based on the relative shares of the carbon budget used
    in the global north, and the global south for each scenario

    Parameters:
    pyam_df: pyam dataframe
    scenario_model_list: csv file with the scenarios and models
    carbon_budgets: csv file with the carbon budgets for each region
    regional_scenario_results: csv file with the regional scenario results of carbon
    budget usage by each region   

    Output: csv file with the carbon budget fairness for each scenario
    """
    # if world present in the R10 list, remove it as not needed
    if Data.R10[0] == 'World':
        Data.R10 = Data.R10[1:]
    
    # calculate global north and global south carbon budgets
    budgets = {'South': 0, 'North': 0}
    for region in Data.R10_codes:
        if Data.R10_development[region] == 'South':
            budgets['South'] += carbon_budgets[region][0] * Robust.remaining_carbon_budget_2030
        else:
            budgets['North'] += carbon_budgets[region][0] * Robust.remaining_carbon_budget_2030

    carbon_budget_fairness = []
    # loop through the scenarios and calculate the north south budgets used
    for scenario, model in zip(scenario_model_list['scenario'], scenario_model_list['model']):
        
        # filter the dataframe for the scenario and model
        regional_scenario_results_filtered = regional_scenario_results[(regional_scenario_results['model'] == model) &
                                                                        (regional_scenario_results['scenario'] == scenario)]
        
        # calculate the north and south budgets used
        budgets_used = {'South': 0, 'North': 0}
        for region, i in zip(Data.R10, range(len(Data.R10_codes))):
            
            # filter the dataframe for the region
            scenario_region_data = regional_scenario_results_filtered[(regional_scenario_results_filtered['region'] == region)]
            if Data.R10_development[Data.R10_codes[i]] == 'South':
                budgets_used['South'] += scenario_region_data['carbon_budget_share'].values[0] * (Robust.remaining_carbon_budget_2030 * carbon_budgets[Data.R10_codes[i]][0])
            else:
                budgets_used['North'] += scenario_region_data['carbon_budget_share'].values[0] * (Robust.remaining_carbon_budget_2030 * carbon_budgets[Data.R10_codes[i]][0])

        # calculate shares of north and south budgets used
        north_share = budgets_used['North'] / budgets['North']
        south_share = budgets_used['South'] / budgets['South']

        # calculate the fairness score (lower score is better for fairness)
        carbon_budget_fairness.append(north_share - south_share)
    
    # create a dataframe with the carbon budget fairness
    carbon_budget_fairness_df = pd.DataFrame({'model': scenario_model_list['model'], 
                                              'scenario': scenario_model_list['scenario'], 
                                              'carbon_budget_fairness': carbon_budget_fairness})
    
    carbon_budget_fairness_df.to_csv(
        'outputs/carbon_budget_fairness' + str(Data.categories) + '.csv', index=False)



    





if __name__ == "__main__":
    main()
    