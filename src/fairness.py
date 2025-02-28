import numpy as np
import pyam
import pandas as pd
from pygini import gini
from robust import run_regional_carbon_budgets
from constants import *
from utils.file_parser import *



def main(pyamdf=None, categories=None, scenarios=None, meta=None) -> None:

    if categories is None:
        categories = CATEGORIES_DEFAULT
        
    if meta is None:
        meta = read_meta_data(META_FILE)
    
    if pyamdf is None:
        pyamdf = read_pyam_add_metadata(PROCESSED_DIR + 'Framework_pyam' + str(categories) + '.csv', meta)

    if scenarios is None:
        scenarios = read_csv(PROCESSED_DIR + 'Framework_scenarios' + str(categories))

    regional_carbon_budget_shares = read_csv(INPUT_DIR + 'R10_carbon_budget_shares.csv')

    try:
        regional_budget_results = pd.read_csv(OUTPUT_DIR + 'carbon_budget_shares_regional' + 
                                              str(categories) + '.csv')
    except FileNotFoundError:
        print('Attempting run of regional carbon budgets')
        run_regional_carbon_budgets()
        regional_budget_results = read_csv(OUTPUT_DIR + 'carbon_budget_shares_regional' + 
                                              str(categories) + '.csv')

    print('Running interregional fairness analysis')
    print('Calculating between region gini')
    between_region_gini(pyamdf, scenarios, 2100, categories)
    print('Calculating carbon budget fairness')
    carbon_budget_fairness(scenarios, 
                           regional_carbon_budget_shares, 
                           regional_budget_results, categories)


# Function that calculates the Gini coefficient between R10 regions
def between_region_gini(pyam_df, scenario_model_list, end_year, categories):

    # filter for the variables needed
    df = pyam_df.filter(variable=['GDP|MER', 'Population'],
                        year=range(2020, end_year+1),
                        scenario=scenario_model_list['scenario'], 
                        model=scenario_model_list['model'])

    ginis = []
    # iterate over the scenarios
    for scenario, model in zip(scenario_model_list['scenario'], scenario_model_list['model']):
        
        # Filter out the data for the required scenario
        scenario_model_df = df.filter(model=model, scenario=scenario)
        region_gdps = np.array([])
        for region in R10_CODES:
            if region == 'World':
                continue
        
            # filter for the regions
            region_df_gdp = pd.Series(scenario_model_df.filter(region=region, variable='GDP|MER').data['value'].values,
                                  index=scenario_model_df.filter(region=region, variable='GDP|MER').data['year'])
            region_df_pop = pd.Series(scenario_model_df.filter(region=region, variable='Population').data['value'].values,
                                    index=scenario_model_df.filter(region=region, variable='Population').data['year'])
            cumulative_gdp = pyam.timeseries.cumulative(region_df_gdp, 2020, end_year)
            cumulative_pop = pyam.timeseries.cumulative(region_df_pop, 2020, end_year)
            gdp_capita = cumulative_gdp / cumulative_pop
            region_gdps = np.append(region_gdps, gdp_capita)

        # Calculate the Gini coefficient for the region using gini package
        ginis.append(gini(region_gdps))

    # Create a dataframe with the ginis
    between_region_gini_df = pd.DataFrame({'model': scenario_model_list['model'], 
                                           'scenario': scenario_model_list['scenario'], 
                                           'between_region_gini': ginis})
    between_region_gini_df.to_csv(
        OUTPUT_DIR + 'between_region_gini' + str(categories) + '.csv', index=False)


# Function that gives a score based on the relative shares of the carbon budget used
# in the global north, and the global south by region
def carbon_budget_fairness(scenario_model_list, 
                           carbon_budgets, regional_scenario_results, categories):
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

    # calculate global north and global south carbon budgets
    budgets = {'South': 0, 'North': 0}
    for region in R10_CODES:
        if R10_DEVELOPMENT[region] == 'South':
            budgets['South'] += carbon_budgets[region][0] * REMAINING_2030_BUDGET
        else:
            budgets['North'] += carbon_budgets[region][0] * REMAINING_2030_BUDGET

    carbon_budget_fairness = []
    # loop through the scenarios and calculate the north south budgets used
    for scenario, model in zip(scenario_model_list['scenario'], scenario_model_list['model']):
        
        # filter the dataframe for the scenario and model
        regional_scenario_results_filtered = regional_scenario_results[(regional_scenario_results['model'] == model) &
                                                                        (regional_scenario_results['scenario'] == scenario)]
        
        # calculate the north and south budgets used
        budgets_used = {'South': 0, 'North': 0}
        for region in R10_CODES:
            
            # filter the dataframe for the region
            scenario_region_data = regional_scenario_results_filtered[(regional_scenario_results_filtered['region'] == region)]
            if R10_DEVELOPMENT[region] == 'South':
                budgets_used['South'] += scenario_region_data['carbon_budget_share'].values[0] * (REMAINING_2030_BUDGET * carbon_budgets[region][0])
            else:
                budgets_used['North'] += scenario_region_data['carbon_budget_share'].values[0] * (REMAINING_2030_BUDGET * carbon_budgets[region][0])

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
        OUTPUT_DIR + 'carbon_budget_fairness' + str(categories) + '.csv', index=False)



    
if __name__ == "__main__":
    main()
    