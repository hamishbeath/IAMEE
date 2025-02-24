import numpy as np
import pyam
import pandas as pd
from constants import *
from utils.file_parser import *



def main(run_regional=None, pyamdf=None, categories=None, scenarios=None, meta=None) -> None:

    if run_regional is None:
        run_regional = True

    if categories is None:
        categories = CATEGORIES_ALL[:2]

    if meta is None:
        meta = read_meta_data(META_FILE)
    
    if pyamdf is None:
        pyamdf = read_pyam_add_metadata(PROCESSED_DIR + 'Framework_pyam' + str(categories) + '.csv', meta)

    if scenarios is None:
        scenarios = read_csv(PROCESSED_DIR + 'Framework_scenarios' + str(categories))

    # import the regional data
    regional_base = read_csv(REGIONAL_BASE_PATH)
    
    if run_regional:
        output_df = pd.DataFrame()
        for region in R10_CODES:
            print('Calculating energy supply investment score for', region)
            to_append = energy_supply_investment_score(pyamdf, 0.023, 2100, scenarios, 
                                                        categories, regional=region, regional_thresholds=regional_base)
            output_df = pd.concat([output_df, to_append], ignore_index=True, axis=0)
            output_df.to_csv(OUTPUT_DIR + 'energy_supply_investment_score_regional' + str(categories) + '.csv')
    
    energy_supply_investment_score(pyamdf, 0.023, 2100, scenarios, categories)



# takes as an input a Pyam dataframe object with n number of scenarios in it. For each scenario it calculates both a binary 
def energy_supply_investment_score(pyam_df, base_value, end_year, scenario_model_list, categories, 
                                   regional=None, regional_thresholds=None):

    """
    This function takes an inputted Pyam dataframe and calculates to what extent the determined threshold for energy supply investment 
    in a given year, also giving the mean value over the whole time period. 

    The metric is the share of GDP that is invested in energy supply. The required variable from Pyam is 'Investment|Energy Supply' and
    'GDP|MER'.

    Inputs: Pyam dataframe object, base value for the threshold

    Outputs: TBC 1) Pyam dataframe object with additional columns for the score and mean value of the investment in energy supply
                2) .csv file with the results in standardised format for the rest of the analysis

    Base value is calculated based on historical shares of energy supply investment as a share of GDP from 2015 to 2023. Data from IMF
    and IEA
    """

    # Check if a regional filter is applied
    if regional is not None:
        region = regional

        # get region position in the R10 list
        region_index = R10_CODES.index(region)
        region_full_text = R10[region_index]

        # Filter out the data for the required region
        regional_threshold_data = regional_thresholds[regional_thresholds['region'] == region_full_text]

        # calculate the mean of the columns 2015 to 2023
        base_value = regional_threshold_data.iloc[:, 3:12].mean(axis=1)

    else:
        region = 'World'


    # Filter out the data for the required variables
    df = pyam_df.filter(variable=['Investment|Energy Supply','GDP|MER'],
                        region=region, year=range(2020, end_year+1),
                        scenario=scenario_model_list['scenario'],
                        model=scenario_model_list['model'])
    
    # get list of years between 2020 and 2100 at decedal intervals
    year_list = list(range(2020, end_year+1, 10))
    
    mean_value_list = []
    mean_value_2050_list = []

    for scenario, model in zip(scenario_model_list['scenario'], scenario_model_list['model']):
        
        # Filter out the data for the required scenario
        scenario_df = df.filter(scenario=scenario)
        scenario__model_df = scenario_df.filter(model=model)
        
        # make lists to calculate the results from
        share_of_gdp = []
        model_year_list = []
        proportion_of_base = []
        
        # Iterate through the years
        for year in year_list:    
 
            # Filter out the data for the required year
            year_df = scenario__model_df.filter(year=year)
            year_df = year_df.as_pandas()
            
            investment = year_df['value'][year_df['variable'] == 'Investment|Energy Supply'].values
            gdp = year_df['value'][year_df['variable'] == 'GDP|MER'].values

            # Calculate the share of GDP that is invested in energy supply
            year_share = investment[0] / gdp[0]
            share_of_gdp.append(year_share)
            model_year_list.append(year)
            proportion_of_base.append((year_share / base_value))
        
        # Calculate the mean value of the share of GDP that is invested in energy supply
        mean_value = np.mean(proportion_of_base)

        # Calculate the mean value up to 2050
        mean_value_2050 = np.mean(proportion_of_base[:4])
        
        mean_value_list.append(mean_value)
        mean_value_2050_list.append(mean_value_2050)

    # Create a dataframe with the mean value and mean value up to 2050
    output_df = pd.DataFrame()
    output_df['scenario'] = scenario_model_list['scenario']
    output_df['model'] = scenario_model_list['model']
    output_df['mean_value'] = mean_value_list
    output_df['mean_value_2050'] = mean_value_2050_list

    if regional is not None:
        output_df['region'] = region
        return output_df

    else:
        output_df.to_csv(OUTPUT_DIR + 'energy_supply_investment_score' + str(categories) + '.csv')



if __name__ == "__main__":
    main()