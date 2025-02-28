import numpy as np
import pyam
import pandas as pd
# from utils import Utils
# from utils import Data
from constants import *
from utils.file_parser import *

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

    if run_regional:
        regional_df = pd.DataFrame()
        for region in R10_CODES:
            to_append = forest_cover_change(pyamdf, 2100, scenarios, BIOENERGY_THRESHOLD, categories, regional=region)    
            regional_df = pd.concat([regional_df, to_append], ignore_index=True, axis=0)

        regional_df.to_csv(OUTPUT_DIR + 'environmental_metrics_regional' + str(categories) + '.csv')
    
    forest_cover_change(pyamdf, 2100, scenarios, BIOENERGY_THRESHOLD, categories, regional=None)


def forest_cover_change(pyam_df, end_year, scenario_model_list, beccs_threshold, categories, regional=None):
    
    """
    This function calculates the change in forest cover from 2020 to 2050 and 2020 to 2100 for a given scenario and model.
    The function also checks if the bioenergy threshold is breached for the given scenario and model.
    
    Inputs:
    pyam_df: A pyam dataframe object with the scenario timeseries data
    end_year: The final year of the analysis
    scenario_model_list: A .csv file with the scenario and model names
    bioenergy_threshold: The sustainability threshold for bioenergy in EJ/yr
    categories: The categories of the scenarios
    regional: The region for which the analysis is done (None for global analysis)

    Outputs: 
    A .csv file with the forest cover change values and whether the bioenergy threshold is breached
    
    """

    # Check if a regional filter is applied
    if regional is not None:
        region = regional
        
        # calculate the beccs threshold for the region

        threshold_df = pyam_df.filter(variable='Land Cover',region=[region,'World'],
                        year=2020,
                        scenario=scenario_model_list['scenario'], 
                        model=scenario_model_list['model'])
        world_land_cover = np.mean(threshold_df['value'][threshold_df['region'] == 'World'].values)
        region_land_cover = np.mean(threshold_df['value'][threshold_df['region'] == region].values)
        # share_of_beccs = int(region_land_cover) / int(world_land_cover)
        beccs_threshold = (region_land_cover / world_land_cover) * beccs_threshold
        
    else:
        region = 'World'
    
    print('Calculating the environmental metrics for the region', region)
    # filter for the variables needed
    df = pyam_df.filter(variable=['Land Cover|Forest|Natural Forest','Land Cover','Primary Energy|Biomass'],region=region,
                        year=range(2020, end_year+1),
                        scenario=scenario_model_list['scenario'], 
                        model=scenario_model_list['model'])

    year_list = list(range(2020, end_year+1, 10))

    bioenergy_threshold_breached = []
    forest_change_2050 = []
    forest_change_2100 = []
    for scenario, model in zip(scenario_model_list['scenario'], scenario_model_list['model']):
        
        # Filter out the data for the required scenario
        scenario_df = df.filter(scenario=scenario)
        scenario__model_df = scenario_df.filter(model=model)

        base_value = 0
        forest_cover_values = []
        bioenergy_seq_values = []
        
        # Iterate through the years
        for year in year_list:    

            # Filter out the data for the required year
            year_df = scenario__model_df.filter(year=year)
            year_df = year_df.as_pandas()

            # extract necessary values 
            land_cover = year_df['value'][year_df['variable'] == 'Land Cover'].values
            forest_cover = year_df['value'][year_df['variable'] == 'Land Cover|Forest|Natural Forest'].values
            bioenergy_seq = year_df['value'][year_df['variable'] == 'Primary Energy|Biomass'].values 
            share_of_forest = forest_cover[0] / land_cover[0]

            # if 2020 store as 'base' year for given scenario 
            if year == 2020:
                base_value = share_of_forest
            # for all other 
            else:
                forest_cover_values.append(share_of_forest - base_value)

            bioenergy_seq_values.append(bioenergy_seq[0])   

        # Check if the beccs threshold is breached
        if any(i > beccs_threshold for i in bioenergy_seq_values):
            bioenergy_threshold_breached.append(1)
        else:
            bioenergy_threshold_breached.append(0)
        
        # Append the forest cover change values to the list
        forest_change_2050.append(forest_cover_values[2])
        forest_change_2100.append(forest_cover_values[-1])


    # Create a dataframe with the mean value and mean value up to 2050
    output_df = pd.DataFrame()
    output_df['scenario'] = scenario_model_list['scenario']
    output_df['model'] = scenario_model_list['model']
    output_df['forest_change_2050'] = forest_change_2050
    output_df['forest_change_2100'] = forest_change_2100
    output_df['bioenergy_threshold_breached'] = bioenergy_threshold_breached
    
    if regional is not None:
        
        # add column for region with region in each row
        output_df['region'] = region
        return output_df
    
    else:
        output_df.to_csv(OUTPUT_DIR + 'environmental_metrics' + str(categories) + '.csv')

        

if __name__ == "__main__":
    main()