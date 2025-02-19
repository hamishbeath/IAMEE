import numpy as np
import pyam
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
from utils import Utils
from utils import Data
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# Defines simple weighting system for indicators of sustainability, adjusting the weights will change the spread of how scenarios 
# within each temperature category will score. The idea is to tease out tradeoffs and synergies. 
class EnvSus:
    
    




    plotting_variables = ['Water Consumption', 'Land Cover|Pasture', 'Land Cover|Forest',
                          'Land Cover|Cropland', 'Land Cover|Cropland|Energy Crops']            
    emissions = ['Emissions|CO2']
  
    plotting_category_colours = {'C1':'#f57200', 'C3':'#6302d9', 'C5':'#1b9e77'}
    violin_colours = ['#8CCFF4','#7FCACC','#006B7F']
    run_mode = 'cat'
    regions = ['World', 'Countries of Sub-Saharan Africa']
    region = ['Countries of Sub-Saharan Africa']
    checked_variables = pd.read_csv('variable_categories.csv')

    # beccs_threshold = 2800 # in mtCO2 / year medium threshold, high conversion efficiency value from deprez et al 2024
    bioenergy_threshold = 100 # in EJ / year medium threshold, high conversion efficiency value from Creutzig et al 2015

def main() -> None:

    # data_download()
    # plot_outputs()
    # plot_using_pyam()
    # violin_plots()
    # Utils.simple_stats(Utils, 'AR6', EnvSus.regions, EnvSus.emissions, EnvSus.categories)
    # joel_data_download()
    # Utils.export_variable_list(Utils, 'AR6', ['C1', 'C2'])
    # Utils.create_variable_sheet(Utils,
    #                              'AR6',
    #                              EnvSus.category_subset_paris,
    #                                regions=EnvSus.regions, 
    #                                variables=EnvSus.checked_variables['variable'].tolist(), 
    #                                variable_sheet=EnvSus.checked_variables)
    # Utils.test_coco()
    # Utils.snapshot_cluster_analysis(Utils, 'World', Data.c1aR10_scenarios,['Land Cover|Forest', 'Land Cover|Cropland'],'C1a_NZGHGs' , 3, 2100)
    # Utils.time_series_cluster_analysis(Utils, 'World', Data.c1aR10_scenarios,['Land Cover|Forest', 'Land Cover|Cropland'],'C1a_NZGHGs' , 4)

    # make_scenario_project_list()
    # Utils.manadory_variables_scenarios(Utils, ['C1','C2'], EnvSus.regions, Data.mandatory_variables, subset=False)
    empty_df = pd.DataFrame()
    for region in Data.R10:
        to_append = forest_cover_change(Data.regional_dimensions_pyamdf, 2100, Data.model_scenarios, EnvSus.beccs_threshold, Data.categories, regional=region)    
        empty_df = pd.concat([empty_df, to_append], ignore_index=True, axis=0)
    print(empty_df)
    
    empty_df.to_csv('outputs/environmental_metrics_regional' + str(Data.categories) + '.csv')
    
    # forest_cover_change(Data.regional_dimensions_pyamdf, 2100, Data.model_scenarios, EnvSus.beccs_threshold, Data.categories, regional=None)


def forest_cover_change(pyam_df, end_year, scenario_model_list, beccs_threshold, categories, regional=None):
    """
    This function calculates the change in forest cover from 2020 to 2050 and 2020 to 2100 for a given scenario and model.
    The function also checks if the BECCS threshold is breached for the given scenario and model.
    
    Inputs:
    pyam_df: A pyam dataframe object with the scenario timeseries data
    end_year: The final year of the analysis
    scenario_model_list: A .csv file with the scenario and model names
    beccs_threshold: The threshold for BECCS in mtCO2/year
    categories: The categories of the scenarios
    regional: The region for which the analysis is done (None for global analysis)

    Outputs: 
    A .csv file with the forest cover change values and whether the BECCS threshold is breached
    
    """

    # Check if a regional filter is applied
    if regional is not None:
        region = regional
        
        # calculate the beccs threshold for the region
        print('Calculating the Bioenergy threshold for the region: ', region)
        threshold_df = pyam_df.filter(variable='Land Cover',region=[region,'World'],
                        year=2020,
                        scenario=scenario_model_list['scenario'], 
                        model=scenario_model_list['model'])
        world_land_cover = np.mean(threshold_df['value'][threshold_df['region'] == 'World'].values)
        region_land_cover = np.mean(threshold_df['value'][threshold_df['region'] == region].values)
        share_of_beccs = int(region_land_cover) / int(world_land_cover)
        beccs_threshold = (region_land_cover / world_land_cover) * beccs_threshold
        
    else:
        region = 'World'
    
    # filter for the variables needed
    df = pyam_df.filter(variable=['Land Cover|Forest|Natural Forest','Land Cover','Primary Energy|Biomass'],region=region,
                        year=range(2020, end_year+1),
                        scenario=scenario_model_list['scenario'], 
                        model=scenario_model_list['model'])

    year_list = list(range(2020, end_year+1, 10))

    beccs_threshold_breached = []
    forest_change_2050 = []
    forest_change_2100 = []
    
    for scenario, model in zip(scenario_model_list['scenario'], scenario_model_list['model']):
        
        # Filter out the data for the required scenario
        scenario_df = df.filter(scenario=scenario)
        scenario__model_df = scenario_df.filter(model=model)

        base_value = 0
        forest_cover_values = []
        beccs_seq_values = []
        
        # Iterate through the years
        for year in year_list:    

            # Filter out the data for the required year
            year_df = scenario__model_df.filter(year=year)
            year_df = year_df.as_pandas()

            # extract necessary values 
            land_cover = year_df['value'][year_df['variable'] == 'Land Cover'].values
            forest_cover = year_df['value'][year_df['variable'] == 'Land Cover|Forest|Natural Forest'].values
            beccs_seq = year_df['value'][year_df['variable'] == 'Primary Energy|Biomass'].values #to do, update variable names to bioenergy
            share_of_forest = forest_cover[0] / land_cover[0]

            # if 2020 store as 'base' year for given scenario 
            if year == 2020:
                base_value = share_of_forest
            # for all other 
            else:
                forest_cover_values.append(share_of_forest - base_value)

            beccs_seq_values.append(beccs_seq[0])   

        # Check if the beccs threshold is breached
        if any(i > beccs_threshold for i in beccs_seq_values):
            beccs_threshold_breached.append(1)
        else:
            beccs_threshold_breached.append(0)
        
        # Append the forest cover change values to the list
        forest_change_2050.append(forest_cover_values[2])
        forest_change_2100.append(forest_cover_values[-1])


    # Create a dataframe with the mean value and mean value up to 2050
    output_df = pd.DataFrame()
    output_df['scenario'] = scenario_model_list['scenario']
    output_df['model'] = scenario_model_list['model']
    output_df['forest_change_2050'] = forest_change_2050
    output_df['forest_change_2100'] = forest_change_2100
    output_df['bioenergy_threshold_breached'] = beccs_threshold_breached
    
    if regional is not None:
        
        # add column for region with region in each row
        output_df['region'] = region
        return output_df
    
    else:
        output_df.to_csv('outputs/environmental_metrics' + str(categories) + '.csv')

        


if __name__ == "__main__":
    main()