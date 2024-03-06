import numpy as np
import pyam
import pandas as pd
# from utils import Utils
from utils import Data



class Resilience:

    energy_variables = variable=['Primary Energy|Coal','Primary Energy|Oil', 
                        'Primary Energy|Gas', 'Primary Energy|Nuclear',
                        'Primary Energy|Biomass', 'Primary Energy|Non-Biomass Renewables']

    gini_between_countries = pd.read_csv('inputs/gini_btw_6.csv')


def main() -> None:

    shannon_index_energy_mix(Data.dimensions_pyamdf, Data.model_scenarios, 2100, Data.categories)
    final_energy_demand(Data.dimensions_pyamdf, Data.model_scenarios, 2100, Data.categories)
    gini_between_countries(Data.dimensions_pyamdf, 
                           Data.model_scenarios, 
                           2100, 
                           Data.meta_df,
                           Resilience.gini_between_countries,
                           Data.categories)



# Function that calculates the shannon index for the energy mix for each scenario
def shannon_index_energy_mix(pyam_df, scenario_model_list, end_year, categories):

    # filter for the variables needed
    df = pyam_df.filter(variable=Resilience.energy_variables,
                        region='World',
                        year=range(2020, end_year+1),
                        scenario=scenario_model_list['scenario'], 
                        model=scenario_model_list['model'])
    
    shannon_indexes = []
     # loop through models and scenarios
    for scenario, model in zip(scenario_model_list['scenario'], scenario_model_list['model']):

        # Filter out the data for the required scenario
        scenario_df = df.filter(scenario=scenario)
        scenario__model_df = scenario_df.filter(model=model)
        
        # create a dictionary to store the summed energy values
        energy_summed = {}
        total = 0
        # loop through the energy variables to interpolate and sum the values
        for variable in Resilience.energy_variables:
            # Filter out the data for the required variable
            variable_df = scenario__model_df.filter(variable=variable) 
            # make pandas series with the values and years as index
            variable_df = variable_df.data
            variable_series = pd.Series(variable_df['value'].values, index=variable_df['year'])
            cumulative_interpolated = pyam.timeseries.cumulative(variable_series, 2020, 2100)
            energy_summed[variable] = cumulative_interpolated
            total += cumulative_interpolated
        
        # make a new dictionary to store the proportions of the energy sources 
        #  and calculate the shannon index
        proportions = {}
        shannon_total = 0
        for variable in Resilience.energy_variables:
            proportion = energy_summed[variable] / total
            proportions[variable] = proportion
            shannon_index_value = proportion * np.log(proportion)
            shannon_total += shannon_index_value
        shannon_index = -1 * shannon_total
        shannon_indexes.append(shannon_index)
    
    # create a new dataframe with the shannon indexes
    shannon_df = pd.DataFrame({'model': scenario_model_list['model'], 'scenario': scenario_model_list['scenario'], 'shannon_index': shannon_indexes})
    shannon_df.to_csv('outputs/shannon_diversity_index' + str(categories) + '.csv', index=False)


# Function that calculates the cumulative final energy demand for each scenario
def final_energy_demand(pyam_df, scenario_model_list, end_year, categories):
    
    # filter for the variables needed
    df = pyam_df.filter(variable='Final Energy',
                        region='World',
                        year=range(2020, end_year+1),
                        scenario=scenario_model_list['scenario'], 
                        model=scenario_model_list['model'])
    
    final_energy_demand = []
    
    # loop through models and scenarios
    for scenario, model in zip(scenario_model_list['scenario'], scenario_model_list['model']):
        
        # Filter out the data for the required scenario
        scenario_df = df.filter(scenario=scenario)
        scenario_model_df = scenario_df.filter(model=model)
        
        # make pandas series with the values and years as index
        variable_df = scenario_model_df.data
        variable_series = pd.Series(variable_df['value'].values, index=variable_df['year'])
        cumulative_interpolated = pyam.timeseries.cumulative(variable_series, 2020, 2100)
        final_energy_demand.append(cumulative_interpolated)

    # create a new dataframe with the shannon indexes
    final_energy_demand_df = pd.DataFrame({'model': scenario_model_list['model'], 
                                           'scenario': scenario_model_list['scenario'], 
                                           'final_energy_demand': final_energy_demand})
    final_energy_demand_df.to_csv('outputs/final_energy_demand' + str(categories) + '.csv', index=False)

    
# Function that gives the gini coefficient and SSP population for each scenario
def gini_between_countries(pyam_df, scenario_model_list, end_year, meta_df, gini_df, categories):
    
    # filter for the variables needed
    df = pyam_df.filter(variable='Emissions|CO2',
                        region='World',
                        year=range(2020, end_year+1),
                        scenario=scenario_model_list['scenario'], 
                        model=scenario_model_list['model'])
    
    # loop through the ssps and the gini data to get the between country 
    # gini coefficients and the population for each ssp
    ssp_ginis = {}
    for ssp in range(1, 6):
        
        ssp_string = 'SSP' + str(ssp)   
        ssp_cells = gini_df[gini_df['scen'] == ssp_string]   
        ssp_cells.set_index('period', inplace=True)
        
        # select the rows for the years 2020 to 2100
        ssp_cells = ssp_cells.loc[2025:2100]
        current_ssp_cells_gini = np.mean(ssp_cells['gini_world_mig'])
        ssp_ginis[ssp_string] = current_ssp_cells_gini
        
    print(ssp_ginis)

    ssp_gini_coefficients = []
    ssps = []
    # loop through models and scenarios
    for scenario, model in zip(scenario_model_list['scenario'], scenario_model_list['model']):
        
        # Filter out the data for the required scenario
        scenario_df = df.filter(scenario=scenario)
        scenario_model_df = scenario_df.filter(model=model)
        scenario_ssp = meta_df[meta_df['model'] == model]
        scenario_ssp = scenario_ssp[scenario_ssp['scenario'] == scenario]    
        scenario_ssp = int(scenario_ssp['Ssp_family'].values[0])
        ssps.append(scenario_ssp)
        scenario_ssp_gini = ssp_ginis['SSP' + str(scenario_ssp)]
        ssp_gini_coefficients.append(scenario_ssp_gini)

    # create a new dataframe with the gini coefficients
    
    # create a new dataframe with the gini coefficients
    gini_df = pd.DataFrame({'model': scenario_model_list['model'], 
                            'scenario': scenario_model_list['scenario'], 
                            'ssp_gini_coefficient': ssp_gini_coefficients,
                            'ssp': ssps})
    gini_df.to_csv('outputs/gini_coefficient' + str(categories) +  '.csv', index=False)
    ssp_gini_coefficients = pd.DataFrame({'ssp': list(ssp_ginis.keys()), 
                                          'gini_coefficient': list(ssp_ginis.values())})



if __name__ == "__main__":
    main()
    