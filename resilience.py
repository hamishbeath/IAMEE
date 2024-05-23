import numpy as np
import pyam
import pandas as pd
# from utils import Utils
from utils import Data
import country_converter as coco
from pygini import gini

class Resilience:

    energy_variables = variable=['Primary Energy|Coal','Primary Energy|Oil', 
                        'Primary Energy|Gas', 'Primary Energy|Nuclear',
                        'Primary Energy|Biomass', 'Primary Energy|Non-Biomass Renewables']

    gini_between_countries = pd.read_csv('inputs/gini_btw_6.csv')
    ssp_gini_data = pd.read_csv('inputs/ssp_population_gdp_projections.csv')
    regional_gini = pd.read_csv('outputs/within_region_gini.csv')

def main() -> None:

    # Run global indicators
    shannon_index_energy_mix(Data.regional_dimensions_pyamdf, Data.model_scenarios, 2100, Data.categories, regional=None)
    final_energy_demand(Data.regional_dimensions_pyamdf, Data.model_scenarios, 2100, Data.categories, regional=None)
    gini_between_countries(Data.regional_dimensions_pyamdf,
                        Data.model_scenarios, 
                        2100, 
                        Data.meta_df,
                        Resilience.gini_between_countries,
                        Data.categories, regional=None)

    # Run regional indicators
    final_energy = pd.DataFrame()
    shannon = pd.DataFrame()
    gini = pd.DataFrame()
    for region in Data.R10:
        # to_append = shannon_index_energy_mix(Data.regional_dimensions_pyamdf, Data.model_scenarios, 2100, Data.categories, regional=region)
        to_append_energy = final_energy_demand(Data.regional_dimensions_pyamdf, Data.model_scenarios, 2100, Data.categories, regional=region)
        to_append_shannon = shannon_index_energy_mix(Data.regional_dimensions_pyamdf, Data.model_scenarios, 2100, Data.categories, regional=region)
        to_append_gini = gini_between_countries(Data.dimensions_pyamdf, Data.model_scenarios, 2100, Data.meta_df, Resilience.gini_between_countries, Data.categories, regional=region)
        final_energy = pd.concat([final_energy, to_append_energy], ignore_index=True, axis=0)
        shannon = pd.concat([shannon, to_append_shannon], ignore_index=True, axis=0)
        gini = pd.concat([gini, to_append_gini], ignore_index=True, axis=0)
    shannon.to_csv('outputs/shannon_diversity_index_regional' + str(Data.categories) + '.csv')
    final_energy.to_csv('outputs/final_energy_demand_regional' + str(Data.categories) + '.csv')
    gini.to_csv('outputs/gini_coefficient_regional' + str(Data.categories) + '.csv')


    # get_within_region_gini(Resilience.ssp_gini_data, Data.region_country_df, 
    #                        Data.R10_codes, 2025)


# Function that calculates the shannon index for the energy mix for each scenario
def shannon_index_energy_mix(pyam_df, scenario_model_list, end_year, categories, regional=None):

     # Check if a regional filter is applied
    if regional is not None:
        region = regional
    else:
        region = 'World'
    
    # filter for the variables needed
    df = pyam_df.filter(variable=Resilience.energy_variables,
                        region=region,
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
    
    if regional is not None:
        shannon_df['region'] = region
        return shannon_df
    else:
        shannon_df.to_csv('outputs/shannon_diversity_index' + str(categories) + '.csv', index=False)


# Function that calculates the cumulative final energy demand for each scenario
def final_energy_demand(pyam_df, scenario_model_list, end_year, categories, regional=None):
    
    if regional is not None:
        region = regional
    else:
        region = 'World'
    print(region)
    # filter for the variables needed
    df = pyam_df.filter(variable=['Final Energy','GDP|MER'],  # mention
                        region=region,
                        year=range(2020, end_year+1),
                        scenario=scenario_model_list['scenario'], 
                        model=scenario_model_list['model'])
    gdp = pyam_df.filter(variable='GDP|MER')
    df = df.filter(variable='Final Energy')
    
    final_energy_demand = []
    gdp_values = []

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

        if regional is not None:
            gdp_df = gdp.filter(scenario=scenario)
            gdp_model_df = gdp_df.filter(model=model)
            gdp_interpolated = gdp_model_df.interpolate(range(2020, end_year)).data.copy()
            gdp_cumulative = gdp_interpolated['value'].sum()
            gdp_values.append(gdp_cumulative)


    # create a new dataframe with the shannon indexes
    final_energy_demand_df = pd.DataFrame({'model': scenario_model_list['model'], 
                                           'scenario': scenario_model_list['scenario'], 
                                           'final_energy_demand': final_energy_demand})
    if regional is not None:
        energy_per_gdp = [x/y for x, y in zip(final_energy_demand, gdp_values)]
        final_energy_demand_df['energy_per_gdp'] = energy_per_gdp
        final_energy_demand_df['region'] = region
        return final_energy_demand_df

    else:
        final_energy_demand_df.to_csv('outputs/final_energy_demand' + str(categories) + '.csv', index=False)

    
# Function that gives the gini coefficient and SSP population for each scenario
def gini_between_countries(pyam_df, scenario_model_list, end_year, meta_df, gini_df, categories, regional=None):
    

    if regional is not None:
        region = regional
    else:
        region = 'World'

    # filter for the variables needed
    df = pyam_df.filter(variable='Emissions|CO2',
                        region=region,
                        year=range(2020, end_year+1),
                        scenario=scenario_model_list['scenario'], 
                        model=scenario_model_list['model'])
    

    if regional is not None:
        

        gini_df = Resilience.regional_gini
        # get list position for the region
        region_position = Data.R10.index(region)
        region_code = Data.R10_codes[region_position]
        region_gini = gini_df[region_code]
        print(region_gini)
        region_ssp_dict = {}
        count = 0
        for row in region_gini:
            region_ssp_dict['SSP' + str(count+1)] = region_gini[count]
            count += 1
        print(region_ssp_dict)
        ssp_ginis = region_ssp_dict
    else:
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
        # scenario_df = df.filter(scenario=scenario)
        # scenario_model_df = scenario_df.filter(model=model)
        scenario_ssp = meta_df[meta_df['model'] == model]
        scenario_model_ssp = scenario_ssp[scenario_ssp['scenario'] == scenario]    
        try:
            scenario_ssp = int(scenario_model_ssp['Ssp_family'].values[0])
        except ValueError:
            print('No SSP family found for the scenario')
            scenario_ssp = 2
            
        ssps.append(scenario_ssp)
        scenario_ssp_gini = ssp_ginis['SSP' + str(scenario_ssp)]
        ssp_gini_coefficients.append(scenario_ssp_gini)

    # create a new dataframe with the gini coefficients
    
    # create a new dataframe with the gini coefficients
    gini_df = pd.DataFrame({'model': scenario_model_list['model'], 
                            'scenario': scenario_model_list['scenario'], 
                            'ssp_gini_coefficient': ssp_gini_coefficients,
                            'ssp': ssps})
    
    if regional is not None:
        gini_df['region'] = region
        return gini_df
    
    else:
        gini_df.to_csv('outputs/gini_coefficient' + str(categories) +  '.csv', index=False)
        ssp_gini_coefficients = pd.DataFrame({'ssp': list(ssp_ginis.keys()), 
                                            'gini_coefficient': list(ssp_ginis.values())})



# calculate regional within region Gini by SSP
def get_within_region_gini(ssp_data, regions_breakdown, region_codes, start_year):

    """
    Inputs: 
    - SSP data (pop and GDP)
    - regional breakdown
    - region codes
    - start_year

    Outputs:
    - df with SSP and region gini coefficients

    """
    # prepare the data
    ssp_data['ISO3'] =  coco.convert(names = ssp_data['Region'], to='ISO3')
    ssp_gdp = ssp_data[ssp_data['Model'] == 'OECD ENV-Growth 2023'] # GDP|PPP  billion USD_2017/yr
    ssp_pop = ssp_data[ssp_data['Model'] == 'IIASA-WiC POP 2023'] # million

    # lists of the values 
    country_list = ssp_gdp['ISO3'].unique()
    processed_ISO3 = []
    processed_ssp = []
    output_gdp_per_capita = pd.DataFrame()

    # loop through all the ssps, countries
    for ssp in range(1, 6):

        
        ssp_gdp_selected = ssp_gdp[ssp_gdp['Scenario'] == 'SSP' + str(ssp)]
        ssp_pop_selected = ssp_pop[ssp_pop['Scenario'] == 'SSP' + str(ssp)]

        for country in country_list:

            gdp = ssp_gdp_selected[ssp_gdp_selected['ISO3'] == country]
            pop = ssp_pop_selected[ssp_pop_selected['ISO3'] == country]

            if not gdp.empty and not pop.empty:
                
                processed_ISO3.append(country)
                gdp.columns = gdp.columns.astype(str)
                pop.columns = pop.columns.astype(str)
                gdp_data = gdp.loc[:, '2025':'2100'].reset_index(drop=True)
                pop_data = pop.loc[:, '2025':'2100'].reset_index(drop=True)
                gdp_per_capita = gdp_data.div(pop_data)           
                output_gdp_per_capita = pd.concat([output_gdp_per_capita, gdp_per_capita], axis=0)
                processed_ssp.append(ssp)

    # form the output dataframe
    output_gdp_per_capita['ISO3'] = processed_ISO3
    output_gdp_per_capita['ssp'] = processed_ssp
    output_gdp_per_capita = output_gdp_per_capita.set_index('ISO3')
    
    # now add the regional breakdown
    regions_breakdown = regions_breakdown.set_index('iso3c')
    output_gdp_per_capita = output_gdp_per_capita.join(regions_breakdown['r10_iamc'], how='inner')
    # now calculate the gini coefficients
    # gini_coefficients = []

    output_ginis = pd.DataFrame()
    for ssp in range(1, 6):
        
        ssp_data = output_gdp_per_capita[output_gdp_per_capita['ssp'] == ssp]
        # list of ginis for each region for the SSP
        ssp_ginis = []
    
        # work through each of the R10 regions
        for region in region_codes:
            region_data = ssp_data [ssp_data ['r10_iamc'] == region]
            
            region_ssp_ginis = []
            for year in range(start_year, 2101, 5):

                year_data = region_data.loc[:, str(year)]
                # convert to numpy array
                year_data = np.array(year_data)
                region_ssp_ginis.append(gini(year_data)) 

            mean_region_gini = np.mean(region_ssp_ginis)
            ssp_ginis.append(mean_region_gini)    

        ssp_ginis.append(int(ssp))
        ssp_ginis = pd.DataFrame(ssp_ginis).T
        output_ginis = pd.concat([output_ginis, ssp_ginis], axis=0)

    # columns as region codes
    region_codes.append('ssp')
    output_ginis.columns = region_codes
    output_ginis.to_csv('outputs/within_region_gini.csv', index=False)







if __name__ == "__main__":
    main()
    