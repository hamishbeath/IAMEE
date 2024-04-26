import numpy as np
import pyam
import pandas as pd
# from utils import Utils
from utils import Data


class Robust:

    historic_emissions = pd.read_csv('inputs/historical_emissions_co2.csv')
    historic_emissions = historic_emissions.set_index('year')
    historic_emissions = historic_emissions['Emissions|CO2']
    remaining_carbon_budget_2030 =  250000  # MtCO2
    flexibility_data = pd.read_csv('inputs/build_life_times.csv')
    cdr_variables = ['Carbon Sequestration|CCS|Biomass', 'Carbon Sequestration|Land Use',
                     'Carbon Sequestration|Direct Air Capture']
    cdr_df = pyam.IamDataFrame("cat_df['C1', 'C2']CDR_Robustness.csv")
    low_carbon_energy_variables = variable=['Primary Energy|Nuclear','Primary Energy|Biomass', 
                                            'Primary Energy|Non-Biomass Renewables']
    

def main() -> None:

    # harmonize_emissions_calc_budgets(Data.dimensions_pyamdf, 'Emissions|CO2', Data.model_scenarios,
    #                      Robust.historic_emissions, 2023, Data.categories, False, 2050)
    # flexibility_score(Data.dimensions_pyamdf, Data.model_scenarios, 
    #                   2100, Data.energy_variables, Robust.flexibility_data, Data.categories)
    # calculate_total_CDR(Data.model_scenarios, Robust.cdr_df, 2051)
    # shannon_index_low_carbon_mix(Data.dimensions_pyamdf, Data.model_scenarios, 2100, Data.categories)
    empty_df = pd.DataFrame()
    for region in Data.R10:
        to_append = calculate_total_CDR(Data.model_scenarios, Robust.cdr_df, Data.regional_dimensions_pyamdf, 2050, regional=region)
        empty_df = pd.concat([empty_df, to_append], ignore_index=True, axis=0)
    empty_df.to_csv('outputs/total_CDR_regional' + str(Data.categories) + '.csv')
    # shannon_index_low_carbon_mix(Data.dimensions_pyamdf, Data.model_scenarios, 2100, Data.categories)


# calculate a flexibility score for the energy mix
def flexibility_score(pyam_df, scenario_model_list, 
                      end_year, energy_variables, 
                      flexibility_data,
                      categories, regional=None):

    # Check if a regional filter is applied
    if regional is not None:
        region = regional
    else:
        region = 'World'
        
    flexibility_data = flexibility_data.set_index('tech')
    flexibility_data = flexibility_data['flexibility_factor']
    
    # filter for the variables needed
    df = pyam_df.filter(variable=Data.energy_variables,
                        region=region,
                        year=range(2020, end_year+1),
                        scenario=scenario_model_list['scenario'], 
                        model=scenario_model_list['model'])

    flexibility_indexes = []
     # loop through models and scenarios
    for scenario, model in zip(scenario_model_list['scenario'], scenario_model_list['model']):

        # Filter out the data for the required scenario
        scenario_df = df.filter(scenario=scenario)
        scenario__model_df = scenario_df.filter(model=model)
                # create a dictionary to store the summed energy values
        energy_summed = {}
        total = 0
        # loop through the energy variables to interpolate and sum the values
        for variable in Data.energy_variables:
            
            # Filter out the data for the required variable
            variable_df = scenario__model_df.filter(variable=variable) 
            
            # make pandas series with the values and years as index
            variable_df = variable_df.data
            variable_series = pd.Series(variable_df['value'].values, index=variable_df['year'])
            cumulative_interpolated = pyam.timeseries.cumulative(variable_series, 2020, 2100)
            energy_summed[variable] = cumulative_interpolated
            total += cumulative_interpolated
        
        proportions = {}
        flexibility_total = 0
        for variable in Data.energy_variables:
            # variable string that removes the 'Primary Energy|' part of the string
            variable_string = variable[15:]
            proportion = energy_summed[variable] / total
            proportions[variable] = proportion
            flexibility_index_value = proportion * flexibility_data[variable_string]
            flexibility_total += flexibility_index_value
        
        flexibility_indexes.append(flexibility_total)
    
    # create a new dataframe with the flexibility scores
    flexibility_df = pd.DataFrame({'model': scenario_model_list['model'], 
                                   'scenario': scenario_model_list['scenario'], 
                                   'flexibility_score': flexibility_indexes})
    if regional is not None:
        flexibility_df['region'] = region
        return flexibility_df
    else:
        flexibility_df.to_csv('outputs/flexibility_scores' + str(categories) + '.csv', index=False)


"""
Code below based on code Robin Lamboll (2024) for harmonising data
https://www.nature.com/articles/s41558-023-01848-5

"""
# Harmonize a variable in a dataframe to match a reference dataframe
def harmonize_emissions_calc_budgets(df, var, scenario_model_list, 
                        harm_df, startyear, categories, offset=False, 
                        unity_year=int, regional=None):

    if regional is not None:
        region = regional

    else:
        region = 'World'

    """
    Implementation notes:
    1. determine the models median division of emissions for each region for 2023, interpolated
    2. using the share, take the total carbon budget and divide it accordingly to give regional carbon budget
    3. take the share and calculate a harmonised emissions trajectory based on the scenario emissions
    4. calculate the sum of emissions between 2023 and 2030 and the share of the remaining carbon budget
    """
    
    
    # Harmonises the variable var in the dataframe df to be equal to the values in the dataframe harmdf
    # for years before startyear up until unity_year. If offset is true, uses a linear offset tailing to 0 in unity_year.
    # If offset is false, uses a ratio correction that tends to 1 in unity_year
    harm_years = np.array([y for y in df.year if y>2005 and y<unity_year])

    carbon_budget_shares = []
    for scenario, model in zip(scenario_model_list['scenario'], scenario_model_list['model']): 
        ret = df.filter(variable=var, region='World', model=model, scenario=scenario, year=range(2005, 2100+1))

       # interpolate the data so that we have values for all years
        ret = ret.interpolate(range(2005, unity_year+1))        
        assert unity_year >= max(harm_years)
        canon2015 = harm_df.loc[startyear]
        ret = ret.timeseries()
        origval = ret[startyear].copy()

        # loop through the values in the input df and switch them to the harmonised values
        for y in [y for y in ret.columns if y<=startyear]:
            try:
                # canony = harm_df.filter(year=y, variable=var).data["value"][0]
                canony = harm_df.loc[y]
                ret[y] = canony
            except IndexError as e:
                print(f"We have only years {harm_df.filter(variable=var).year}, need {y}")

        # loop through the years and harmonise the values
        if not offset:
            fractional_correction_all = canon2015 / origval
            for year in [y for y in harm_years if y > startyear]:
                ret[year] *= (fractional_correction_all - 1) * (1 - (year - startyear) / (unity_year-startyear)) + 1
        else:
            offset_val = canon2015 - origval
            for year in [y for y in harm_years if y > startyear]:
                ret[year] += offset_val * (1 - (year - startyear) / (unity_year-startyear))
        
        output = pyam.IamDataFrame(ret)
        # calculate the sum of emissions between 2023 and 2030 and the share of the remaining carbon budget
        sum_emissions = output.filter(year=range(2023, 2031)).timeseries().sum().sum()
        carbon_budget_shares.append(sum_emissions / Robust.remaining_carbon_budget_2030)
    
    # create a new dataframe with the carbon budget shares
    carbon_budget_df = pd.DataFrame({'model': scenario_model_list['model'], 
                                     'scenario': scenario_model_list['scenario'], 
                                     'carbon_budget_share': carbon_budget_shares})

    carbon_budget_df.to_csv('outputs/carbon_budget_shares' + str(categories) + '.csv', index=False)



# total CDR by 2050 from BECCS, DACC or land-based CDR
def calculate_total_CDR(scenario_model_list, cdr_df, pyam_df,
                        end_year, regional=None):
    
    # Check if a regional filter is applied
    if regional is not None:
        region = regional
        # calculate the division indicators for the region
        division_df = pyam_df.filter(variable=['GDP|MER','Land Cover'], 
                                region=region, year=range(2020, end_year+1), 
                                scenario=scenario_model_list['scenario'], 
                                model=scenario_model_list['model'])
        gdp_values = []
        land_values = []
        for scenario, model in zip(scenario_model_list['scenario'], scenario_model_list['model']):
            gdp_scenario_df = division_df.filter(scenario=scenario, model=model, variable='GDP|MER')
            gdp_series = pd.Series(gdp_scenario_df['value'].values, index=gdp_scenario_df['year'])
            gdp_cumulative = pyam.timeseries.cumulative(gdp_series, 2020, end_year)
            gdp_values.append(gdp_cumulative)
            land_scenario_df = division_df.filter(scenario=scenario, model=model, variable='Land Cover')
            land_values.append(land_scenario_df['value'][0])
        # create a new dataframe with the land values
        division_basis_df = pd.DataFrame({'model': scenario_model_list['model'], 
                            'scenario': scenario_model_list['scenario'], 
                            'land_area': land_values,'gdp': gdp_values})
    else:
        region = 'World'

    cdr_df = cdr_df.filter(variable=Robust.cdr_variables,
                        region=region,
                        year=range(2020, end_year+1),
                        scenario=scenario_model_list['scenario'], 
                        model=scenario_model_list['model'])

    total_CDR_values = []
    for scenario, model in zip(scenario_model_list['scenario'], 
                               scenario_model_list['model']):
        scenario_df = cdr_df.filter(scenario=scenario, model=model)
        total_CDR = 0
        # add up cumulative CCS from bioenergy values
        beccs = scenario_df.filter(variable='Carbon Sequestration|CCS|Biomass')
        if beccs.empty:
            beccs = 0
        else:
            interpolated_beccs = beccs.interpolate(range(2020, end_year)).data.copy()
            beccs = interpolated_beccs['value'].sum()
        total_CDR += beccs

        # add up cumulative land-based CDR values
        land_use = scenario_df.filter(variable='Carbon Sequestration|Land Use')
        # land_use = land_use.as_pandas()
        if land_use.empty:
            print("No land based CDR data in AR6")
            land_use = Data.land_use_seq_data.filter(scenario=scenario, model=model, 
                                                     year=range(2020, end_year+1), region='World')
            land_use = land_use.filter(variable='Imputed|Carbon Sequestration|Land Use')
            # land_use = land_use.as_pandas()
            
            if land_use.empty:
                print("No land based CDR data in imputed file")
        if land_use.empty:
            land_use = 0
        
        else:
            land_use_interpolated = land_use.interpolate(range(2020, end_year)).data.copy() 
            land_use = land_use_interpolated['value'].sum()
        
        total_CDR += land_use
                
        # add up cumulative DACC values
        dacc = scenario_df.filter(variable='Carbon Sequestration|Direct Air Capture')
        if dacc.empty:
            print("No DACC data in AR6")
            dacc = 0
        else:
            dacc_interpolated = dacc.interpolate(range(2020, end_year)).data.copy()
            dacc = dacc_interpolated['value'].sum()
        total_CDR += dacc
            # dacc_series = pd.Series(dacc['value'].values, index=dacc['year'])
            # dacc_cumulative = pyam.timeseries.cumulative(dacc_series, 2020, end_year)
            # total_CDR += dacc_cumulative[end_year]

        total_CDR_values.append(total_CDR)
    
    if regional is not None:

        # divide the total CDR by the division basis
        total_CDR_gdp = total_CDR_values / division_basis_df['gdp'].values
        total_CDR_land = total_CDR_values / division_basis_df['land_area'].values

        # create a new dataframe with the total CDR values
        total_CDR_df = pd.DataFrame({'model': scenario_model_list['model'],
                                        'scenario': scenario_model_list['scenario'],
                                        'total_CDR': total_CDR_values,
                                        'total_CDR_gdp': total_CDR_gdp,
                                        'total_CDR_land': total_CDR_land})
        total_CDR_df['region'] = region
        return total_CDR_df
    else:
        # create a new dataframe with the total CDR values
        total_CDR_df = pd.DataFrame({'model': scenario_model_list['model'], 
                                    'scenario': scenario_model_list['scenario'], 
                                    'total_CDR': total_CDR_values})
        
        total_CDR_df.to_csv('outputs/total_CDR' + str(Data.categories) + '.csv', index=False)


# Function that calculates the shannon index for low-carbon energy mix for each scenario
def shannon_index_low_carbon_mix(pyam_df, scenario_model_list, end_year, categories, regional=None):

    # Check if a regional filter is applied
    if regional is not None:
        region = regional
    else:
        region = 'World'

    # filter for the variables needed
    df = pyam_df.filter(variable=Robust.low_carbon_energy_variables,
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
        for variable in Robust.low_carbon_energy_variables:
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
        for variable in Robust.low_carbon_energy_variables:
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
        shannon_df.to_csv('outputs/low_carbon_shannon_diversity_index' + str(categories) + '.csv', index=False)

# sub_function to calculate the regional emission shares for the categories
def calculate_regional_emission_share(regional_pyam_df, scenario_model_list, end_year):

    pass




if __name__ == "__main__":
    main()