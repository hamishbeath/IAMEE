import numpy as np
import pyam
import pandas as pd
import country_converter as coco
import math
from constants import *
from utils.file_parser import *

class Robust:

    historic_emissions = pd.read_csv(INPUT_DIR + 'historical_emissions_co2.csv')
    historic_emissions = historic_emissions.set_index('year')
    historic_emissions = historic_emissions['Emissions|CO2']
    
    R10_historical_emissions = pd.read_csv(INPUT_DIR + 'R10_emissions_historical.csv', index_col=0)
    R10_budgets = read_csv(INPUT_DIR + 'R10_carbon_budget_shares.csv')
    
    historic_population = pd.read_csv(INPUT_DIR + 'historical_population.csv')
    remaining_carbon_budget_2030 = 250000  # MtCO2 (global)
    flexibility_data = pd.read_csv(INPUT_DIR + 'build_life_times.csv')
    cdr_variables = ['Carbon Sequestration|CCS|Biomass', 'Carbon Sequestration|Land Use',
                     'Carbon Sequestration|Direct Air Capture']

    land_use_emissions_by_country = read_csv(INPUT_DIR + 'land_use_emissions_by_country.csv')
    territorial_emissions_by_country = read_csv(INPUT_DIR + 'export_emissions_by_country.csv')
    R10_emissions_historical = read_csv(INPUT_DIR + 'R10_emissions_historical.csv')


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

    harmonize_emissions_calc_budgets(pyamdf, 
                                    'Emissions|CO2', 
                                     scenarios,
                                    Robust.historic_emissions, 
                                    2023, 
                                    categories,
                                    REMAINING_2030_BUDGET, 
                                    False, 
                                    unity_year=2050, 
                                    regional=None) 
    flexibility_score(pyamdf, scenarios, 
                      2050, ENERGY_VARIABLES, Robust.flexibility_data, categories, regional=None)
    calculate_total_CDR(scenarios, pyamdf, 2051, regional=None)
    shannon_index_low_carbon_mix(pyamdf, scenarios, 2050, categories)
    

    # run the regional analysis
    print('Running regional analysis')
    if run_regional:
        
        flex_df = pd.DataFrame()
        shannon_df = pd.DataFrame()
        cdr_df = pd.DataFrame()
        
        for region in R10_CODES:
            to_append = flexibility_score(pyamdf, scenarios, 
                        2050, ENERGY_VARIABLES, Robust.flexibility_data, categories, regional=region)
            to_append_shannon = shannon_index_low_carbon_mix(pyamdf, scenarios, 2050, categories, regional=region)
            to_append_cdr = calculate_total_CDR(scenarios, pyamdf, 2050, regional=region)
            flex_df = pd.concat([flex_df, to_append], ignore_index=True, axis=0)
            shannon_df = pd.concat([shannon_df, to_append_shannon], ignore_index=True, axis=0)
            cdr_df = pd.concat([cdr_df, to_append_cdr], ignore_index=True, axis=0)
        
        flex_df.to_csv(OUTPUT_DIR + 'flexibility_scores_regional' + str(categories) + '.csv', index=False)
        shannon_df.to_csv(OUTPUT_DIR + 'low_carbon_shannon_diversity_index_regional' + str(categories) + '.csv', index=False)
        cdr_df.to_csv(OUTPUT_DIR + 'total_CDR_regional' + str(categories) + '.csv')
    
        run_regional_carbon_budgets(Robust.R10_historical_emissions, Robust.R10_budgets, pyamdf, scenarios, categories)


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
    df = pyam_df.filter(variable=energy_variables,
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
        for variable in energy_variables:
            
            # Filter out the data for the required variable
            variable_df = scenario__model_df.filter(variable=variable) 
            
            # make pandas series with the values and years as index
            variable_df = variable_df.data
            variable_series = pd.Series(variable_df['value'].values, index=variable_df['year'])
            cumulative_interpolated = pyam.timeseries.cumulative(variable_series, 2020, end_year)
            energy_summed[variable] = cumulative_interpolated
            total += cumulative_interpolated
        
        proportions = {}
        flexibility_total = 0
        for variable in energy_variables:
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
        flexibility_df.to_csv(OUTPUT_DIR + 'flexibility_scores' + str(categories) + '.csv', index=False)


"""
Code below based on code Robin Lamboll (2024) for harmonising data
https://www.nature.com/articles/s41558-023-01848-5

"""

def run_regional_carbon_budgets(R10_historical_emissions, R10_budgets, pyamdf, scenarios, categories):


    output_df = pd.DataFrame()    
    
    for region in R10_CODES:
        historical_emissions = R10_historical_emissions[region]
        
        # set column header to 'Emissions|CO2'
        historical_emissions.columns = ['Emissions|CO2']
        region_remaining_carbon_budget = R10_budgets[region] * REMAINING_2030_BUDGET
        region_remaining_carbon_budget = region_remaining_carbon_budget.values[0]

        to_append = harmonize_emissions_calc_budgets(pyamdf,
                                          'Emissions|CO2',
                                            scenarios,
                                            historical_emissions,
                                            2022,
                                            categories,
                                            region_remaining_carbon_budget,
                                            False,
                                            2050,
                                            regional=region)
        output_df = pd.concat([output_df, to_append], ignore_index=True, axis=0)
    
    output_df.to_csv(OUTPUT_DIR + 'carbon_budget_shares_regional' + str(Data.categories) + '.csv', index=False)


# Harmonize a variable in a dataframe to match a reference dataframe
def harmonize_emissions_calc_budgets(df, var, scenario_model_list, 
                        harm_df, startyear, categories, budget, offset=False, 
                        unity_year=int, regional=None):

    if regional is not None:
        region = regional

    else:
        region = ['World']

    # Harmonises the variable var in the dataframe df to be equal to the values in the dataframe harmdf
    # for years before startyear up until unity_year. If offset is true, uses a linear offset tailing to 0 in unity_year.
    # If offset is false, uses a ratio correction that tends to 1 in unity_year
    harm_years = np.array([y for y in df.year if y>2005 and y<unity_year])
    carbon_budget_shares = []
    for scenario, model in zip(scenario_model_list['scenario'], scenario_model_list['model']): 
        
        ret = df.filter(variable=var, region=region, model=model, scenario=scenario, year=range(2005, 2100+1))
       # interpolate the data so that we have values for all years
        ret = ret.interpolate(range(2005, unity_year+1))    
        assert unity_year >= max(harm_years)
        canon2015 = harm_df.loc[startyear]
        ret = ret.timeseries()
        origval = ret[startyear].copy()

        # loop through the values in the input df and switch them to the harmonised values
        for y in [y for y in ret.columns if y <= startyear]:
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
        carbon_budget_share = sum_emissions / budget
        carbon_budget_shares.append(sum_emissions / budget)

    # create a new dataframe with the carbon budget shares
    carbon_budget_df = pd.DataFrame({'model': scenario_model_list['model'], 
                                     'scenario': scenario_model_list['scenario'], 
                                     'carbon_budget_share': carbon_budget_shares})
    if regional is not None:
        carbon_budget_df['region'] = region
        return carbon_budget_df
    
    else:
        carbon_budget_df.to_csv(OUTPUT_DIR + 'carbon_budget_shares' + str(categories) + '.csv', index=False)


# total CDR by 2050 from BECCS, DACC or land-based CDR
def calculate_total_CDR(scenario_model_list, pyam_df,
                        end_year, regional=None):
    
    # Check if a regional filter is applied
    if regional is not None:
        
        print('Calculating the total CDR for the region', regional)
        region = regional
        # calculate the division indicators for the region
        division_df = pyam_df.filter(variable=['GDP|MER','Land Cover'], 
                                region=[region,'World'], year=range(2020, end_year+1), 
                                scenario=scenario_model_list['scenario'], 
                                model=scenario_model_list['model'])        
        gdp_proportion_values = []
        land_proportion_values = []
        gdp = []
        land_area = []
        for scenario, model in zip(scenario_model_list['scenario'], scenario_model_list['model']):
            
            # get both the regional and world GDP values
            gdp_scenario_df = division_df.filter(scenario=scenario, model=model, variable='GDP|MER', region=region)
            world_gdp_scenario_df = division_df.filter(scenario=scenario, model=model, variable='GDP|MER', region='World')
            gdp_series = pd.Series(gdp_scenario_df['value'].values, index=gdp_scenario_df['year'])
            world_gdp_series = pd.Series(world_gdp_scenario_df['value'].values, index=gdp_scenario_df['year'])
            gdp_cumulative = pyam.timeseries.cumulative(gdp_series, 2020, end_year)
            world_gdp_cumulative = pyam.timeseries.cumulative(world_gdp_series, 2020, end_year)
            gdp_proportion_values.append(gdp_cumulative / world_gdp_cumulative)
            gdp.append(gdp_cumulative)

            land_scenario_df = division_df.filter(scenario=scenario, model=model, variable='Land Cover', region=region)
            world_land_scenario_df = division_df.filter(scenario=scenario, model=model, variable='Land Cover', region='World')
            land_proportion_values.append(land_scenario_df['value'][0]/world_land_scenario_df['value'][0])
            land_area.append(land_scenario_df['value'][0])

        # create a new dataframe with the land values
        division_basis_df = pd.DataFrame({'model': scenario_model_list['model'], 
                            'scenario': scenario_model_list['scenario'], 
                            'land_area_share': land_proportion_values,'gdp_share': gdp_proportion_values,
                            'land_area': land_area, 'gdp': gdp})
    else:
        region = 'World'

    cdr_df = pyam_df.filter(variable=Robust.cdr_variables,
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

        land_use_interpolated = land_use.interpolate(range(2020, end_year)).data.copy() 
        
        land_use = land_use_interpolated['value'].sum()

        total_CDR += land_use
                
        # add up cumulative DACC values
        dacc = scenario_df.filter(variable='Carbon Sequestration|Direct Air Capture')
        if dacc.empty:
            dacc = 0
        else:
            dacc_interpolated = dacc.interpolate(range(2020, end_year)).data.copy()
            dacc = dacc_interpolated['value'].sum()
        total_CDR += dacc
        

        total_CDR_values.append(total_CDR)
    
    print('Warning: DACC values are not available for all scenarios. Scenarios that report it will be included in the total CDR calculation')
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
        
        total_CDR_df.to_csv(OUTPUT_DIR + 'total_CDR' + str(Data.categories) + '.csv', index=False)


# Function that calculates the shannon index for low-carbon energy mix for each scenario
def shannon_index_low_carbon_mix(pyam_df, scenario_model_list, end_year, categories, regional=None):

    # Check if a regional filter is applied
    if regional is not None:
        region = regional
    else:
        region = 'World'

    # filter for the variables needed
    df = pyam_df.filter(variable=LOW_CARBON_ENERGY_VARIABLES,
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
        for variable in LOW_CARBON_ENERGY_VARIABLES:
            # Filter out the data for the required variable
            variable_df = scenario__model_df.filter(variable=variable) 
            # make pandas series with the values and years as index
            variable_df = variable_df.data
            variable_series = pd.Series(variable_df['value'].values, index=variable_df['year'])
            cumulative_interpolated = pyam.timeseries.cumulative(variable_series, 2020, end_year)
            energy_summed[variable] = cumulative_interpolated
            total += cumulative_interpolated
        
        # make a new dictionary to store the proportions of the energy sources 
        #  and calculate the shannon index
        proportions = {}
        shannon_total = 0
        for variable in LOW_CARBON_ENERGY_VARIABLES:
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
        shannon_df.to_csv(OUTPUT_DIR + 'low_carbon_shannon_diversity_index' + str(categories) + '.csv', index=False)



# Function that calculates the remaining carbon budgets for R10 regions in 2023
def get_regional_level_remaining_budgets(emissions_by_country, 
                                         land_use_emissions,
                                         population_by_country, 
                                         regional_breakdown, end_year):
    """
    Function that uses historical emissions and historical population at the 
    country level to calculate regional remaining carbon budgets for 2023. 
    
    Inputs:
    - historical emissions data per country
    - historical population data per country
    - breakdown of regions into countries
    
    Outputs:
    - csv file with the remaining carbon budgets for each region

    """
    # filter years between 1990 to 2023
    population_by_country = population_by_country[(population_by_country['Time'] >= 1990) & (population_by_country['Time'] < 2023)]
    # restructure territorial emissions file
    emissions_by_country = emissions_by_country.set_index('Territorial emissions in MtCO₂')
    emissions_by_country = emissions_by_country.T
    emissions_by_country['ISO3'] = coco.convert(names = emissions_by_country.index, to='ISO3')

    # restructure land use emissions file
    land_use_emissions = land_use_emissions[land_use_emissions['Year'] >= 1990]

    failed = []
    total_emissions = []
    total_population = []
    emissions_1990 = []
    population_1990 = []
    country_worked = []
    # calculate cumulative emissions and population for each country
    country_list = emissions_by_country['ISO3'].unique().tolist()
    emissions_df = pd.DataFrame()
    emissions_out = []
    for country in country_list:

        try:
            # deal with emissions, resulting in cumulative emissions since 1990 (MtCO2)
            country_emissions = emissions_by_country[emissions_by_country['ISO3'] == country]
            country_emissions = country_emissions.drop(columns=['ISO3'])
            land_country_emissions = land_use_emissions[land_use_emissions['Code'] == country]
            land_country_emissions = land_country_emissions.set_index('Year').T
            land_country_emissions = land_country_emissions[land_country_emissions.index == 'Annual CO₂ emissions from land-use change']
            total_country_emissions = country_emissions.values + (land_country_emissions.values/1000000)
            total_country_emissions_cum = np.sum(total_country_emissions)
            emissions_df = pd.concat([emissions_df, country_emissions], axis=0)
            emissions_out.append(country)
            # check for nan values and skip otherwise
            if not math.isnan(total_country_emissions_cum):
                total_emissions.append(total_country_emissions_cum)
            else:
                failed.append(country)
                continue # skip the country if there are nan values
        except ValueError as e:
            # after error, try to do it without the land use emissions
            total_country_emissions_cum = np.sum(country_emissions.values)
            if not math.isnan(total_country_emissions_cum):
                total_emissions.append(total_country_emissions_cum)
                emissions_df = pd.concat([emissions_df, country_emissions], axis=0)
                emissions_out.append(country)
            else:
                failed.append(country)
                continue 
        
        # deal with the population by country, calculate 
        country_population = population_by_country[population_by_country['ISO3_code'] == country]    
        country_population_1990 = country_population[country_population['Time'] == 1990]['TPopulation1Jan'].values[0]
        total_country_population = country_population['TPopulation1Jan'].sum()
        if not math.isnan(total_country_population):
            total_population.append(total_country_population)
            population_1990.append(country_population_1990)
        else:
            failed(country)
            continue
        country_worked.append(country)

    # new df
    country_emissions_population = pd.DataFrame({'ISO3': country_worked, 
                                                'total_emissions': total_emissions,
                                                'total_person_years': total_population,
                                                'population_1990': population_1990})
    country_emissions_population['Cumulative emissions per capita'] = country_emissions_population['total_emissions'] / country_emissions_population['total_person_years']
    max_cumulative_emissions_per_capita = country_emissions_population['Cumulative emissions per capita'].max()
    country_emissions_population['normalised_per_capita_cumulative_emissions'] = country_emissions_population['Cumulative emissions per capita'] / max_cumulative_emissions_per_capita
    country_emissions_population['adjustment_factors'] = 1 / country_emissions_population['normalised_per_capita_cumulative_emissions'] 
    total_adjustment = country_emissions_population['adjustment_factors'].sum()
    country_emissions_population['normalised_adjustment_factors'] = country_emissions_population['adjustment_factors'] / total_adjustment
    country_emissions_population.set_index('ISO3', inplace=True)    

    R10_budget_shares = {}
    
    # calculate the regional breakdown for R10
    regional_breakdown = regional_breakdown.set_index('iso3c')
    regional_breakdown = regional_breakdown.join(country_emissions_population['normalised_adjustment_factors'], how='left')
    
    for region in Data.R10_codes:
        region_countries = regional_breakdown[regional_breakdown['r10_iamc'] == region]
        region_budget_share = region_countries['normalised_adjustment_factors'].sum()
        R10_budget_shares[region] = region_budget_share

    R10_budget_shares = pd.DataFrame(R10_budget_shares, index=[0])
    R10_budget_shares.to_csv('inputs/R10_carbon_budget_shares.csv', index=False)

    emissions_df['ISO3'] = emissions_out    
    regional_breakdown = regional_breakdown.join(emissions_df.set_index('ISO3'), how='left')

    # calculate the timeseries of emissions for R10 regions 2005-2022
    R10_emissions = pd.DataFrame()
    for region in Data.R10_codes:
        region_countries = regional_breakdown[regional_breakdown['r10_iamc'] == region]
        region_countries.columns = region_countries.columns.astype(str)
        region_countries_data = region_countries.loc[:, '2005':'2022']
        region_emissions = region_countries_data.sum()
        R10_emissions = pd.concat([R10_emissions, region_emissions], axis=1)
    
    # set column names as the R10 codes
    R10_emissions.columns = Data.R10_codes
    R10_emissions.to_csv(INPUT_DIR + 'R10_emissions_historical.csv')


if __name__ == "__main__":
    main()


