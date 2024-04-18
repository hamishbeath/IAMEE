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

def main() -> None:

    # harmonize_emissions_calc_budgets(Data.dimensions_pyamdf, 'Emissions|CO2', Data.model_scenarios,
    #                      Robust.historic_emissions, 2023, Data.categories, False, 2050)
    # flexibility_score(Data.dimensions_pyamdf, Data.model_scenarios, 
    #                   2100, Data.energy_variables, Robust.flexibility_data, Data.categories)
    calculate_total_CDR(Data.model_scenarios, Robust.cdr_df, 2051)

# calculate a flexibility score for the energy mix
def flexibility_score(pyam_df, scenario_model_list, 
                      end_year, energy_variables, 
                      flexibility_data,
                      categories):

    flexibility_data = flexibility_data.set_index('tech')
    flexibility_data = flexibility_data['flexibility_factor']
    
    # filter for the variables needed
    df = pyam_df.filter(variable=Data.energy_variables,
                        region='World',
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

    # print(flexibility_df)
    flexibility_df.to_csv('outputs/flexibility_scores' + str(categories) + '.csv', index=False)

"""
Code below based on code Robin Lamboll (2024) for harmonising data
https://www.nature.com/articles/s41558-023-01848-5

"""
# Harmonize a variable in a dataframe to match a reference dataframe
def harmonize_emissions_calc_budgets(df, var, scenario_model_list, 
                        harm_df, startyear, categories, offset=False, 
                        unity_year=int):
    # harm_df is a dataframe with the actual values to harmonise to
    # df is the pyamdataframe to harmonise
    
    
    # Harmonises the variable var in the dataframe df to be equal to the values in the dataframe harmdf
    # for years before startyear up until unity_year. If offset is true, uses a linear offset tailing to 0 in unity_year.
    # If offset is false, uses a ratio correction that tends to 1 in unity_year
    harm_years = np.array([y for y in df.year if y>2005 and y<unity_year])

    carbon_budget_shares = []
    for scenario, model in zip(scenario_model_list['scenario'], scenario_model_list['model']): 
        ret = df.filter(variable=var, region='World',model=model, scenario=scenario, year=range(2005, 2100+1))

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
def calculate_total_CDR(scenario_model_list, cdr_df,
                        end_year):
    if end_year != 2051:
        raise ValueError("Indicator for 2050 values")
    # # filter for the variables needed
    # df = pyam_df.filter(variable=Robust.cdr_variables,
    #                     region='World',
    #                     year=range(2020, end_year+1),
    #                     scenario=scenario_model_list['scenario'], 
    #                     model=scenario_model_list['model'])
    
    cdr_df = cdr_df.filter(variable=Robust.cdr_variables,
                        region='World',
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
        beccs = beccs.as_pandas()
        # add all the values in the values column
        total_CDR += beccs['value'].sum()

        # add up cumulative land-based CDR values
        land_use = scenario_df.filter(variable='Carbon Sequestration|Land Use')
        land_use = land_use.as_pandas()
        if land_use.empty:
            print("No land based CDR data in AR6")
            land_use = Data.land_use_seq_data.filter(scenario=scenario, model=model, year=range(2020, end_year), region='World')
            land_use = land_use.filter(variable='Imputed|Carbon Sequestration|Land Use')
            land_use = land_use.as_pandas()
            
            if land_use.empty:
                print("No land based CDR data in imputed file")
        total_CDR += land_use['value'].sum()
                
        # add up cumulative DACC values
        dacc = scenario_df.filter(variable='Carbon Sequestration|Direct Air Capture')
        dacc = dacc.as_pandas()
        if dacc.empty:
            print("No DACC data in AR6")
        total_CDR += dacc['value'].sum()
            # dacc_series = pd.Series(dacc['value'].values, index=dacc['year'])
            # dacc_cumulative = pyam.timeseries.cumulative(dacc_series, 2020, end_year)
            # total_CDR += dacc_cumulative[end_year]

        total_CDR_values.append(total_CDR)
    
    # create a new dataframe with the total CDR values
    total_CDR_df = pd.DataFrame({'model': scenario_model_list['model'], 
                                 'scenario': scenario_model_list['scenario'], 
                                 'total_CDR': total_CDR_values})
    
    total_CDR_df.to_csv('outputs/total_CDR' + str(Data.categories) + '.csv', index=False)


if __name__ == "__main__":
    main()