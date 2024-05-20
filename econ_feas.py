import numpy as np
import pyam
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import country_converter as coco
from matplotlib import rcParams
from utils import Data
from utils import Utils
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# Defines simple weighting system for indicators of sustainability, adjusting the weights will change the spread of how scenarios 
# within each temperature category will score. The idea is to tease out tradeoffs and synergies. 
class EconFeas:
    
# import the scenario data for C1, C2, C3 and C4

    # connAr6 = pyam.iiasa.Connection(name='ar6-public', 
    #                                 creds=None, 
    #                                 auth_url='https://api.manager.ece.iiasa.ac.at')

    categories = ['C1', 'C2', 'C3', 'C4', 'C5']
    # variable=['Water Consumption', 'Land Cover|Pasture', 'Land Cover|Forest', 
    #           'Land Cover', 'Land Cover|Cropland', 'Land Cover|Cropland', 
    #           'Ecotoxicity|Marine|Electricity', 'Material recycling|Plastics']
    variables=['GDP|PPP', 'Investment', 'Investment|Energy Supply']

    # plotting_variables = ['Water Consumption', 'Land Cover|Pasture', 'Land Cover|Forest',
    #                       'Land Cover|Cropland', 'Land Cover|Cropland|Energy Crops']
                            
    plotting_categories = ['C1', 'C3', 'C5']
    # plotting_categories = ['C3']
    plotting_category_colours = {'C1':'#f57200', 'C3':'#6302d9', 'C5':'#1b9e77'}
    # violin_colours = ['#f57200','#6302d9','#1b9e77']
    econ_scenarios = pd.read_csv('econ_world_World.csv')
    # econ_data = pyam.IamDataFrame(data="cat_meta['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'].csv")
    run_mode = 'cat'
    alpha = -0.037
    beta = -0.0018
    warming_variable = 'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile' 
    present_warming = 1.25
    # AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|95.0th Percentile
    iea_country_groupings = pd.read_csv('econ/iea_country_groupings.csv')
    by_country_gdp = pd.read_csv('econ/IMF_GDP_data_all_countries.csv')

def main() -> None:

    # data_download()
    # plot_outputs()
    # plot_using_pyam()
    # violin_plots()
    # assess_variable_data()
    # energy_supply_investment_score(Data.dimensions_pyamdf, 0.023, 2100, Data.model_scenarios, Data.categories)
    # energy_supply_investment_analysis(0.023, 2100, EconFeas.econ_scenarios, apply_damages=None)
    # map_countries_to_regions(EconFeas.iea_country_groupings, EconFeas.by_country_gdp)
    # scenarios_list = pd.read_csv('econ_regional_Countries of Sub-Saharan Africa.csv')
    # data = pyam.IamDataFrame(data="pyamdf_econ_data_R10['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'].csv")
    output_df = pd.DataFrame()
    for region in Data.R10:
        print(region)
        to_append = energy_supply_investment_score(Data.regional_dimensions_pyamdf, 0.023, 2100, Data.model_scenarios, Utils.categories, regional=region)
        output_df = pd.concat([output_df, to_append], ignore_index=True, axis=0)
    output_df.to_csv('outputs/energy_supply_investment_score_regional' + str(Utils.categories) + '.csv')
    # energy_supply_investment_score(Data.dimensions_pyamdf, 0.023, 2100, Data.model_scenarios, Data.categories, regional='World')




# takes as an input a Pyam dataframe object with n number of scenarios in it. For each scenario it calculates both a binary 
def energy_supply_investment_score(pyam_df, base_value, end_year, scenario_model_list, categories, regional=None):

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
        try:
            regional_thresholds = pd.read_csv('econ/energy_investment_regions.csv')
        except FileNotFoundError:
            print('Regional investment file not found. Please ensure regional investment thresholds file is available.')

        # Filter out the data for the required region
        regional_threshold_data = regional_thresholds[regional_thresholds['region'] == region]

        # calculate the mean of the columns 2015 to 2023
        base_value = regional_threshold_data.iloc[:, 3:12].mean(axis=1)
        print(base_value)
    else:
        region = 'World'

    # Filter out the data for the required variables
    df = pyam_df.filter(variable=['Investment|Energy Supply','GDP|MER'],
                        region=region, year=range(2020, end_year+1),
                        scenario=scenario_model_list['scenario'],
                        model=scenario_model_list['model'])
    print(df)
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
        output_df.to_csv('outputs/energy_supply_investment_score' + str(categories) + '.csv')

    

def energy_supply_investment_analysis(base_value, end_year, scenario_model_list, 
                                      apply_damages=None):

    """
    Function similar to the above but that has additional indicator and different
    outputs required for analysis beyond the assessment framework.
    """
    # connAr6 = pyam.iiasa.Connection(name='ar6-public', 
    #                 creds=None, 
    #                 auth_url='https://api.manager.ece.iiasa.ac.at') 
    # Filter out the data for the required variables
    df = pyam.IamDataFrame(data="cat_df['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']econ_world.csv")
    
    df = df.filter(variable=['Investment|Energy Supply','GDP|MER', EconFeas.warming_variable], 
                        region='World', year=range(2020, end_year+1), 
                        scenario=scenario_model_list['scenario'], 
                        model=scenario_model_list['model'])
    
    meta_data = pd.read_csv("cat_meta['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']econ_world.csv")

    # lists of the mean values and temp categories 
    # to be appended as columns to the dataframe
    mean_value_list = []
    mean_value_2050_list = []
    temperature_category_list = []
    largest_annual_increase_list = []
    mean_annual_increase_list = []
    # make a dataframe to store the results with scenario and model 
    output_df = pd.DataFrame()
    # output_df['scenario'] = scenario_model_list['scenario']
    # output_df['model'] = scenario_model_list['model']

    # get list of years between 2020 and 2100 at decedal intervals
    year_list = list(range(2030, end_year+1, 10))

    # make a column for each year in year list
    for year in year_list:
        output_df[str(year)] = []
    
    # print(output_df)

    # loop through the scenario model list
    for scenario, model in zip(scenario_model_list['scenario'], scenario_model_list['model']):

         # Filter out the data for the required scenario
        scenario_df = df.filter(scenario=scenario)
        scenario__model_df = scenario_df.filter(model=model)

        # # make pandas series with the values and years as index
        variable_df = scenario__model_df.filter(variable='Investment|Energy Supply')
        # variable_series = pd.Series(variable_df['value'].values, index=variable_df['year'])
        variable_df = variable_df.interpolate(range(2020, 2101))
        # make a list of the values
        investment_values = list(variable_df['value'].values)     

        # apply climate damages if the damages variable != None
        if apply_damages != None:
            # scenario__model_df.to_csv('outputs/gdp_untreated.csv')
            scenario__model_df = apply_damage_function(scenario__model_df,
                                                       scenario, model, 
                                                       EconFeas.warming_variable,
                                                       EconFeas.alpha,
                                                       EconFeas.beta,
                                                       EconFeas.present_warming, 
                                                       meta_data, year_list)

        # make lists to calculate the results from
        share_of_gdp = {}
        # model_year_list = []
        # proportion_of_base = []
        # investment_values = []
        # Iterate through the years
        for year in year_list:    
 
            # Filter out the data for the required year
            year_df = scenario__model_df.filter(year=year)
            year_df_new = year_df.data
            investment = year_df['value'][year_df['variable'] == 'Investment|Energy Supply'].values
            gdp = year_df['value'][year_df['variable'] == 'GDP|MER'].values

            # Calculate the share of GDP that is invested in energy supply
            year_share = investment[0] / gdp[0]
            share_of_gdp[year] = year_share
            # investment_values.append(investment[0])
        
        
        # Calculate the mean value of the share of GDP that is invested in energy supply
        list_of_values = list(share_of_gdp.values())
        mean_value_list.append(np.mean(list_of_values))

        # Calculate the mean value up to 2050
        list_of_values_2050 = list_of_values[:3]
        mean_value_2050_list.append(np.mean(list_of_values_2050))
        
        # from all the investment values, calculate the largest percentage increase
        # from the previous year
        max_increase_total = 0
        max_increase = 0
        for i in range(1, len(investment_values)):
            increase = (investment_values[i] - investment_values[i-1]) / investment_values[i-1]
            if increase > max_increase:
                max_increase = increase
            max_increase_total += increase
        
        # calculate the mean annual increase
        mean_increase = max_increase_total / len(investment_values)
        largest_annual_increase_list.append(max_increase)
        mean_annual_increase_list.append(mean_increase)

        values_list = list(share_of_gdp.values())
        years_list = list(share_of_gdp.keys())
        to_append_df = pd.DataFrame()
        to_append_df['year'] = years_list
        to_append_df['value'] = values_list
        to_append_df = to_append_df.T

        # make the years the column headers
        to_append_df.columns = to_append_df.iloc[0].astype(int).astype(str)
        to_append_df = to_append_df[1:]

        # append the values to the output dataframe
        output_df = pd.concat([output_df, to_append_df], axis=0)

        # get the temperature category for the scenario
        temp_category = meta_data[meta_data['scenario'] == scenario]
        temp_category = temp_category[temp_category['model'] == model]
        temperature_category_list.append(temp_category['Category'].values[0])

    output_df = output_df.reset_index(drop=True)
    output_df['mean_value'] = mean_value_list
    output_df['mean_value_2050'] = mean_value_2050_list
    output_df['largest_annual_increase'] = largest_annual_increase_list
    output_df['mean_annual_increase'] = mean_annual_increase_list
    output_df['temperature_category'] = temperature_category_list
    output_df['scenario'] = scenario_model_list['scenario']
    output_df['model'] = scenario_model_list['model']

    # move the scenario and model columns to the front
    cols = output_df.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    output_df = output_df[cols]
    if apply_damages != None:
        output_df.to_csv('outputs/energy_supply_investment_analysis_damages' + 
                         EconFeas.warming_variable + '.csv')
    else:
        output_df.to_csv('outputs/energy_supply_investment_analysis.csv')
    

def apply_damage_function(gdp_untreated, 
                          scenario, model, 
                          warming_variable, 
                          alpha, beta, 
                          present_warming, 
                          meta_data,
                          year_list):
    """
    This function takes GDP data, scenario name and model as an input, warming variable, 
    and then applies a damage function to the GDP data to calculate the economic impact of
    the warming.

    Inputs: 
    - Pyam dataframe with the GDP 
    - Scenario name and model
    - Warming variable to use (e.g. MAGICC or FaIR)
    - alpha 
    - beta
    - metadata
    Outputs: 
    - GDP timeseries treated for damages 
    """
    # check in metadata whether climate impacts have been accounted for
    damages_accounted_for = meta_data[meta_data['scenario'] == scenario]
    damages_accounted_for = damages_accounted_for[damages_accounted_for['model'] == model]
    if damages_accounted_for['Climate impacts'].values[0] == 'yes':
        return gdp_untreated
    else:
        # loop through each of the years to calculate the temperature difference 
        # and apply the damage function
        gdp_treated = gdp_untreated.as_pandas()
        for year in year_list:
            # print(year)
            # print(type(gdp_untreated))
            year_df = gdp_untreated.filter(year=year)
            year_df_pd = year_df.as_pandas()
            warming = year_df_pd['value'][year_df_pd['variable'] == warming_variable].values
            gdp = year_df_pd['value'][year_df_pd['variable'] == 'GDP|MER'].values
            decade_warming = warming - present_warming
            damage = (alpha + (beta * present_warming)) * decade_warming + (beta/2) * decade_warming**2
            new_gdp = gdp * (1 + damage)
            # update the GDP value in the dataframe for the year
            gdp_treated.loc[(gdp_treated['year'] == year) & 
                            (gdp_treated['variable'] == 'GDP|MER'), 'value'] = new_gdp
        # convert back to a Pyam dataframe and drop exclude column
        gdp_treated = pyam.IamDataFrame(data=gdp_treated)
        # gdp_treated.to_csv('outputs/gdp_treated.csv')
        return gdp_treated


def map_countries_to_regions(country_groups, country_data):

    # print(country_groups)
    # print(country_data)
    output_dict = {}
    # get the list groups of countries
    country_groups_list = country_groups['group'].unique().tolist()
    country_data_countries = country_data['country'].unique().tolist()
    
    output_df = pd.DataFrame()

    for group in country_groups_list:

        group_to_append = pd.DataFrame()
        # get the countries in the group
        countries = country_groups[country_groups['group'] == group]
        countries_list = countries['countries'].unique().tolist()
        # extract list items from the string, broken up by commas
        countries_list = [i.split(', ') for i in countries_list]
        countries_list = [item for sublist in countries_list for item in sublist]
        countries_list = [i.split(' and ') for i in countries_list]
        countries_list = [item for sublist in countries_list for item in sublist]
        # convert to ISO3 codes
        countries_list = coco.convert(names=countries_list, to='ISO3')
        # append the countries to the output dictionary
        output_dict[group] = countries
        for country in country_data_countries:
            # convert to ISO3 codes
            country_iso3 = coco.convert(names=country, to='ISO3')
            # check if the country is in the list of countries
            if country_iso3 in countries_list:
                country_row = country_data[country_data['country'] == country]
                group_to_append = pd.concat([group_to_append, country_row], axis=0)
        group_to_append['group'] = group
        output_df = pd.concat([output_df, group_to_append], axis=0)
    
    print(output_df)
    output_df.to_csv('econ/country_groupings_IMF.csv')
                



    print(output_dict)


    # print(output_dict)


if __name__ == "__main__":
    main()