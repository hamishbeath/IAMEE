import numpy as np
import pyam
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import country_converter as coco
from matplotlib import rcParams
from utils import Data
from utils import Utils
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']
plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.major.left'] = True
plt.rcParams['ytick.major.right'] = True
plt.rcParams['ytick.minor.visible'] = True
#plt.rcParams['ytick.labelright'] = True
#plt.rcParams['ytick.major.size'] = 0
#plt.rcParams['ytick.major.pad'] = -56
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

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
                            
    plotting_categories = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
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

    cb_friendly_cat_colours = {'C1':'#332288', 
                               'C2':'#117733', 
                               'C3':'#44AA99', 
                               'C4':'#88CCEE', 
                               'C5':'#DDCC77', 
                               'C6':'#CC6677', 
                               'C7':'#AA4499', 
                               'C8':'#882255'}


class EconData:

    # Global Data
    global_scenarios_models = pd.read_csv('econ/scenarios_models_world.csv')
    global_econ_pyamdf = pyam.IamDataFrame(data="econ/scenarios_models_worldcat_df['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'].csv")
    global_econ_meta = pd.read_csv("econ/meta_data_c1_c8.csv")
    global_econ_results_no_damages = pd.read_csv('econ/energy_supply_investment_analysis.csv')
    # global_econ_results_damages = pd.read_csv('econ/energy_supply_investment_analysis_damages' + EconFeas.warming_variable + '.csv')        

    # Regional Data
    regional_scenarios_models = pd.read_csv('econ/scenarios_models_R10.csv')
    regional_econ_pyamdf = pyam.IamDataFrame(data='econ/pyamdf_econ_analysis_R10.csv')
    regional_damage_estimates = pd.read_csv('econ/R10_damages_temps.csv')
    regional_categories = ['C1', 'C2', 'C3', 'C4']


def main() -> None:

    # data_download()
    # plot_outputs()
    # plot_using_pyam()
    # violin_plots()
    # assess_variable_data()
    # energy_supply_investment_score(Data.dimensions_pyamdf, 0.023, 2100, Data.model_scenarios, Data.categories)
    # energy_supply_investment_analysis(0.023, 2100, EconData.global_scenarios_models, 
    #                                   EconData.global_econ_pyamdf, 
    #                                   EconData.global_econ_meta,
    #                                   apply_damages=None)
    # map_countries_to_regions(EconFeas.iea_country_groupings, EconFeas.by_country_gdp)
    # scenarios_list = pd.read_csv('econ_regional_Countries of Sub-Saharan Africa.csv')
    # data = pyam.IamDataFrame(data="pyamdf_econ_data_R10['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'].csv")
    output_df = pd.DataFrame()
    damages = None
    for region in Data.R10:
        print(region)
        to_append = energy_supply_investment_analysis(0.023, 2100, EconData.regional_scenarios_models, 
                                                      EconData.global_econ_pyamdf, 
                                                      EconData.global_econ_meta,
                                                        apply_damages=damages, 
                                                        region=region,
                                                        region_codes=Data.R10_codes,
                                                        df_regional=EconData.regional_econ_pyamdf,
                                                        R10_temps_df=EconData.regional_damage_estimates, 
                                                        included_categories=EconData.regional_categories)
        
        output_df = pd.concat([output_df, to_append], ignore_index=True, axis=0)
    if damages != None:
        output_df.to_csv('econ/energy_supply_investment_analysis_R10' + damages + '.csv')
    else:
        output_df.to_csv('econ/energy_supply_investment_analysis_R10.csv')
    
    # output_df.to_csv('outputs/energy_supply_investment_score_regional' + str(Utils.categories) + '.csv')
    # # energy_supply_investment_score(Data.regional_dimensions_pyamdf, 0.023, 2100, Data.model_scenarios, Data.categories, regional=None)
    # temporal_subplots(EconData.global_econ_results_no_damages, EconData.global_econ_results_damages, EconFeas.plotting_categories, 'mean_value')
    # calculate_R10_damages_temps(Data.R10_codes, 'default', 1.3, 2.5, Data.region_country_df)


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

    
def energy_supply_investment_analysis(base_value, end_year, scenario_model_list, df_main, meta_data,
                                      apply_damages=None, region=None, region_codes=None, 
                                      df_regional=None, R10_temps_df=None, included_categories=None):

    """
    Function similar to the above but that has additional indicator and different
    outputs required for analysis beyond the assessment framework.
    """

    if region is not None:
        df = df_regional.filter(variable=['Investment|Energy Supply','GDP|MER', EconFeas.warming_variable, 'Policy Cost|GDP Loss'], 
                    region=region, year=range(2020, end_year+1), 
                    scenario=scenario_model_list['scenario'], 
                    model=scenario_model_list['model'])
        
        df_temp = df_main.filter(variable=EconFeas.warming_variable, 
            region='World', year=range(2020, end_year+1), 
            scenario=scenario_model_list['scenario'], 
            model=scenario_model_list['model'])

    else:
        df = df.filter(variable=['Investment|Energy Supply','GDP|MER', EconFeas.warming_variable, 'Policy Cost|GDP Loss'], 
                            region='World', year=range(2020, end_year+1), 
                            scenario=scenario_model_list['scenario'], 
                            model=scenario_model_list['model'])
    
    # lists of the mean values and temp categories 
    mean_value_list = []
    mean_value_2050_list = []
    temperature_category_list = []
    largest_annual_increase_list = []
    mean_annual_increase_list = []
    max_values_list = []
    # make a dataframe to store the results with scenario and model 
    output_df = pd.DataFrame()
    # output_df['scenario'] = scenario_model_list['scenario']
    # output_df['model'] = scenario_model_list['model']

    # get list of years between 2020 and 2100 at decedal intervals
    year_list = list(range(2020, end_year+1, 10))

    # make a column for each year in year list
    for year in year_list:
        output_df[str(year)] = []
    
    # print(output_df
    timeseries_output = pd.DataFrame()

    # loop through the scenario model list
    for scenario, model in zip(scenario_model_list['scenario'], scenario_model_list['model']):

        
         # Filter out the data for the required scenario
        scenario_model_df = df.filter(model=model, scenario=scenario)

        variable_df = scenario_model_df.filter(variable='Investment|Energy Supply')
        # variable_series = pd.Series(variable_df['value'].values, index=variable_df['year'])
        variable_df = variable_df.interpolate(range(2020, 2101))
        # scenario_policy_cost = scenario_policy_cost.interpolate(range(2020, 2101))
        # make a list of the values
        investment_values = list(variable_df['value'].values)     

        # apply climate damages if the damages variable != None
        if apply_damages != None:
            
            if region is not None:

                temp_df = df_temp.filter(scenario=scenario, model=model)
                
                # get region position in the R10 list
                region_position = Data.R10.index(region)
                region_code = region_codes[region_position]

                scenario_model_df = apply_damages_regional(region,
                                                           region_code,
                                                           scenario_model_df,
                                                           temp_df,
                                                            scenario, model,
                                                           EconFeas.warming_variable,
                                                            meta_data,
                                                            year_list, 
                                                            R10_temps_df,
                                                            included_categories)
                                                           
            else:
                # scenario__model_df.to_csv('outputs/gdp_untreated.csv')
                scenario_model_df = apply_damage_function(scenario_model_df,
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
            year_df = scenario_model_df.filter(year=year)
            year_df_new = year_df.data
            investment = year_df['value'][year_df['variable'] == 'Investment|Energy Supply'].values
            gdp = year_df['value'][year_df['variable'] == 'GDP|MER'].values

            # Calculate the share of GDP that is invested in energy supply
            year_share = investment[0] / gdp[0]
            share_of_gdp[year] = year_share
            investment_values.append(investment[0])
        
        
        # Calculate the mean value of the share of GDP that is invested in energy supply
        list_of_values = list(share_of_gdp.values())
        mean_value_list.append(np.mean(list_of_values))

        # list of values as df to append to the timeseries output as a row
        list_of_values_df = pd.DataFrame(list_of_values).T
        list_of_values_df.columns = year_list
        timeseries_output = pd.concat([timeseries_output, list_of_values_df], axis=0)

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
        
        # get the maximum value of the share of GDP that is invested in energy supply
        max_values_list.append(max(list_of_values))

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
    output_df['scenario'] = scenario_model_list['scenario']
    output_df['model'] = scenario_model_list['model']
    output_df['temperature_category'] = temperature_category_list
    output_df['max_value'] = max_values_list
    output_df['mean_value'] = mean_value_list
    output_df['mean_value_2050'] = mean_value_2050_list
    output_df['largest_annual_increase'] = largest_annual_increase_list
    output_df['mean_annual_increase'] = mean_annual_increase_list

    # add the model, scenarios and temp categories to the timeseries output dataframe
    timeseries_output['scenario'] = scenario_model_list['scenario']
    timeseries_output['model'] = scenario_model_list['model']
    timeseries_output['temperature_category'] = temperature_category_list

    # move the scenario and model columns to the front
    # cols = output_df.columns.tolist()
    # cols = cols[-2:] + cols[:-2]
    # output_df = output_df[cols]
    
    if region is not None:

        # add the region to the output dataframe
        output_df['region'] = region
        # timeseries_output['region'] = region

        return output_df

    else:
        if apply_damages != None:
            output_df.to_csv('econ/energy_supply_investment_analysis_damages' + 
                            EconFeas.warming_variable + '.csv')
            timeseries_output.to_csv('econ/energy_supply_investment_timeseries_damages' +
                                        EconFeas.warming_variable + '.csv')
        
        else:
            output_df.to_csv('econ/energy_supply_investment_analysis.csv')
            timeseries_output.to_csv('econ/energy_supply_investment_timeseries.csv')


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
    - Pyam dataframe with the GDP, and policy cost data 
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
        
            year_df = gdp_untreated.filter(year=year)
            year_df_pd = year_df.as_pandas()
            warming = year_df_pd['value'][year_df_pd['variable'] == warming_variable].values
            gdp = year_df_pd['value'][year_df_pd['variable'] == 'GDP|MER'].values
            policy_cost = year_df_pd['value'][year_df_pd['variable'] == 'Policy Cost|GDP Loss'].values
            # add the policy cost to the GDP value
            baseline_gdp = gdp + policy_cost
            decade_warming = warming - present_warming
            damage = (alpha + (beta * present_warming)) * decade_warming + (beta/2) * decade_warming**2
            new_gdp = gdp * (1 + damage)

            # calculate the damage gap between the baseline GDP and the new GDP
            damage_gap = baseline_gdp - new_gdp

            # apply the damage gap to the scenario GDP
            new_gdp = gdp - damage_gap         
            
            # update the GDP value in the dataframe for the year
            gdp_treated.loc[(gdp_treated['year'] == year) & 
                            (gdp_treated['variable'] == 'GDP|MER'), 'value'] = new_gdp
            
        # convert back to a Pyam dataframe and drop exclude column
        gdp_treated = pyam.IamDataFrame(data=gdp_treated)
        # gdp_treated.to_csv('outputs/gdp_treated.csv')
        return gdp_treated


def apply_damages_regional(region, 
                           region_code,
                           scenario_model_df,
                            temp_df, 
                          scenario, model, 
                          warming_variable, 
                          meta_data,
                          year_list, 
                          regional_damages_df,
                          included_categories):

    # check in metadata whether climate impacts have been accounted for
    damages_accounted_for = meta_data[meta_data['scenario'] == scenario]
    damages_accounted_for = damages_accounted_for[damages_accounted_for['model'] == model]
    if damages_accounted_for['Climate impacts'].values[0] == 'yes':
        return scenario_model_df

    elif damages_accounted_for['Category'].values[0] not in included_categories:
            return scenario_model_df
    # otherwise, calculate the damages based on the warming variable for the scenario
    else:
        # print(damages_accounted_for['Category'].values[0])
        # extract the relevant warming data for the region
        region_temp_df = regional_damages_df[regional_damages_df['region'] == region_code]
        # loop through the years to calculate the temperature difference
        
        gdp_treated = scenario_model_df.as_pandas()
        for year in year_list:
            
            # calculate the current warming level
            year_temp_df = temp_df.filter(year=year)
            year_temp_df = year_temp_df.as_pandas()
            year_warming = year_temp_df['value'][year_temp_df['variable'] == warming_variable].values[0]
            year_warming = round(year_warming, 1)

            if year_warming < 1.25:
                
                continue
                # output GDP as is
            else:
                # extract the warming data for the year
                region_temp_year = region_temp_df[region_temp_df['temperature'] == year_warming]
                
                # get the damages for the region
                # print(year_warming)
                damages = region_temp_year[region_temp_year['temperature'] == year_warming]['damages'].values[0]

                # filter out the data for the year
                year_df = scenario_model_df.filter(year=year)
                year_df_pd = year_df.as_pandas()
                gdp = year_df_pd['value'][year_df_pd['variable'] == 'GDP|MER'].values
                policy_cost = year_df_pd['value'][year_df_pd['variable'] == 'Policy Cost|GDP Loss'].values
                # add the policy cost to the GDP value
                baseline_gdp = gdp + policy_cost
                
                # calculate the damage gap between the baseline GDP and the new damaged GDP
                damage_gap = baseline_gdp - (baseline_gdp / (1 + damages))
                
                # apply the damage gap to the scenario GDP
                new_gdp = gdp - damage_gap

                # update the GDP value in the dataframe for the year
                gdp_treated.loc[(gdp_treated['year'] == year) &
                                        (gdp_treated['variable'] == 'GDP|MER'), 'value'] = new_gdp
                
        gdp_treated = pyam.IamDataFrame(data=gdp_treated)
        return gdp_treated

            
def calculate_R10_damages_temps(regions, damage_model, 
                                min_temp, max_temp, 
                                country_sheet):
    
    """
    Function that runs through the country-level data and estimates
    the damages at each temperature level for each R10 region using
    the GDPcap for weighting and the overall GDP loss for the region.
    
    Inputs:
    - regions: list of R10 regions
    - damage_model: the damage model to use (links to files)
    - min_temp: minimum temperature to calculate damages for
    - max_temp: maximum temperature to calculate damages for
    - country_sheet: the sheet connecting R10 regions to countries

    Outputs:
    - .csv file with the results of the analysis with R10 regions, 
    temperatures and damages

    points:
    - if file doesn't exist, interpolate over the temperature range
    - interpolate anyway to .01 degree intervals?
    - 

    """
    
    if damage_model == 'dice':
        file_format = "cost_est_dicetemp_"
    else:
        file_format = "cost_esttemp_"
    
    # make list of temperatures to calculate damages for at .1 degree intervals, one decimal place
    temp_list = np.arange(min_temp, max_temp, 0.1, dtype=float)
    temp_list = np.round(temp_list, 1)
    temp_list = [float(temp) for temp in temp_list]
    temp_list = [int(temp) if temp.is_integer() else temp for temp in temp_list]


    # make a dataframe to store the results
    regions_list = []
    temps_list = []
    damages_list = []

    # loop through the regions
    for region in regions:

        print("Calculating damages for region: ", region)
        # get the countries in the region
        countries = country_sheet[country_sheet['r10_iamc'] == region]
        countries_list = countries['iso3c'].tolist()

        # loop through temperatures
        for temp in temp_list:

            try:
                # retrieve file for the temperature
                model_damages = pd.read_csv('econ/price_comparisons/' + file_format + str(temp) + ".csv")

                # list to store the damages for each country and weighting
                weighting = []
                damages = []
                
                # loop through the countries
                for country in countries_list:

                    try:
                        # get the GDPcap and damages for the country
                        weighting.append(model_damages[model_damages['Group.1'] == country]['gdpcap'].values[0])
                        damages.append(model_damages[model_damages['Group.1'] == country]['gdploss'].values[0])
                    
                    # if country not in the sheet due to missing data, skip
                    except IndexError:
                        continue

                # calculate the weighted average of the damages
                weighted_damages = np.average(damages, weights=weighting)

                # append the results to the lists
                regions_list.append(region)
                temps_list.append(temp)
                damages_list.append(weighted_damages)

            except FileNotFoundError:
                print("File not found for temperature: ", temp)
                continue

    # make a dataframe with the results
    output_df = pd.DataFrame()
    output_df['region'] = regions_list
    output_df['temperature'] = temps_list
    output_df['damages'] = damages_list

    # check_temp_list = output_df['temperature'].unique().tolist()    
    # print(type(check_temp_list[0]), type(temp_list[0]))
    
    # for temp in temp_list:
    #     if temp not in output_df['temperature']:
    #         print("Interpolating damages for temperature: ", temp)
    #         interpolated_damages = np.interp(temp, output_df['temperature'], output_df['damages'])
    #         interpolated_row = pd.DataFrame({'region': [region], 'temperature': [temp], 'damages': [interpolated_damages]})
    #         output_df = pd.concat([output_df, interpolated_row], ignore_index=True)

    # save the results to a .csv file
    output_df.to_csv('econ/R10_damages_temps.csv')
    

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


def make_temporal_plots(df_results, df_results_damages, categories, metric):

    """
    Function to make temporal line plots for the results of the analysis.
    
    To do: - try plotting all on one plot with low alpha (and IQR) for the different categories
        - add in the damages data for second subplot
        - try plotting individual ones for each category with both the damaged and undamaged data on each
        - depending on outcome, add on boxplots for each category
    
    Inputs:
    - df_results: dataframe with the results of the analysis
    - df_results_damages: dataframe with the results of the analysis with damages applied
    - categories: list of categories to plot
    - metric: the metric to plot!

    Outputs:
    - line plots for each of the categories

    """
    # plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.direction'] = 'in'
    # plt.rcParams['ytick.major.left'] = True
    # # plt.rcParams['ytick.major.right'] = True
    # plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.linewidth'] = 0.65

    # set up figure with width 180mm and height of 100mm with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(7.08, 6), facecolor='white')


    # first subplot for undamaged data
    for category in categories:

        # filter out the data for the category
        category_data = df_results[df_results['temperature_category'] == category]
        category_data_damages = df_results_damages[df_results_damages['temperature_category'] == category]


        median_values = []
        percentile_25 = []
        percentile_75 = []
        median_values_damages = []
        percentile_25_damages = []
        percentile_75_damages = []

        for year in range(2020, 2110, 10):

            # filter out the data for the year
            year_data = category_data[str(year)]
            year_data_damages = category_data_damages[str(year)]

            median_values.append(np.median(year_data))
            median_values_damages.append(np.median(year_data_damages))
            percentile_25.append(np.percentile(year_data, 25))
            percentile_75.append(np.percentile(year_data, 75))
            percentile_25_damages.append(np.percentile(year_data_damages, 25))
            percentile_75_damages.append(np.percentile(year_data_damages, 75))

        # plot the data
        axs[0].plot(range(2020, 2110, 10), median_values, label=category, color=EconFeas.cb_friendly_cat_colours[category])
        axs[0].fill_between(range(2020, 2110, 10), percentile_25, percentile_75, alpha=0.1, color=EconFeas.cb_friendly_cat_colours[category])
        axs[1].plot(range(2020, 2110, 10), median_values_damages, linestyle='--', color=EconFeas.cb_friendly_cat_colours[category])
        axs[1].fill_between(range(2020, 2110, 10), percentile_25_damages, percentile_75_damages, alpha=0.1, color=EconFeas.cb_friendly_cat_colours[category])



    plt.show()


def temporal_subplots(df_results, df_results_damages, categories, metric):

    """
    Function to make temporal line plots for the results of the analysis.
    
    To do: - try plotting all on one plot with low alpha (and IQR) for the different categories
        - add in the damages data for second subplot
        - try plotting individual ones for each category with both the damaged and undamaged data on each
        - depending on outcome, add on boxplots for each category
    
    Inputs:
    - df_results: dataframe with the results of the analysis
    - df_results_damages: dataframe with the results of the analysis with damages applied
    - categories: list of categories to plot
    - metric: the metric to plot!

    Outputs:
    - line plots for each of the categories

    """
    # plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.direction'] = 'in'
    # plt.rcParams['ytick.major.left'] = True
    # # plt.rcParams['ytick.major.right'] = True
    # plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.linewidth'] = 0.65
    plt.rcParams['font.size'] = 7
    plt.rcParams['axes.titlesize'] = 7
    plt.rcParams['figure.dpi'] = 150

    # set up figure with width 180mm and height of 100mm with two subplots
    fig, axs = plt.subplots(2, 4, figsize=(7.08, 6), facecolor='white', sharey=True)

    axs = axs.flatten()

    # first subplot for undamaged data
    for i, category in enumerate(categories):

        # filter out the data for the category
        category_data = df_results[df_results['temperature_category'] == category]
        category_data_damages = df_results_damages[df_results_damages['temperature_category'] == category]

        median_values = []
        percentile_25 = []
        percentile_75 = []
        median_values_damages = []
        percentile_25_damages = []
        percentile_75_damages = []

        for year in range(2020, 2110, 10):

            # filter out the data for the year
            year_data = category_data[str(year)]
            year_data_damages = category_data_damages[str(year)]

            median_values.append(np.median(year_data))
            median_values_damages.append(np.median(year_data_damages))
            percentile_25.append(np.percentile(year_data, 25))
            percentile_75.append(np.percentile(year_data, 75))
            percentile_25_damages.append(np.percentile(year_data_damages, 25))
            percentile_75_damages.append(np.percentile(year_data_damages, 75))

        
        # multiple the values by 100 to get percentage  
        median_values = [i*100 for i in median_values]
        median_values_damages = [i*100 for i in median_values_damages]
        percentile_25 = [i*100 for i in percentile_25]
        percentile_75 = [i*100 for i in percentile_75]
        percentile_25_damages = [i*100 for i in percentile_25_damages]
        percentile_75_damages = [i*100 for i in percentile_75_damages]

        # plot the data
        axs[i].plot(range(2020, 2110, 10), median_values, label=category, color=EconFeas.cb_friendly_cat_colours[category], linewidth=0.75)
        axs[i].fill_between(range(2020, 2110, 10), percentile_25, percentile_75, alpha=0.2, color=EconFeas.cb_friendly_cat_colours[category])
        axs[i].plot(range(2020, 2110, 10), median_values_damages, linestyle='--', color=EconFeas.cb_friendly_cat_colours[category], linewidth=0.75)
        axs[i].fill_between(range(2020, 2110, 10), percentile_25_damages, percentile_75_damages, alpha=0.1, color=EconFeas.cb_friendly_cat_colours[category], hatch='//')
    
        # set y axis limits
        axs[i].set_ylim(0.6, 3.5)

        # set x axis limits
        axs[i].set_xlim(2020, 2100)

        # set the subplot title
        axs[i].set_title(category)

        # set the y axis label
        if i == 0 or i == 4:
            axs[i].set_ylabel('Investment in energy supply as a share of GDP (%)')

    # plt.tight_layout()
    plt.show()


def box_plots_by_cat(df_results, df_results_damages, categories, metric):
    
    """
    function that plot box plots of the results for each of the categories


    Inputs:
    - df_results: dataframe with the results of the analysis
    - df_results_damages: dataframe with the results of the analysis with damages applied
    - categories: list of categories to plot
    - metric: the metric to plot

    """
    plt.rcParams['font.size'] = 7
    plt.rcParams['axes.titlesize'] = 7
    plt.rcParams['figure.dpi'] = 150
    # set up figure with width 180mm and height of 100mm 
    fig, axs = plt.subplots(1, 1, figsize=(7.08, 6), facecolor='white')

    tick_positions = []
    x_position = 0
    for category in categories:

        # filter out the data for the category
        category_data = df_results[df_results['temperature_category'] == category]
        category_data_damages = df_results_damages[df_results_damages['temperature_category'] == category]

        # plot matplotlib boxplot
        axs.boxplot(category_data[metric]*100, positions=[x_position], widths=0.3, 
                    meanline=True, showmeans=True, showfliers=False, patch_artist=True, 
                    boxprops=dict(facecolor=EconFeas.cb_friendly_cat_colours[category], edgecolor='white', linewidth=0.5), 
                    meanprops=dict(color='black', linewidth=.75), medianprops=dict(color='black'),
                    whiskerprops=dict(color='black', linewidth=0.5), capprops=dict(color='black', linewidth=0.5))
        
        axs.boxplot(category_data_damages[metric]*100, positions=[x_position+0.35], widths=0.3, 
                    meanline=True, showmeans=True, showfliers=False, patch_artist=True, 
                    boxprops=dict(facecolor=EconFeas.cb_friendly_cat_colours[category], edgecolor='white', linewidth=0.5, alpha=0.5), meanprops=dict(color='black', linewidth=.75), medianprops=dict(color='black'),
                    whiskerprops=dict(color='black', linewidth=0.5), capprops=dict(color='black', linewidth=0.5))
        
        tick_positions.append(x_position + 0.175)
        x_position += 1

    axs.set_xticks(tick_positions)
    axs.set_xticklabels(categories)

    # set the y axis label
    axs.set_ylabel('Investment in energy supply as a share of GDP (%) (2020-2100)')

    # plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()