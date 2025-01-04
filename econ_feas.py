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

# import mpatches
import matplotlib.patches as mpatches

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
    warming_variable_5th = 'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|5.0th Percentile'
    warming_variable_95th = 'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|95.0th Percentile'
    warming_variable_50th = 'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile'
    present_warming = 1.25
    # AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|95.0th Percentile
    iea_country_groupings = pd.read_csv('econ/iea_country_groupings.csv')
    by_country_gdp = pd.read_csv('econ/IMF_GDP_data_all_countries.csv')

    temp_groupings = [('C1', 'C2'), ('C3', 'C4'), ('C5', 'C6'), ('C7', 'C8')]

    cb_friendly_cat_colours = {'C1':'#332288', 
                               'C2':'#117733', 
                               'C3':'#44AA99', 
                               'C4':'#88CCEE', 
                               'C5':'#DDCC77', 
                               'C6':'#CC6677', 
                               'C7':'#AA4499', 
                               'C8':'#882255'}

    regional_colours = {'R10LATIN_AM':'#882255',
                        'R10INDIA+':'#CC6677',
                        'R10AFRICA':'#AA4499',
                        'R10CHINA+':'#DDCC77',
                        'R10MIDDLE_EAST':'#117733',
                        'R10EUROPE':'#332288',
                        'R10REST_ASIA':'#AA4499',
                        'R10PAC_OECD':'#88CCEE',
                        'R10REF_ECON':'#88CCEE',
                        'R10NORTH_AM':'#88CCEE'}

class EconData:

    # Global Data
    global_scenarios_models = pd.read_csv('econ/scenarios_models_world.csv')
    # global_econ_pyamdf = pyam.IamDataFrame(data="econ/scenarios_models_worldcat_df['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'].csv")
    global_econ_pyamdf = pyam.IamDataFrame(data="econ/econ_data_world.csv")
    global_econ_meta = pd.read_csv("econ/meta_data_c1_c8.csv")
    global_econ_results_no_damages = pd.read_csv('econ/energy_supply_investment_analysis.csv')
    global_econ_results_damages = pd.read_csv('econ/energy_supply_investment_analysis_damages' + EconFeas.warming_variable + '.csv')        
    global_econ_results_damages_95 = pd.read_csv('econ/energy_supply_investment_analysis_damagesAR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|95.0th Percentile.csv')
    global_econ_results_damages_5 = pd.read_csv('econ/energy_supply_investment_analysis_damagesAR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|5.0th Percentile.csv')

    imps = {'scenarios':['SusDev_SDP-PkBudg1000',
                          'PEP_1p5C_red_eff', 'DeepElec_SSP2_ HighRE_Budg900'], 
            'models':['REMIND-MAgPIE 2.1-4.2',
                       'REMIND-MAgPIE 1.7-3.0', 'REMIND-MAgPIE 2.1-4.3'], 
            'names':['Low energy demand','High CDR', 
                     'High Renewables']}
    imp_colours =  ['#56B4E9','#e69f00','#009E73']
    # imps_all = {'scenarios':['SusDev_SDP-PkBudg1000', 'CD-LINKS_NPi2020_1000',
    #                       'CEMICS_SSP2-1p5C-minCDR','PEP_1p5C_red_eff', 'CO_2Deg2030','DeepElec_SSP2_ HighRE_Budg900'], 
    #         'models':['REMIND-MAgPIE 2.1-4.2', 'REMIND-MAgPIE 1.7-3.0',
    #                    'REMIND-MAgPIE 2.1-4.2', 'REMIND-MAgPIE 1.7-3.0','REMIND-MAgPIE 1.7-3.0', 'REMIND-MAgPIE 2.1-4.3'], 
    #         'names':['Low energy demand', 'High energy demand', 'Low CDR', 'High CDR', 
    #                  'Low Renewables','High Renewables']}
    # imp_colours_all =  ['#56B4E9','#3A94C3', '#F7CC84', '#E69F00', '#59ab95', '#009E73']

    imps_fairness = {'scenarios':['EN_INDCi2030_500f','EN_NPi2020_450'],	
                     'models':['WITCH 5.0','MESSAGEix-GLOBIOM_1.1'],
                     'names':['Low Regional Fairness','High Regional Fairness']}
    imps_fairness_colours = ['#CC79A7', '#A35485']

    imps_overshoot = {'scenarios':['EN_NPi2020_600','CO_2Deg2030'],    
                     'models':['GEM-E3_V2021','REMIND-MAgPIE 1.7-3.0'],
                     'names':['Low Overshoot (C2)','High Overshoot (C2)']}
    
    imps_overshoot_colours = ['#009E73', '#D55E00']

    # Regional Data
    regional_scenarios_models = pd.read_csv('econ/scenarios_models_R10.csv')
    regional_econ_pyamdf = pyam.IamDataFrame(data='econ/pyamdf_econ_analysis_R10.csv')
    regional_damage_estimates = pd.read_csv('econ/R10_damages_temps.csv')
    regional_categories = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    regional_results_no_damages = pd.read_csv('econ/energy_supply_investment_analysis_R10.csv')
    regional_results_damages = pd.read_csv('econ/energy_supply_investment_analysis_R10_damages.csv')
    gdp_data = pd.read_csv('econ/gdp_R10.csv')  
    gdp_data_damages = pd.read_csv('econ/gdp_damages_R10_damages.csv')
    north_south_split = pd.read_csv('econ/north_south_regional_investment.csv')
    historical_values = pd.read_csv('econ/energy_investment_regions.csv')
    north_south_split_5th = pd.read_csv('econ/north_south_regional_investment' + EconFeas.warming_variable_5th + '.csv')
    north_south_split_95th = pd.read_csv('econ/north_south_regional_investment' + EconFeas.warming_variable_95th + '.csv')
    north_south_split_50th = pd.read_csv('econ/north_south_regional_investment' + EconFeas.warming_variable_50th + '.csv')

    # plotting data
    plot_regions = ['Countries of Latin America and the Caribbean', 'Countries of South Asia; primarily India', 'Eastern and Western Europe (i.e., the EU28)']



def main() -> None:


    # energy_supply_investment_score(Data.dimensions_pyamdf, 0.023, 2100, Data.model_scenarios, Data.categories)
    # energy_supply_investment_analysis(0.023, 2100, EconData.global_scenarios_models, 
    #                                   EconData.global_econ_pyamdf, 
    #                                   EconData.global_econ_meta,
    #                                   apply_damages=True)
    # map_countries_to_regions(EconFeas.iea_country_groupings, EconFeas.by_country_gdp)
    # scenarios_list = pd.read_csv('econ_regional_Countries of Sub-Saharan Africa.csv')
    # data = pyam.IamDataFrame(data="pyamdf_econ_data_R10['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'].csv")
    # box_plots_regional_categories(EconData.regional_results_no_damages,
    #                               EconData.regional_results_damages,
    #                                EconData.regional_categories, 
    #                                'mean_value', Data.R10m)
    # output_df.to_csv('outputs/energy_supply_investment_score_regional' + str(Utils.categories) + '.csv')
    # # energy_supply_investment_score(Data.regional_dimensions_pyamdf, 0.023, 2100, Data.model_scenarios, Data.categories, regional=None)
    # temporal_subplots(EconData.global_econ_results_no_damages, EconData.global_econ_results_damages, EconFeas.plotting_categories, 'mean_value')
    
    # calculate_R10_damages_temps(Data.R10_codes, 'default', 1.3, 4.6, Data.region_country_df)
    # run_econ_analysis_regional()

    # def north_south_regional_investment(scenario_model_list, 
    #                                 df_results, df_results_damages,
    #                                 gdp_data, gdp_data_damages,
    #                                 regions, metric, start_year, end_year, 
    #                                 ignore_positive_damages=False):
    # north_south_regional_investment(EconData.regional_scenarios_models,
    #                                 EconData.regional_results_no_damages,
    #                                 EconData.regional_results_damages,
    #                                 EconData.gdp_data,
    #                                 EconData.gdp_data_damages,
    #                                 Data.R10, 'mean_value', 2020, 2100, 
    #                                 ignore_positive_damages=True)
    # temporal_subplots_north_south(EconData.north_south_split, EconFeas.plotting_categories,
    #                               2020, 2100)
    # temporal_subplots_regional(EconData.regional_results_no_damages, EconData.regional_results_damages,
    #                            EconFeas.plotting_categories, 'mean_value', EconData.plot_regions, 
    #                            EconData.historical_values)
    # temporal_subplots_select_scenarios(EconData.global_econ_results_no_damages,
    #                                     EconData.global_econ_results_damages, 
    #                                     EconData.imps_overshoot, EconData.imps_overshoot_colours)
    # temporal_subplots_select_scenarios_north_south(EconData.north_south_split_5th, 
    #                                    EconData.north_south_split_50th,
    #                                     EconData.north_south_split_95th,
    #                                 EconData.imps_fairness)
    temporal_subplots(EconData.global_econ_results_no_damages,
                      EconData.global_econ_results_damages,
                      EconFeas.temp_groupings)  
    # temporal_subplots_north_south(EconData.north_south_split_50th,
    #                               EconFeas.temp_groupings, 2020, 2100)                 
    # temporal_subplots_north_south_example_regions(EconData.north_south_split_50th,
    #                                               EconData.regional_results_no_damages,
    #                                               EconData.regional_results_damages,
    #                                               EconFeas.temp_groupings,
    #                                               2020, 2100, 
    #                                               north_south='None',
    #                                               example_regions=['R10INDIA+', 'R10MIDDLE_EAST'])



def run_econ_analysis_regional():

    output_df = pd.DataFrame()
    gdp_output = pd.DataFrame()
    export_gdp = pd.DataFrame()
    damages = 'damages'
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

        output_df = pd.concat([output_df, to_append[0]], ignore_index=True, axis=0)
        export_gdp = pd.concat([export_gdp, to_append[1]], ignore_index=True,  axis=0)
        
    if damages != None:
        output_df.to_csv('econ/energy_supply_investment_analysis_R10_' + damages + '.csv')
        export_gdp.to_csv('econ/gdp_damages_R10_' + damages + '.csv') 
    else:
        output_df.to_csv('econ/energy_supply_investment_analysis_R10.csv')
        export_gdp.to_csv('econ/gdp_R10.csv')


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
                                      df_regional=None, R10_temps_df=None, 
                                      included_categories=None):

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
        df = df_main.filter(variable=['Investment|Energy Supply','GDP|MER', EconFeas.warming_variable, 'Policy Cost|GDP Loss'], 
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

    gdp_output = pd.DataFrame()

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
        variable_df = variable_df.interpolate(range(2020, 2101))

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
        gdp_dict = {}
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
            gdp_dict[year] = gdp[0]
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

        # make GDP dict into a df 
        scenario_gdp = pd.DataFrame(gdp_dict.items(), columns=['year', 'value'])
        scenario_gdp = scenario_gdp.T
        scenario_gdp.columns = scenario_gdp.iloc[0].astype(int).astype(str)
        scenario_gdp = scenario_gdp[1:]

        # append the values to the output dataframe
        gdp_output = pd.concat([gdp_output, scenario_gdp], axis=0)


    output_df = output_df.reset_index(drop=True)
    output_df['scenario'] = scenario_model_list['scenario']
    output_df['model'] = scenario_model_list['model']
    output_df['temperature_category'] = temperature_category_list
    output_df['max_value'] = max_values_list
    output_df['mean_value'] = mean_value_list
    output_df['mean_value_2050'] = mean_value_2050_list
    output_df['largest_annual_increase'] = largest_annual_increase_list
    output_df['mean_annual_increase'] = mean_annual_increase_list

    gdp_output = gdp_output.reset_index(drop=True)
    gdp_output['scenario'] = scenario_model_list['scenario']
    gdp_output['model'] = scenario_model_list['model']
    gdp_output['temperature_category'] = temperature_category_list

    if region is not None:

        # add the region to the output dataframe
        output_df['region'] = region
        gdp_output['region'] = region
        return [output_df, gdp_output]

    else:
        if apply_damages != None:
            output_df.to_csv('econ/energy_supply_investment_analysis_damages' + 
                            EconFeas.warming_variable + '.csv')

            gdp_output.to_csv('econ/gdp_damages' + 
                            EconFeas.warming_variable + '.csv')
        else:
            output_df.to_csv('econ/energy_supply_investment_analysis.csv')
            gdp_output.to_csv('econ/gdp.csv')


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

    for region in regions:
        # interpolate the damages for any missing temperatures
        
        # filter for the region
        region_check = output_df[output_df['region'] == region]
        for temp in temp_list:
            # filter for the temperature data
            temp_check = region_check[region_check['temperature'] == temp]
            if len(temp_check) == 0:
                print("Interpolating damages for temperature: ", temp)
                interpolated_damages = np.interp(temp, region_check['temperature'], region_check['damages'])
                interpolated_row = pd.DataFrame({'region': [region], 'temperature': [temp], 'damages': [interpolated_damages]})
                output_df = pd.concat([output_df, interpolated_row], ignore_index=True)

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


def north_south_regional_investment(scenario_model_list, 
                                    df_results, df_results_damages,
                                    gdp_data, gdp_data_damages,
                                    regions, metric, start_year, end_year, 
                                    ignore_positive_damages=False):
    
    """
    Function that runs through each scenario in the list and uses the gdp to 
    calculated the global north, global south investment shares by year, the 
    function uses the decadal GDP to calculate weighted value by regions. 
    For values where the GDP damage is 'positive' this is set to the same 
    as the GDP value.

    Inputs:
    - scenario_model_list: list of scenarios and models to use
    - df_results: dataframe with the results of the analysis
    - df_results_damages: dataframe with the results of the analysis with damages
    - gdp_data: dataframe with the GDP data
    - gdp_data_damages: dataframe with the GDP data with damages
    - regions: list of regions to use
    - metric: the metric to use
    - start_year: the start year of the analysis
    - end_year: the end year of the analysis
    - ignore_positive_damages: whether to ignore positive damages and set to zero

    Outputs:
    - .csv file with the results of the analysis providing the investment share
    for the global north and south regions for each year and scenario.

    """

    # get the list of years between the start and end year
    year_list = list(range(start_year, end_year+1, 10))

    # make a dataframe to store the results
    output_df = pd.DataFrame()

    # loop through the scenarios
    for scenario, model in zip(scenario_model_list['scenario'], scenario_model_list['model']):

        # filter out the data for the scenario and model
        scenario_model_df = df_results[(df_results['scenario'] == scenario) & (df_results['model'] == model)]
        scenario_model_df_damages = df_results_damages[(df_results_damages['scenario'] == scenario) & (df_results_damages['model'] == model)]

        # filter out the GDP data for the scenario and model
        gdp_scenario_model = gdp_data[(gdp_data['scenario'] == scenario) & (gdp_data['model'] == model)]
        gdp_scenario_model_damages = gdp_data_damages[(gdp_data_damages['scenario'] == scenario) & (gdp_data_damages['model'] == model)]

        # loop through the years
        for year in year_list:

            # dictionaries to store the gdp and investment values in for the year
            north_values = {}
            south_values = {}
            north_values_damages = {}
            south_values_damages = {}

            # loop through the regions
            for region in regions:

                # get region code and north south split
                region_code = Data.R10_codes[Data.R10.index(region)]
                north_south = Data.R10_development[region_code]

                # filter out the data for the region
                region_df =  scenario_model_df[scenario_model_df['region'] == region]
                region_df_damages = scenario_model_df_damages[scenario_model_df_damages['region'] == region]

                # filter out the GDP data for the region
                gdp_region = gdp_scenario_model[gdp_scenario_model['region'] == region]
                gdp_region_damages = gdp_scenario_model_damages[gdp_scenario_model_damages['region'] == region]

                # extract the values
                region_year_investment_value = region_df[str(year)].values[0]
                region_year_investment_value_damages = region_df_damages[str(year)].values[0]
                region_year_gdp = gdp_region[str(year)].values[0]
                region_year_gdp_damages = gdp_region_damages[str(year)].values[0]

                category = region_df['temperature_category'].values[0]

                # if the ignore positive damages flag is set, check if the damages are positive
                # and if so, set the damages value to the same as the normal investment value
                # This is in place the damage functions do not properly calculate the damages
                # for some colder regions
                if ignore_positive_damages:
                    if region_year_investment_value_damages < region_year_investment_value:
                        region_year_investment_value_damages = region_year_investment_value
                        region_year_gdp_damages = region_year_gdp
                
                # append the values to the dictionaries
                if north_south == 'North':
                    north_values[region_year_gdp] = region_year_investment_value
                    north_values_damages[region_year_gdp_damages] = region_year_investment_value_damages
                else:
                    south_values[region_year_gdp] = region_year_investment_value
                    south_values_damages[region_year_gdp_damages] = region_year_investment_value_damages

            # calculate the weighted gdp average of the investment values 
            # for the north and south regions for the year
            north_weighted_value = np.average(list(north_values.values()), weights=list(north_values.keys()))
            south_weighted_value = np.average(list(south_values.values()), weights=list(south_values.keys()))
            north_weighted_value_damages = np.average(list(north_values_damages.values()), weights=list(north_values_damages.keys()))
            south_weighted_value_damages = np.average(list(south_values_damages.values()), weights=list(south_values_damages.keys()))
                                                      
            # concat the results to the output dataframe
            output_df = pd.concat([output_df, pd.DataFrame({'scenario': [scenario], 'model': [model], 'category':[category], 'year': [year],
                                                            'north_value': [north_weighted_value], 'south_value': [south_weighted_value],
                                                            'north_value_damages': [north_weighted_value_damages], 'south_value_damages': [south_weighted_value_damages]})], axis=0)

    # save the results to a .csv file
    output_df.to_csv('econ/north_south_regional_investment' + EconFeas.warming_variable + '.csv')


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
        
        # filter out the data for the category
        # category_data = df_results[df_results['temperature_category'] == category[0], category[1]]
        # category_data_damages = df_results_damages[df_results_damages['temperature_category'] == category[0], category[1]]


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
        axs[0].fill_between(range(2020, 2110, 10), percentile_25, percentile_75, alpha=0.2, color=EconFeas.cb_friendly_cat_colours[category])
        axs[1].plot(range(2020, 2110, 10), median_values_damages, linestyle='--', color=EconFeas.cb_friendly_cat_colours[category])
        axs[1].fill_between(range(2020, 2110, 10), percentile_25_damages, percentile_75_damages, alpha=0.2, color=EconFeas.cb_friendly_cat_colours[category])



    plt.show()


def temporal_subplots(df_results, df_results_damages, categories):

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
    fig, axs = plt.subplots(1, 4, figsize=(7.08, 3), facecolor='white', sharey=True)

    axs = axs.flatten()

    # first subplot for undamaged data
    for i, category in enumerate(categories):

        category_data = df_results[df_results['temperature_category'].isin(category)]
        category_data_damages = df_results_damages[df_results_damages['temperature_category'].isin(category)]

        # filter out the data for the category
        # category_data = df_results[df_results['temperature_category'] == category]
        # category_data_damages = df_results_damages[df_results_damages['temperature_category'] == category]

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
        axs[i].plot(range(2020, 2110, 10), median_values/median_values[0], label=category, color=EconFeas.cb_friendly_cat_colours[category[0]], linewidth=0.75)
        axs[i].fill_between(range(2020, 2110, 10), percentile_25/median_values[0], percentile_75/median_values[0], alpha=0.3, color=EconFeas.cb_friendly_cat_colours[category[0]])
        axs[i].plot(range(2020, 2110, 10), median_values_damages/median_values[0], linestyle='--', color=EconFeas.cb_friendly_cat_colours[category[0]], linewidth=0.75)
        axs[i].fill_between(range(2020, 2110, 10), percentile_25_damages/median_values[0], percentile_75_damages/median_values[0], alpha=0.15, color=EconFeas.cb_friendly_cat_colours[category[0]], hatch='//')
    
        # # set y axis limits
        # axs[i].set_ylim(0.6, 3.5)

        # set x axis limits
        axs[i].set_xlim(2020, 2100)

        # set the subplot title
        axs[i].set_title(category)

        # set the y axis label
        if i == 0 or i == 4:
            axs[i].set_ylabel('Investment in energy supply as a share of GDP (%)')

    # plt.tight_layout()
    plt.show()


def box_plots_regional_categories(df_results, df_results_damages, categories, metric, regions):

    """
    Figure that makes boxplots on subplots (categories) for all regions inputted. 
    On the box plots, the undamaged and damaged data for each region is plotted 
    side by side.

    Inputs: 
    - df_results: dataframe with the results of the analysis
    - df_results_damages: dataframe with the results of the analysis with damages applied
    - categories: list of categories to plot
    - metric: the metric to plot
    - regions: list of regions to plot

    Outputs:
    - box plots for each of the categories showing each region's data
    
    """
    plt.rcParams['font.size'] = 7
    plt.rcParams['axes.titlesize'] = 7
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['axes.linewidth'] = 0.65

    # set up figure with width 180mm and height of 100mm with two subplots
    fig, axs = plt.subplots(2, 4, figsize=(7.08, 8), facecolor='white', sharey=True)

    axs = axs.flatten()

    for i, category in enumerate(categories):

        # filter out the data for the category
        category_data = df_results[df_results['temperature_category'] == category]
        category_data_damages = df_results_damages[df_results_damages['temperature_category'] == category]

        tick_positions = []
        x_position = 0
        
        for region in regions:

            # filter out the data for the region
            region_data = category_data[category_data['region'] == region]
            region_data_damages = category_data_damages[category_data_damages['region'] == region]

            # plot matplotlib boxplot
            axs[i].boxplot(region_data[metric]*100, positions=[x_position], widths=0.3, 
                        meanline=True, showmeans=True, showfliers=False, patch_artist=True, 
                        boxprops=dict(facecolor=EconFeas.cb_friendly_cat_colours[category], edgecolor='white', linewidth=0.5), 
                        meanprops=dict(color='black', linewidth=.75), medianprops=dict(color='black'),
                        whiskerprops=dict(color='black', linewidth=0.5), capprops=dict(color='black', linewidth=0.5))
            
            axs[i].boxplot(region_data_damages[metric]*100, positions=[x_position+.35], widths=0.3, 
                        meanline=True, showmeans=True, showfliers=False, patch_artist=True, 
                        boxprops=dict(facecolor=EconFeas.cb_friendly_cat_colours[category], edgecolor='white', linewidth=0.5, alpha=0.5), 
                        meanprops=dict(color='black', linewidth=.75), medianprops=dict(color='black'),
                        whiskerprops=dict(color='black', linewidth=0.5), capprops=dict(color='black', linewidth=0.5))

            tick_positions.append(x_position + .175)
            x_position += 1

        axs[i].set_xticks(tick_positions)
        axs[i].set_xticklabels(Data.R10_codes)

        # make the x axis labels vertical
        for tick in axs[i].get_xticklabels():
            tick.set_rotation(90)

        axs[i].set_title(category)
        # axs[i].set_xticks([0, 0.35])
        # axs[i].set_xticklabels(['Undamaged', 'Damaged'])
        # axs[i].set_ylabel('Investment in energy supply as a share of GDP (%)')
        # set the y axis label
        if i == 0 or i == 4:
            axs[i].set_ylabel('Investment in energy supply as a share of GDP (%) (2020-2100)')

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


def temporal_subplots_regional(df_results, df_results_damages, categories, metric,
                               regions, historical_values):

    """
    Function that builds subplots (one for each category) with temporal regional data
    plotted on each subplot for selected regions. The timeseries data for each region
    for both the undamaged and damaged data is plotted on each subplot.

    Inputs:
    - df_results: dataframe with the results of the analysis
    - df_results_damages: dataframe with the results of the analysis with damages applied
    - categories: list of categories to plot
    - metric: the metric to plot
    - regions: list of regions to plot

    Outputs:
    - subplots with regional data for each category plotted on each

    """

    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.linewidth'] = 0.65
    plt.rcParams['font.size'] = 7
    plt.rcParams['axes.titlesize'] = 7
    plt.rcParams['figure.dpi'] = 150

    # set up figure with width 180mm and height of 100mm with four subplots
    fig, axs = plt.subplots(2, 4, figsize=(7.08, 8), facecolor='white', sharey=True)

    axs = axs.flatten()

    for i, category in enumerate(categories):

        # filter out the data for the category
        category_data = df_results[df_results['temperature_category'] == category]
        category_data_damages = df_results_damages[df_results_damages['temperature_category'] == category]

        # tick_positions = []
        # x_position = 0

        for region in regions:

            region_code = Data.R10_codes[Data.R10.index(region)]
            # filter out the data for the region
            region_data = category_data[category_data['region'] == region]
            region_data_damages = category_data_damages[category_data_damages['region'] == region]

            median_values = []
            percentile_25 = []
            percentile_75 = []
            median_values_damages = []
            percentile_25_damages = []
            percentile_75_damages = []

            for year in range(2020, 2110, 10):

                # filter out the data for the year
                year_data = region_data[str(year)]
                year_data_damages = region_data_damages[str(year)]

                median_values.append(np.median(year_data)*100)
                median_values_damages.append(np.median(year_data_damages)*100)
                percentile_25.append(np.percentile(year_data, 25)*100)
                percentile_75.append(np.percentile(year_data, 75)*100)
                percentile_25_damages.append(np.percentile(year_data_damages, 25)*100)
                percentile_75_damages.append(np.percentile(year_data_damages, 75)*100)

            # plot the data
            axs[i].plot(range(2020, 2110, 10), median_values, label=region, color=EconFeas.regional_colours[region_code], linewidth=0.75)
            axs[i].fill_between(range(2020, 2110, 10), percentile_25, percentile_75, alpha=0.2, color=EconFeas.regional_colours[region_code])

            axs[i].plot(range(2020, 2110, 10), median_values_damages, linestyle='--', color=EconFeas.regional_colours[region_code], linewidth=0.75)
            axs[i].fill_between(range(2020, 2110, 10), percentile_25_damages, percentile_75_damages, alpha=0.1, color=EconFeas.regional_colours[region_code], hatch='//')

            # plot historical values from columns '2015' to '2023'
            historical_values_region = historical_values[historical_values['region'] == region]
            historical_values_region = historical_values_region.drop(columns=['region', 'metric', 'unit'])
            historical_values_region = historical_values_region.T
            historical_values_region = historical_values_region.reset_index()
            historical_values_region.columns = ['year', 'value']
            historical_values_region['year'] = historical_values_region['year'].astype(int)

            mean_value = np.mean(historical_values_region['value'])*100
            # plot horizontal line for historical mean
            # axs[i].axhline(y=mean_value, color=EconFeas.regional_colours[region_code], linestyle='dotted', linewidth=0.75)

            axs[i].plot(historical_values_region['year'], historical_values_region['value']*100, color=EconFeas.regional_colours[region_code], linewidth=2)


        axs[i].set_title(category, fontstyle='oblique')

        # set x axis limits
        axs[i].set_xlim(2015, 2100)

        if i == 0 or i == 4:
            axs[i].set_ylabel('Investment in energy supply as a share of GDP (%) (2020-2100)')

            # make a legend
            axs[i].legend(loc='upper right', bbox_to_anchor=(1.2, 1), title='Regions', fontsize=6)

            # make legend for regions their region code
            # patches = []
            # for region in regions:
            #     region_code = Data.R10_codes[Data.R10.index(region)]
            #     patches.append(mpatches.Patch(color=EconFeas.regional_colours[region_code], label=region))

            # axs[i].legend(handles=patches, loc='upper right', bbox_to_anchor=(1.2, 1))

    plt.show()  


def temporal_subplots_north_south(north_south_investment_data, categories, start_year,
                                  end_year):
    
    """
    A function which takes the annual values of the north and south (undamaged and damaged)
    investment shares and calculates the median and IQR for each category and plots them
    by category on subplots.
    
    Inputs:
    - north_south_investment_data: dataframe with the annual investment shares for the north
    and south regions for each year both undamaged and damaged for each category
    - categories: list of categories to plot
    - start_year: the start year of the analysis
    - end_year: the end year of the analysis

    Outputs:
    - subplots with the median and IQR for each category for the north and south regions
    showing both undamaged and damaged data

    """

    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.top'] = True    
    plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.linewidth'] = 0.65
    plt.rcParams['font.size'] = 7
    plt.rcParams['axes.titlesize'] = 7
    plt.rcParams['figure.dpi'] = 150

    # set up figure with width 180mm and height of 100mm with four subplots
    fig, axs = plt.subplots(1, 4, figsize=(7.08, 3), facecolor='white', sharey=True)

    axs = axs.flatten()

    for i, category in enumerate(categories):

        # category_data = north_south_investment_data[north_south_investment_data['category'] == category]
        category_data = north_south_investment_data[north_south_investment_data['category'].isin(category)]

        years = list(range(start_year, end_year+1, 10))
        north_median_values = []
        north_percentile_5 = []
        north_percentile_25 = []
        north_percentile_75 = []
        north_percentile_95 = []
        south_percentile_5 = []
        south_median_values = []
        south_percentile_25 = []
        south_percentile_75 = []
        south_percentile_95 = []

        north_median_values_damages = []
        north_percentile_5_damages = []
        north_percentile_25_damages = []
        north_percentile_75_damages = []
        north_percentile_95_damages = []
        south_median_values_damages = []
        south_percentile_5_damages = []
        south_percentile_25_damages = []
        south_percentile_75_damages = []
        south_percentile_95_damages = []
        
        for year in range(start_year, end_year+1, 10):

            year_data = category_data[category_data['year'] == year]

            north_median_values.append(np.median(year_data['north_value'])*100)
            north_percentile_25.append(np.percentile(year_data['north_value'], 25)*100)
            north_percentile_75.append(np.percentile(year_data['north_value'], 75)*100)
            south_median_values.append(np.median(year_data['south_value'])*100)
            south_percentile_25.append(np.percentile(year_data['south_value'], 25)*100)
            south_percentile_75.append(np.percentile(year_data['south_value'], 75)*100)

            north_median_values_damages.append(np.median(year_data['north_value_damages'])*100)
            north_percentile_25_damages.append(np.percentile(year_data['north_value_damages'], 25)*100)
            north_percentile_75_damages.append(np.percentile(year_data['north_value_damages'], 75)*100)
            south_median_values_damages.append(np.median(year_data['south_value_damages'])*100)
            south_percentile_25_damages.append(np.percentile(year_data['south_value_damages'], 25)*100)
            south_percentile_75_damages.append(np.percentile(year_data['south_value_damages'], 75)*100)

        axs[i].plot(years, south_median_values/south_median_values[0], label='South', color='red', linewidth=0.75, zorder=2)
        axs[i].fill_between(years, south_percentile_25/south_median_values[0], south_percentile_75/south_median_values[0], alpha=0.3, color='red', zorder=2)
        axs[i].plot(years, north_median_values/north_median_values[0], label='North', color='blue', linewidth=0.75, zorder=1)
        axs[i].fill_between(years, north_percentile_25/north_median_values[0], north_percentile_75/north_median_values[0], alpha=0.15, color='blue', zorder=1)
        
        axs[i].plot(years, south_median_values_damages/south_median_values[0], linestyle='--', color='red', linewidth=0.75, zorder=2)
        axs[i].fill_between(years, south_percentile_25_damages/south_median_values[0], south_percentile_75_damages/south_median_values[0], alpha=0.2, color='red', hatch='//', zorder=2)
        axs[i].plot(years, north_median_values_damages/north_median_values[0], linestyle='--', color='blue', linewidth=0.75, zorder=1)
        axs[i].fill_between(years, north_percentile_25_damages/north_median_values[0], north_percentile_75_damages/north_percentile_75[0], alpha=0.1, color='blue', hatch='//', zorder=1)

        axs[i].set_title(category, fontstyle='oblique')

        # set x axis limits
        axs[i].set_xlim(start_year, end_year)

        if i == 0 or i == 4:

            axs[i].set_ylabel('Investment in energy supply as a share of GDP (%) (2020-2100)')

            # make a legend
            axs[i].legend(loc='upper right', bbox_to_anchor=(1.2, 1), title='Regions', fontsize=6)

    plt.show()


def temporal_subplots_select_scenarios(df_results, df_results_damages, scenario_models=dict, 
                                       colours=list):
    
    """
    Single panel plot showing the damaged and undamaged results for the selected scenarios
    globally.

    Inputs:
    - scenario_models: dictionary with the scenarios and models to plot and their names
    - df_results: dataframe with the results of the analysis
    - df_results_damages: dataframe with the results of the analysis with damages applied

    Outputs:
    - single panel plot showing the damaged and undamaged results for the selected scenarios
    globally.
    
    """
    damages_5th = EconData.global_econ_results_damages_5
    damages_95th = EconData.global_econ_results_damages_95

    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.linewidth'] = 0.65
    plt.rcParams['font.size'] = 7
    plt.rcParams['axes.titlesize'] = 7
    plt.rcParams['figure.dpi'] = 150

    # set up figure to occupy 1/4 with width 180mm and height of 180mm
    fig, axs = plt.subplots(1, 1, figsize=(3.54, 3.54), facecolor='white')

    scenarios = scenario_models['scenarios']
    models = scenario_models['models']
    names = scenario_models['names']

    # loop through the items in the dictionary
    for i in range(0, len(scenarios)):

        scenario = scenarios[i]
        model = models[i]
        name = names[i]
        scenario_model_df = df_results[(df_results['scenario'] == scenario) & (df_results['model'] == model)]
        scenario_model_df = scenario_model_df.drop(columns=['scenario', 'model', 'temperature_category', 'max_value',
                                                            'mean_value', 'mean_value_2050', 'largest_annual_increase',
                                                            'mean_annual_increase', 'Unnamed: 0']).T
        scenario_model_df_damages = df_results_damages[(df_results_damages['scenario'] == scenario) & (df_results_damages['model'] == model)]
        scenario_model_df_damages = scenario_model_df_damages.drop(columns=['scenario', 'model', 'temperature_category', 'max_value',
                                                    'mean_value', 'mean_value_2050', 'largest_annual_increase',
                                                    'mean_annual_increase', 'Unnamed: 0']).T
        scenario_model_df_damages_5 = damages_5th[(damages_5th['scenario'] == scenario) & (damages_5th['model'] == model)]
        scenario_model_df_damages_5 = scenario_model_df_damages_5.drop(columns=['scenario', 'model', 'temperature_category', 'max_value',
                                                    'mean_value', 'mean_value_2050', 'largest_annual_increase',
                                                    'mean_annual_increase', 'Unnamed: 0']).T
        scenario_model_df_damages_95 = damages_95th[(damages_95th['scenario'] == scenario) & (damages_95th['model'] == model)]
        scenario_model_df_damages_95 = scenario_model_df_damages_95.drop(columns=['scenario', 'model', 'temperature_category', 'max_value',
                                                    'mean_value', 'mean_value_2050', 'largest_annual_increase',
                                                    'mean_annual_increase', 'Unnamed: 0']).T
        scenario_model_df  = scenario_model_df.reset_index()
        scenario_model_df_damages  = scenario_model_df_damages.reset_index()
        scenario_model_df_damages_5  = scenario_model_df_damages_5.reset_index()
        scenario_model_df_damages_95  = scenario_model_df_damages_95.reset_index()
        scenario_model_df.columns = ['year', 'value']
        scenario_model_df_damages.columns = ['year', 'value']
        scenario_model_df_damages_5.columns = ['year', 'value']
        scenario_model_df_damages_95.columns = ['year', 'value']       

        # get the 2020 value
        value_2020 = scenario_model_df[scenario_model_df['year'] == '2020']['value'].values[0]
        
        # plot the data
        axs.plot(range(2020, 2110, 10), scenario_model_df['value'] / value_2020, label=name, color=colours[i], linewidth=0.8)
        axs.plot(range(2020, 2110, 10), scenario_model_df_damages['value'] / value_2020, linestyle='--', color=colours[i], linewidth=0.8)

        # fill between the 5th and 95th percentiles
        axs.fill_between(range(2020, 2110, 10), scenario_model_df_damages_5['value'] / value_2020, scenario_model_df_damages_95['value'] / value_2020, alpha=0.1, color=colours[i])
        
        axs.set_xlim(2020, 2100)
        axs.set_ylim(0.4, 2)

    axs.legend(loc='upper right', title='Illustrative Scenarios', fontsize=6, frameon=False)  
    axs.set_ylabel('Energy supply investment as \% GDP, 2020=1')

    plt.show()


# plot that plots three subplots, one for each select scenario, but for global north and south
def temporal_subplots_select_scenarios_north_south(df_results_5th, df_results_50th, df_results_95th, scenario_models=dict,
                                                   ):
    
    """
    Function that takes selected scenarios and models and plots the results for the \
    global north and south, for the 5th, 50th and 95th percentiles.

    Inputs:
    - df_results_5th: dataframe with the 5th percentile results
    - df_results_50th: dataframe with the 50th percentile results
    - df_results_95th: dataframe with the 95th percentile results
    - scenario_models: dictionary with the scenarios and models to plot and their names
    - colours: list of colours to use for the plot
    
    """

    # set up the plot
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.linewidth'] = 0.65
    plt.rcParams['font.size'] = 7
    plt.rcParams['axes.titlesize'] = 7
    plt.rcParams['figure.dpi'] = 150

    # set up figure to occupy 1/4 with width 180mm and height of 180mm
    fig, axs = plt.subplots(3, 1, figsize=(3.54, 3.54), facecolor='white', sharex=True)

    scenarios = scenario_models['scenarios']
    models = scenario_models['models']
    names = scenario_models['names']

    for i in range(0, len(scenarios)):

        # extract the scenario and model        
        scenario = scenarios[i]
        model = models[i]
        name = names[i]

        # extract the data for the scenario and model
        scenario_model_df_5th = df_results_5th[(df_results_5th['scenario'] == scenario) & (df_results_5th['model'] == model)]
        scenario_model_df_50th = df_results_50th[(df_results_50th['scenario'] == scenario) & (df_results_50th['model'] == model)]
        scenario_model_df_95th = df_results_95th[(df_results_95th['scenario'] == scenario) & (df_results_95th['model'] == model)]

        # get the north and south values undamaged
        north_values_ = scenario_model_df_5th['north_value'].values
        south_values_ = scenario_model_df_5th['south_value'].values

        # get the north and south values damaged 5th
        north_values_damages_5 = scenario_model_df_5th['north_value_damages'].values
        south_values_damages_5 = scenario_model_df_5th['south_value_damages'].values

        # get the north and south values damaged 50th
        north_values_damages_50 = scenario_model_df_50th['north_value_damages'].values
        south_values_damages_50 = scenario_model_df_50th['south_value_damages'].values

        # get the north and south values damaged 95th
        north_values_damages_95 = scenario_model_df_95th['north_value_damages'].values
        south_values_damages_95 = scenario_model_df_95th['south_value_damages'].values

        # get the 2020 value
        north_value_2020 = north_values_[0]
        south_value_2020 = south_values_[0]

        # plot the data
        axs[i].plot(range(2020, 2110, 10), north_values_ / north_value_2020, label='North', color='blue', linewidth=0.5)
        axs[i].plot(range(2020, 2110, 10), south_values_ / south_value_2020, label='South', color='red', linewidth=0.5)

        axs[i].plot(range(2020, 2110, 10), north_values_damages_50 / north_value_2020, linestyle='--', color='blue', linewidth=0.5)
        axs[i].plot(range(2020, 2110, 10), south_values_damages_50 / south_value_2020, linestyle='--', color='red', linewidth=0.5)

        axs[i].fill_between(range(2020, 2110, 10), north_values_damages_5 / north_value_2020, north_values_damages_95 / north_value_2020, alpha=0.2, color='blue')
        axs[i].fill_between(range(2020, 2110, 10), south_values_damages_5 / south_value_2020, south_values_damages_95 / south_value_2020, alpha=0.2, color='red')

        axs[i].set_xlim(2020, 2100)

        axs[i].set_title(name, fontstyle='oblique')

        if i == 2:
            axs[i].set_xlabel('Year')
        
        axs[i].set_ylabel('Energy supply investment as \% GDP, 2020=1')

        # axs[i].set_ylim(0.25, 3)

    axs[0].legend(loc='upper right', fontsize=5, frameon=False)
        
    plt.show()


def temporal_subplots_north_south_example_regions(north_south_investment_data, 
                                                  regional_data,
                                                  regional_data_damages,
                                                  categories, 
                                                  start_year,
                                                  end_year, 
                                                  north_south=str, 
                                                  example_regions=list):
    
    """
    A function which takes the annual values of the north and south (undamaged and damaged)
    investment shares and calculates the median and IQR for each category and plots them
    by category on subplots.
    
    Inputs:
    - north_south_investment_data: dataframe with the annual investment shares for the north
    and south regions for each year both undamaged and damaged for each category
    - categories: list of categories to plot
    - start_year: the start year of the analysis
    - end_year: the end year of the analysis

    Outputs:
    - subplots with the median and IQR for each category for the north and south regions
    showing both undamaged and damaged data

    """

    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.top'] = True    
    plt.rcParams['ytick.right'] = True
    plt.rcParams['axes.linewidth'] = 0.65
    plt.rcParams['font.size'] = 7
    plt.rcParams['axes.titlesize'] = 7
    plt.rcParams['figure.dpi'] = 150

    # set up figure with width 180mm and height of 100mm with four subplots
    fig, axs = plt.subplots(1, 4, figsize=(7.08, 3), facecolor='white', sharey=True)

    axs = axs.flatten()

    for i, category in enumerate(categories):

        # category_data = north_south_investment_data[north_south_investment_data['category'] == category]
        category_data = north_south_investment_data[north_south_investment_data['category'].isin(category)]
        # filter out the data for the category
        category_data_regional = regional_data[regional_data['temperature_category'].isin(category)]
        category_data_regional_damages = regional_data_damages[regional_data_damages['temperature_category'].isin(category)]

        years = list(range(start_year, end_year+1, 10))
        north_median_values = []
        north_percentile_5 = []
        north_percentile_25 = []
        north_percentile_75 = []
        north_percentile_95 = []
        south_percentile_5 = []
        south_median_values = []
        south_percentile_25 = []
        south_percentile_75 = []
        south_percentile_95 = []

        north_median_values_damages = []
        north_percentile_5_damages = []
        north_percentile_25_damages = []
        north_percentile_75_damages = []
        north_percentile_95_damages = []
        south_median_values_damages = []
        south_percentile_5_damages = []
        south_percentile_25_damages = []
        south_percentile_75_damages = []
        south_percentile_95_damages = []
        
        for year in range(start_year, end_year+1, 10):

            year_data = category_data[category_data['year'] == year]

            north_median_values.append(np.median(year_data['north_value'])*100)
            north_percentile_25.append(np.percentile(year_data['north_value'], 25)*100)
            north_percentile_75.append(np.percentile(year_data['north_value'], 75)*100)
            south_median_values.append(np.median(year_data['south_value'])*100)
            south_percentile_25.append(np.percentile(year_data['south_value'], 25)*100)
            south_percentile_75.append(np.percentile(year_data['south_value'], 75)*100)

            north_median_values_damages.append(np.median(year_data['north_value_damages'])*100)
            north_percentile_25_damages.append(np.percentile(year_data['north_value_damages'], 25)*100)
            north_percentile_75_damages.append(np.percentile(year_data['north_value_damages'], 75)*100)
            south_median_values_damages.append(np.median(year_data['south_value_damages'])*100)
            south_percentile_25_damages.append(np.percentile(year_data['south_value_damages'], 25)*100)
            south_percentile_75_damages.append(np.percentile(year_data['south_value_damages'], 75)*100)

        if north_south == 'North': 
        
            axs[i].plot(years, north_median_values/north_median_values[0], label='North', color='blue', linewidth=0.75, zorder=1)
            axs[i].fill_between(years, north_percentile_25/north_median_values[0], north_percentile_75/north_median_values[0], alpha=0.15, color='blue', zorder=1)
            axs[i].plot(years, north_median_values_damages/north_median_values[0], linestyle='--', color='blue', linewidth=0.75, zorder=1)
            axs[i].fill_between(years, north_percentile_25_damages/north_median_values[0], north_percentile_75_damages/north_percentile_75[0], alpha=0.1, color='blue', hatch='//', zorder=1)

        if north_south == 'South':
        
            axs[i].plot(years, south_median_values/south_median_values[0], label='South', color='red', linewidth=0.75, zorder=2)
            axs[i].fill_between(years, south_percentile_25/south_median_values[0], south_percentile_75/south_median_values[0], alpha=0.3, color='red', zorder=2)
            axs[i].plot(years, south_median_values_damages/south_median_values[0], linestyle='--', color='red', linewidth=0.75, zorder=2)
            axs[i].fill_between(years, south_percentile_25_damages/south_median_values[0], south_percentile_75_damages/south_median_values[0], alpha=0.2, color='red', hatch='//', zorder=2)
       
        else:
            pass

        axs[i].set_title(category, fontstyle='oblique')

        # set x axis limits
        axs[i].set_xlim(start_year, end_year)

        if i == 0 or i == 4:

            axs[i].set_ylabel('Investment in energy supply as a share of GDP (%) (2020-2100)')

            # make a legend
            axs[i].legend(loc='upper right', bbox_to_anchor=(1.2, 1), title='Regions', fontsize=6)

        for region in example_regions:

            region_full = Data.R10[Data.R10_codes.index(region)]
            # filter out the data for the region
            region_data = category_data_regional[category_data_regional['region'] == region_full]
            region_data_damages = category_data_regional_damages[category_data_regional_damages['region'] == region_full]

            median_values = []
            percentile_25 = []
            percentile_75 = []
            median_values_damages = []
            percentile_25_damages = []
            percentile_75_damages = []

            for year in range(2020, 2110, 10):

                # filter out the data for the year
                year_data = region_data[str(year)]
                year_data_damages = region_data_damages[str(year)]

                median_values.append(np.median(year_data)*100)
                median_values_damages.append(np.median(year_data_damages)*100)
                percentile_25.append(np.percentile(year_data, 25)*100)
                percentile_75.append(np.percentile(year_data, 75)*100)
                percentile_25_damages.append(np.percentile(year_data_damages, 25)*100)
                percentile_75_damages.append(np.percentile(year_data_damages, 75)*100)

            # plot the data
            axs[i].plot(range(2020, 2110, 10), median_values/median_values[0], label=region, color=EconFeas.regional_colours[region], linewidth=0.75)
            axs[i].fill_between(range(2020, 2110, 10), percentile_25/median_values[0], percentile_75/median_values[0], alpha=0.2, color=EconFeas.regional_colours[region])

            axs[i].plot(range(2020, 2110, 10), median_values_damages/median_values[0], linestyle='--', color=EconFeas.regional_colours[region], linewidth=0.75)
            axs[i].fill_between(range(2020, 2110, 10), percentile_25_damages/median_values[0], percentile_75_damages/median_values[0], alpha=0.1, color=EconFeas.regional_colours[region], hatch='//')

            # add legend
            axs[i].legend(loc='upper right', bbox_to_anchor=(1.2, 1), title='Regions', fontsize=6)

    plt.show()



if __name__ == "__main__":
    main()