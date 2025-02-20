import numpy as np
import pyam
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import country_converter as coco
from matplotlib import rcParams
from utils import Data
from utils import Utils
from src.constants import *
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
import matplotlib.patches as mpatches


class EconFeas:
    

    variables=['GDP|PPP', 'Investment', 'Investment|Energy Supply']
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


    # Regional Data
    regional_scenarios_models = pd.read_csv('econ/scenarios_models_R10.csv')
    regional_econ_pyamdf = pyam.IamDataFrame(data='econ/pyamdf_econ_analysis_R10.csv')
    regional_damage_estimates = pd.read_csv('econ/R10_damages_temps.csv')
    regional_categories = CATEGORIES_ALL[:7]
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
                                                        region_codes=R10_CODES,
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





if __name__ == "__main__":
    main()