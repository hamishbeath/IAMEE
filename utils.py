import numpy as np
import pyam
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import country_converter as coco
import pickle as pkl
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
# import logging

# # Set up logging
# logging.basicConfig(level=logging.DEBUG)

cc = coco.CountryConverter()
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']


"""
This is a set of utils that can be used to do analysis, create plots, or do other things with the data, 
regardless of the variables, database or dimensions being explored.

"""

class Data:

    c1aR10_scenarios = ['EN_NPi2020_300f', 'SSP1-DACCS-1p9-3pctHR', 'SSP1-noDACCS-1p9-3pctHR', 'SSP4-noDACCS-1p9-3pctHR', 
                    'SSP1_SPA1_19I_D_LB', 'SSP1_SPA1_19I_LIRE_LB', 'SSP1_SPA1_19I_RE_LB', 'SSP2_SPA2_19I_LI', 
                    'CEMICS_GDPgrowth_1p5', 'CEMICS_HotellingConst_1p5', 'CEMICS_Linear_1p5', 
                    'LeastTotalCost_LTC_brkLR15_SSP1_P50', 'R2p1_SSP1-PkBudg900', 'R2p1_SSP5-PkBudg900', 
                    'CD-LINKS_NPi2020_400', 'PEP_1p5C_full_eff', 'PEP_1p5C_red_eff', 'CEMICS_SSP1-1p5C-fullCDR',
                        'EN_NPi2020_200f', 'EN_NPi2020_400f', 'SusDev_SDP-PkBudg1000', 'SusDev_SSP1-PkBudg900', 
                        'DeepElec_SSP2_def_Budg900', 'DISCRATE_cb400_cdrno_dr5p', 'EN_NPi2020_450f', 'EN_NPi2020_500f']

    R10 = ['Countries of Latin America and the Caribbean','Countries of South Asia; primarily India',
        'Countries of Sub-Saharan Africa', 'Countries of centrally-planned Asia; primarily China',
        'Countries of the Middle East; Iran, Iraq, Israel, Saudi Arabia, Qatar, etc.',
        'Eastern and Western Europe (i.e., the EU28)',
        'Other countries of Asia',
        'Pacific OECD', 'Reforming Economies of Eastern Europe and the Former Soviet Union; primarily Russia',
        'North America; primarily the United States of America and Canada']

    R10_codes = ['R10LATIN_AM', 'R10INDIA+', 'R10AFRICA', 'R10CHINA+', 'R10MIDDLE_EAST',
                 'R10EUROPE', 'R10REST_ASIA', 'R10PAC_OECD', 'R10REF_ECON','R10NORTH_AM'] #r10_iamc

    region_country_df = pd.read_csv('iso3c_regions.csv')
    #https://github.com/setupelz/regioniso3c/blob/main/iso3c_region_mapping_20240319.csv    


    mandatory_variables = ['Emissions|CO2', 'Investment|Energy Supply','Capacity|Electricity|Wind',
                           'Capacity|Electricity|Solar|PV', 'Final Energy',
                           'Primary Energy|Coal', 'Primary Energy|Oil', 'Primary Energy|Gas', 'Primary Energy|Nuclear',
                           'Primary Energy|Biomass', 'Primary Energy|Non-Biomass Renewables','Carbon Sequestration|CCS|Biomass',
                            'GDP|MER', 'Land Cover|Forest','Land Cover']

    mandatory_econ_variables = ['GDP|MER', 'Investment|Energy Supply',
                                'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile',
                                'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|95.0th Percentile']

    mandatory_CDR_variables = ['Carbon Sequestration|Direct Air Capture', 'Carbon Sequestration|Land Use','Carbon Sequestration|CCS|Biomass']


    narrative_variables = ['Final Energy|Transportation', 'Final Energy|Transportation|Liquids', 'Final Energy|Transportation|Liquids|Oil', 
                           'Primary Energy|Fossil', 'Land Cover|Pasture', 'Land Cover|Cropland', 'Land Cover|Forest', 'Land Cover',  'Carbon Sequestration|CCS|Biomass', 
                           'Carbon Sequestration|CCS|Fossil', 'Primary Energy|Non-Biomass Renewables', 'Primary Energy|Fossil|w/ CCS', 'Carbon Sequestration|Direct Air Capture',
                           'Primary Energy|Fossil|w/o CCS','Final Energy', 'Agricultural Demand', 'Emissions|CO2|AFOLU', 'Capacity|Electricity|Wind',
                           'Capacity|Electricity|Solar|PV','Energy Service|Transportation|Passenger', 'Energy Service|Transportation|Freight', 'Carbon Sequestration|Land Use' ]

    c1a_scenarios_selected = ['PEP_1p5C_red_eff', 'SSP1_SPA1_19I_RE_LB', 'EN_NPi2020_300f', 'EN_NPi2020_400f']
    c1a_models_selected = ['REMIND-MAgPIE 1.7-3.0', 'IMAGE 3.2', 'AIM/CGE 2.2', 'WITCH 5.0']

    dollar_2022_2010 = 0.25 # reduction in value of 2022 dollars to 2010 dollars
    # categories = ''
    categories = ['C1', 'C2']
    # categories = ['C1', 'C2', 'C3', 'C4', 'C5','C6', 'C7', 'C8']
    # model_scenarios = pd.read_csv('Countries of Sub-Saharan Africa_mandatory_variables_scenarios' + str(categories) + '.csv')
    dimensions_pyamdf = cat_df = pyam.IamDataFrame(data='cat_df' + str(categories) + '.csv')
    meta_df = pd.read_csv('cat_meta' + str(categories) + '.csv') 

    energy_variables = ['Primary Energy|Coal','Primary Energy|Oil',
                        'Primary Energy|Gas', 'Primary Energy|Nuclear',
                        'Primary Energy|Biomass', 'Primary Energy|Non-Biomass Renewables']
    econ_regions = ['Countries of Sub-Saharan Africa']
    
    try:
        regional_dimensions_pyamdf = pyam.IamDataFrame(data='pyamdf_dimensions_data_R10' + str(categories) + '.csv')
    except FileNotFoundError:
        print('No regional dimensions data found')
    
    try:
        scenario_archetypes = pd.read_csv('outputs/scenario_archetypes '+ str(categories) + '.csv')
    except:
        print('No scenario archetypes file found for the category of', categories)
    try:
        land_use_seq_data = pyam.IamDataFrame(data='land_sequestration_imputed.csv')
    except:
        print('No land use imputed data found')

class Utils:

    
    test_variables = ['Water Consumption', 'Land Cover|Pasture', 'Land Cover|Forest',
                            'Land Cover', 'Land Cover|Cropland',
                            'Land Cover|Cropland|Energy Crops']
    
    categories = ['C1', 'C2']
    # categories = ['C1', 'C2', 'C3', 'C4', 'C5','C6', 'C7', 'C8']
    

    
    # connSR15 = pyam.iiasa.Connection(name='iamc15', 
    #                             creds=None, 
    #                             auth_url='https://api.manager.ece.iiasa.ac.at')
    
    selected_variables = pd.read_csv('variables_filtered.csv')['variables'].tolist()
    
    
                           
    # Function that provides simple statistics for a given set of inputs for Pyam
    # Inputs:
    # - database (AR6 or SR15)
    # - temperature scenarios OR SSPs
    # - region
    # - variable(s)


    def simple_stats(self, db, region, variables, categories):
        
        if db == 'AR6':

            df = Utils.connAr6.query(model='*', scenario='*',
                variable=variables, region=region
                )
        if db == 'SR15':

            df = Utils.connSR15.query(model='*', scenario='*',
                variable=variables, region=region,
                )
        
        # filter by temperature category
        df = df.filter(Category=categories)

        # filter by SSP
        #df = df.filter(SSP=SSPs)
        region = []
        category = []
        variable = []


        for region in region:
            df_region = df.filter(region=region)
            # count number of scenarios, allowing for multiple models per scenario
            for variable in variables:
                df_variable = df_region.filter(variable=variable)
                total_scenario_count = 0
                model_list = df_variable['model'].unique().tolist()
                for model in model_list:
                    
                    model_df = df_variable.filter(model=model)
                    scenario_list = model_df['scenario'].unique().tolist()
                    scenario_count = len(scenario_list)
                    total_scenario_count += scenario_count

                print('Number of scenarios for', categories, 'categories with data for', 
                    variable, 'in the region of', region, 'is: ', total_scenario_count)
    
    
    def export_variable_list(self, db, categories):

        if db == 'AR6':

            df = Utils.connAr6.query(model='*', scenario='*',
                variable=Utils.selected_variables, year=2100, region='World', 
                )
            
        if db == 'SR15':

            df = Utils.connSR15.query(model='*', scenario='*',
                variable='*', year=2100, region='World'
                )
        
        # filter by temperature category
        df = df.filter(Category=categories)

        variable_list = df['variable'].unique().tolist()
        
        # Export variable list to csv
        variable_list = pd.DataFrame(variable_list)
        variable_list.columns = ['variables']
        variable_list.to_csv('variable_list_checked.csv')

        return variable_list
    
    
    def create_variable_sheet(self, db, categories, regions, variables, variable_sheet):

        """
        Creates a datasheet with stats for each variable, against the regions and categories provided

            Inputs:
            - database (AR6 or SR15)
            - temperature scenarios
            - regions
            - variables

            Outputs:
            - datasheet with stats for each region, variable and temperature category
        
        """
            
        if db == 'AR6':

            df = Utils.connAr6.query(model='*', scenario='*',
                variable=variables, region=regions, year=2100
                )
        if db == 'SR15':

            df = Utils.connSR15.query(model='*', scenario='*',
                variable=variables, region=regions, year=2100
                    )
        
        # Make dataframe with a row for each variable
        
        datasheet = pd.DataFrame(variables)
        datasheet.columns = ['variable']
        
        # loop through each temperature category
        for category in categories:
            df_category = df.filter(Category_subset=category)
            

            # Get the number of scenarios reporting emissions for CO2 in 2100 for World as a basis for %
            emissions_scenarios = 0
            df_category_emissions = df_category.filter(variable='Emissions|CO2')
            model_list_emissions = df_category_emissions['model'].unique().tolist()
            for model in model_list_emissions:
                model_emissions_df = df_category_emissions.filter(model=model)
                scenario_list_emissions = model_emissions_df['scenario'].unique().tolist()
                scenario_count_emissions = len(scenario_list_emissions)
                emissions_scenarios += scenario_count_emissions

            # loop through each region
            for region in regions:
                
                df_region = df_category.filter(region=region)

                # count number of scenarios, allowing for multiple models per scenario
                variables_count_list = []
                variables_percentage_list = []
                for variable in variables:
                    current_scenario_count = 0
                    
                    df_variable = df_region.filter(variable=variable)
                    model_list = df_variable['model'].unique().tolist()
                    for model in model_list:
                        model_df = df_variable.filter(model=model)
                        scenario_list = model_df['scenario'].unique().tolist()
                        scenario_count = len(scenario_list)
                        current_scenario_count += scenario_count

                    variables_count_list.append(current_scenario_count)
                    variables_percentage_list.append(current_scenario_count/emissions_scenarios*100)
                
                # add to datasheet
                datasheet[str(region),str(category), 'count'] = pd.Series(variables_count_list)
                datasheet[str(region),str(category), 'percentage'] = pd.Series(variables_percentage_list)

        datasheet['category_1'] = variable_sheet['category_1']
        datasheet['category_2'] = variable_sheet['category_2']
        
        # Export datasheet to csv
        datasheet.to_csv('stats_datasheet_CDR.csv')


    # function that takes as an input a list of mandatory variables and regional coverage and 
    # provides a list of scenarios that report on all of the mandatory variables for the given region
    def manadory_variables_scenarios(self, categories, regions, variables, subset=False, special_file_name=None
                                     ,call_sub=None):

        """
        Function that takes as an input a list of mandatory variables and regional coverage and 
        provides a list of scenarios that report on all of the mandatory variables for the given region

        Inputs:
        - database (AR6 or SR15)
        - temperature scenarios
        - regions
        - variables

        Outputs:
        - list of scenarios that report on all of the mandatory variables for the given region

        """
        connAr6 = pyam.iiasa.Connection(name='ar6-public', 
                            creds=None, 
                            auth_url='https://api.manager.ece.iiasa.ac.at')    

        print('Querying data')
        df = connAr6.query(model='*', scenario='*',
                variable=variables, region=regions)
        print(df)
        # ensure filtering by temperature category (subset or not)
        if subset == True:
            cat_df = df.filter(Category_subset=categories)        
        elif subset == False:
            cat_df = df.filter(Category=categories)
        else:
            raise ValueError('Subset must be a boolean')
        
        # cat_df = df.filter(Category_subset=categories)
        if call_sub == None:
            
        # except:
        # cat_df = df.filter(Category=categories)
            if special_file_name != None:
                cat_df.to_csv('cat_df' + str(categories) + special_file_name + '.csv')
                cat_meta = cat_df.as_pandas(meta_cols=True)
                cat_meta.to_csv('cat_meta' + str(categories) + special_file_name + '.csv')
            else:    
                cat_df.to_csv('cat_df' + str(categories) + '.csv')
                cat_meta = cat_df.as_pandas(meta_cols=True)
                cat_meta.to_csv('cat_meta' + str(categories) + '.csv')

        # cat_df = pyam.IamDataFrame(data='cat_df.csv')

        # Get the list of model scenario pairs reporting on all of the mandatory variables
        for region in regions:
            output_df = pd.DataFrame()
            model_list = []
            scenario_list = []
            region_df = cat_df.filter(region=region)
            for model in region_df['model'].unique().tolist():
                model_df = region_df.filter(model=model)
                # make list of available scenarios
                model_scenarios = model_df['scenario'].unique().tolist()
                for scenario in model_scenarios:
                    scenario_df = model_df.filter(scenario=scenario)
                    if scenario_df['variable'].nunique() == len(variables):
                        model_list.append(model)
                        scenario_list.append(scenario)

            output_df['model'] = model_list
            output_df['scenario'] = scenario_list

            if call_sub == None:
                if special_file_name != None:
                    output_df.to_csv(special_file_name + '_' + region + '.csv')
                else:
                    output_df.to_csv(region + '_mandatory_variables_scenarios' + str(categories) + '.csv')
            elif call_sub != None:
                return output_df

    # function that takes the pyam dataframe and loops through each mandatory variable to assess the 
    # whether or not removing the variable substantially increases the number of scenarios or models
    def filter_data_sheet_variable_prevelance(self, category, region, variables):

        cat_df = pyam.IamDataFrame(data='cat_df.csv')
        cat_df = cat_df.filter(region=region)
        # cat_df = cat_df.filter(Category_subset=category)
        variable_list = []
        model_number = []
        scenario_number = []
        
        output_df = pd.DataFrame()
        
        # Iterate through variables to see the effect of excluding each one
        for variable in variables:

            # pop out the variable from the list
            variables_copy = variables.copy()
            variables_copy.remove(variable)
            variable_df = cat_df.filter(variable=variables_copy)

            model_list = []
            scenario_list = []

            for model in variable_df['model'].unique().tolist():
                model_df = variable_df.filter(model=model)

                # make list of available scenarios
                model_scenarios = model_df['scenario'].unique().tolist()
                for scenario in model_scenarios:
                    
                    scenario_df = model_df.filter(scenario=scenario)
                    
                    # now make a list of the scenario variables
                    scenario_variables = scenario_df['variable'].unique().tolist()
                    scenario_counter = 0
                    for test_variable in variables_copy:
                        if test_variable not in scenario_variables:
                            break
                    
                        else:
                            scenario_counter += 1
                    
                    if scenario_counter == len(variables_copy):
                        scenario_list.append(scenario)
                        model_list.append(model)
        
            print('The scenarios for when excluding', variable, 'in the region of', region, 'is: ')
            # print(scenario_list)

            if variable == 'Carbon Sequestration|CCS|Fossil':

                print(scenario_list)
                print(model_list)
            # count the number of unique models in the list
            unique_models = len(set(model_list))
            # print(unique_models)
            
            model_number.append(unique_models)
            scenario_number.append(len(scenario_list))
            variable_list.append(variable)
            # print(scenario_number)

        output_df['variable'] = variable_list
        output_df['model_number'] = model_number
        output_df['scenario_number'] = scenario_number

        output_df.to_csv('variable_prevalance.csv')


    
    def data_download(variables, models, scenarios, region, categories, file_name=str):
   
        connAr6 = pyam.iiasa.Connection(name='ar6-public', 
                    creds=None, 
                    auth_url='https://api.manager.ece.iiasa.ac.at')    

        df = connAr6.query(model=models, scenario=scenarios,
            variable=variables, region=region, category=categories
            )


        df.to_csv(file_name + '.csv')
        
    def data_download_sub(variables, models, scenarios, region, end_year):

        connAr6 = pyam.iiasa.Connection(name='ar6-public', 
                    creds=None, 
                    auth_url='https://api.manager.ece.iiasa.ac.at')    

        df = connAr6.query(model='*', scenario='*', categories=Data.categories,
            variable=variables, region=region, year=range(2020, end_year+1)
            )

        return df


    # def filter_data_sheet_variable_prevelance(self, db, categories, region, threshold):

    #     """"
    #     Function that takes a datasheet from the create_variable_sheet function and filters it to only 
    #     include variables that are reported by a certain percentage of scenarios for a given region and
    #     temperature category.
    #     Inputs: 
        
        
    # Function that performs cluster analysis for a set of input variables, a set region and a set of predetermined scenarios. 
    # The cluster analysis is performed using the k-means algorithm but implements time as feature. 
    # The function returns a dataframe with the cluster labels for each scenario and the cluster centroids.



    # Cluster analysis for a set of input variables, a set region and a set of predetermined scenarios. This function performs 
    # cluster analysis using the k-means algorithm for a snapshot in time (2100) and for the entire time series.
    def snapshot_cluster_analysis(self, region, scenarios, variables, category, n_clusters, snapshot_year):
        
        # # query data
        # df = Utils.connAr6.query(model='*', scenario=scenarios,
        #     variable=variables, region=region, year=snapshot_year
        #     )

        # read in data


        # # filter by temperature category
        # df_category = df.filter(Category_subset=category)

        # # save df as a csv
        # df_category.to_csv('snapshot_data.csv')

        df_category = pyam.IamDataFrame(data='snapshot_data.csv')
        
        k_means_data = pd.DataFrame()
        for variable in variables:
            df_variable = df_category.filter(variable=variable)
            df_variable = df_variable.filter(year=snapshot_year)
            df_variable = df_variable.as_pandas()
            print(df_variable)
            # set index to scenario and model
            df_variable = df_variable.set_index(['model', 'scenario'])
            k_means_data[variable] = df_variable['value']
            
        
        print(k_means_data)


        # # arrange data in a way that is suitable for clustering so that each scenario is a row and each variable is a column
        # df_category = df_category.pivot_table(index=['model', 'scenario'], columns=variables, values='value').reset_index()
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(k_means_data[variables])

        # Get cluster assignments
        labels = kmeans.labels_

        # Get cluster centroids
        centroids = kmeans.cluster_centers_

        # visualise
        plt.scatter(k_means_data[variables[0]], k_means_data[variables[1]], c=labels, s=50, cmap='viridis')
        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.5)
        plt.xlabel(variables[0])
        plt.ylabel(variables[1])
        plt.show()


    # function that performs cluster analysis for a set of n variables, 
    # but for each year in the time series including time as a feature
    def time_series_cluster_analysis(self, region, scenarios, variables, category, n_clusters):
        
        # # query data
        # df = Utils.connAr6.query(model='*', scenario=scenarios,
        #     variable=variables, region=region
        #     )

        df_category = pyam.IamDataFrame(data='snapshot_data.csv')

        # filter by temperature category
        df_category = df_category.filter(year=range(2020, 2101))

        k_means_data = pd.DataFrame()
        for variable in variables:
            df_variable = df_category.filter(variable=variable)
            df_variable = df_variable.as_pandas()
            print(df_variable)
            # set index to scenario and model
            df_variable = df_variable.set_index(['model', 'scenario', 'year'])
            k_means_data[variable] = df_variable['value']
        
        
        
        
        # # arrange data in a way that is suitable for clustering so that each scenario is a row and each variable is a column
        # df_category = df_category.pivot_table(index=['model', 'scenario', 'year'], columns=variables, values='value').reset_index()
        
        print(k_means_data)
            # Elbow Method
        sse = []
        list_k = list(range(1, 10))

        for k in list_k:
            km = KMeans(n_clusters=k)
            km.fit(k_means_data[variables])
            sse.append(km.inertia_)

        # Plot sse against k
        plt.figure(figsize=(6, 6))
        plt.plot(list_k, sse, '-o')
        plt.xlabel(r'Number of clusters *k*')
        plt.ylabel('Sum of squared distance')
        # plt.show()

        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(k_means_data[variables])

        # # Get cluster assignments
        labels = kmeans.labels_

        # # Get cluster centroids
        centroids = kmeans.cluster_centers_

        # visualise
        plt.scatter(k_means_data[variables[0]], k_means_data[variables[1]], c=labels, s=50, cmap='viridis')
        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.5)
        plt.xlabel(variables[0])
        plt.ylabel(variables[1])
        # plt.show()

        # # add cluster labels to dataframe
        k_means_data['cluster'] = labels


        print(k_means_data)
        k_means_data_reset = k_means_data.reset_index()
                # make a column combining scenario and model values for each row 
        k_means_data_reset['scenario_model'] = k_means_data_reset['scenario'] + ' ' + k_means_data_reset['model']
        # create a plotly figure where scenarios can be toggled on and off and the cluster is indicated by colour
        fig = px.scatter(k_means_data_reset, x=variables[0], y=variables[1], color='cluster', hover_data=k_means_data_reset.columns)

        for scenario in  k_means_data_reset['scenario_model'].unique():

            # Filter the data for the scenario
            df_scenario = k_means_data_reset[k_means_data_reset['scenario_model'] == scenario]

            # Sort the data by year
            df_scenario = df_scenario.sort_values('year')

            # Add a line trace for the scenario with the same colour as the points being overlaid
            fig.add_trace(go.Scatter(x=df_scenario[variables[0]], y=df_scenario[variables[1]], mode='lines', name=scenario))

        

        # Show the plot
        fig.show()
        
        # # save dataframe as csv
        # k_means_data.to_csv('cluster_data_timeseries.csv')

        return df_category