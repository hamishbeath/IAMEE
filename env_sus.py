import numpy as np
import pyam
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
from utils import Utils
from utils import Data
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# Defines simple weighting system for indicators of sustainability, adjusting the weights will change the spread of how scenarios 
# within each temperature category will score. The idea is to tease out tradeoffs and synergies. 
class EnvSus:
    
    sustainability_weighting = {'water': 0.2, 'land': 0.2, 'resources': 0.2, 'biodiversity': 0.2, 'GHGs': 0.2}

    def __init__(self, water, land, resources, biodiversity, GHGs):
        self.water = water
        self.land = land
        self.resources = resources
        self.biodiversity = biodiversity
        self.GHGs = GHGs


# import the scenario data for C1, C2, C3 and C4

    # connAr6 = pyam.iiasa.Connection(name='ar6-public', 
    #                                 creds=None, 
    #                                 auth_url='https://api.manager.ece.iiasa.ac.at')

    categories = ['C1', 'C2', 'C3', 'C4', 'C5']
    category_subset_paris = ['C1a_NZGHGs']
    categories = ['C1']
    # variable=['Water Consumption', 'Land Cover|Pasture', 'Land Cover|Forest', 
    #           'Land Cover', 'Land Cover|Cropland', 'Land Cover|Cropland', 
    #           'Ecotoxicity|Marine|Electricity', 'Material recycling|Plastics']
    variables=['Water Consumption', 'Land Cover|Pasture', 'Land Cover|Forest', 
               'Land Cover|Cropland', 'Land Cover|Cropland|Energy Crops']
    plotting_variables = ['Water Consumption', 'Land Cover|Pasture', 'Land Cover|Forest',
                          'Land Cover|Cropland', 'Land Cover|Cropland|Energy Crops']            
    emissions = ['Emissions|CO2']
    plotting_categories = ['C1', 'C3', 'C5']
    # plotting_categories = ['C3']
    plotting_category_colours = {'C1':'#f57200', 'C3':'#6302d9', 'C5':'#1b9e77'}
    violin_colours = ['#8CCFF4','#7FCACC','#006B7F']
    run_mode = 'cat'
    regions = ['World', 'Countries of Sub-Saharan Africa']
    region = ['Countries of Sub-Saharan Africa']
    checked_variables = pd.read_csv('variable_categories.csv')

    # beccs_threshold = 20500 # in mtCO2 / year
    beccs_threshold = 13000 # in mtCO2 / year

def main() -> None:

    # data_download()
    # plot_outputs()
    # plot_using_pyam()
    # violin_plots()
    # Utils.simple_stats(Utils, 'AR6', EnvSus.regions, EnvSus.emissions, EnvSus.categories)
    # joel_data_download()
    # Utils.export_variable_list(Utils, 'AR6', ['C1', 'C2'])
    # Utils.create_variable_sheet(Utils,
    #                              'AR6',
    #                              EnvSus.category_subset_paris,
    #                                regions=EnvSus.regions, 
    #                                variables=EnvSus.checked_variables['variable'].tolist(), 
    #                                variable_sheet=EnvSus.checked_variables)
    # Utils.test_coco()
    # Utils.snapshot_cluster_analysis(Utils, 'World', Data.c1aR10_scenarios,['Land Cover|Forest', 'Land Cover|Cropland'],'C1a_NZGHGs' , 3, 2100)
    # Utils.time_series_cluster_analysis(Utils, 'World', Data.c1aR10_scenarios,['Land Cover|Forest', 'Land Cover|Cropland'],'C1a_NZGHGs' , 4)

    # make_scenario_project_list()
    # Utils.manadory_variables_scenarios(Utils, ['C1','C2'], EnvSus.regions, Data.mandatory_variables, subset=False)
    empty_df = pd.DataFrame()
    for region in Data.R10:
        to_append = forest_cover_change(Data.regional_dimensions_pyamdf, 2100, Data.model_scenarios, EnvSus.beccs_threshold, Data.categories, regional=region)    
        empty_df = pd.concat([empty_df, to_append], ignore_index=True, axis=0)
    empty_df.to_csv('outputs/environmental_metrics_regional' + str(Data.categories) + '.csv')
    # Utils.filter_data_sheet_variable_prevelance(Utils, 'C1a_NZGHGs', EnvSus.region, Data.mandatory_variables)


def forest_cover_change(pyam_df, end_year, scenario_model_list, beccs_threshold, categories, regional=None):
    """
    This function calculates the change in forest cover from 2020 to 2050 and 2020 to 2100 for a given scenario and model.
    The function also checks if the BECCS threshold is breached for the given scenario and model.
    
    Inputs:
    pyam_df: A pyam dataframe object with the scenario timeseries data
    end_year: The final year of the analysis
    scenario_model_list: A .csv file with the scenario and model names
    beccs_threshold: The threshold for BECCS in mtCO2/year
    categories: The categories of the scenarios
    regional: The region for which the analysis is done (None for global analysis)

    Outputs: 
    A .csv file with the forest cover change values and whether the BECCS threshold is breached
    
    """

    # Check if a regional filter is applied
    if regional is not None:
        region = regional
        
        # calculate the beccs threshold for the region
        print('Calculating the BECCS threshold for the region: ', region)
        df = pyam_df.filter(variable='Land Cover',region=[region,'World'],
                        year=2020,
                        scenario=scenario_model_list['scenario'], 
                        model=scenario_model_list['model'])
        world_land_cover = np.mean(df['value'][df['region'] == 'World'].values)
        region_land_cover = np.mean(df['value'][df['region'] == region].values)
        share_of_beccs = int(region_land_cover) / int(world_land_cover)
        beccs_threshold = (region_land_cover / world_land_cover) * beccs_threshold
        
    else:
        region = 'World'
    
    # filter for the variables needed
    df = pyam_df.filter(variable=['Land Cover|Forest','Land Cover','Carbon Sequestration|CCS|Biomass'],region=region,
                        year=range(2020, end_year+1),
                        scenario=scenario_model_list['scenario'], 
                        model=scenario_model_list['model'])

    year_list = list(range(2020, end_year+1, 10))

    beccs_threshold_breached = []
    forest_change_2050 = []
    forest_change_2100 = []
    
    for scenario, model in zip(scenario_model_list['scenario'], scenario_model_list['model']):
        
        # Filter out the data for the required scenario
        scenario_df = df.filter(scenario=scenario)
        scenario__model_df = scenario_df.filter(model=model)

        base_value = 0
        forest_cover_values = []
        beccs_seq_values = []
        
        # Iterate through the years
        for year in year_list:    

            # Filter out the data for the required year
            year_df = scenario__model_df.filter(year=year)
            year_df = year_df.as_pandas()

            # extract necessary values 
            land_cover = year_df['value'][year_df['variable'] == 'Land Cover'].values
            forest_cover = year_df['value'][year_df['variable'] == 'Land Cover|Forest'].values
            beccs_seq = year_df['value'][year_df['variable'] == 'Carbon Sequestration|CCS|Biomass'].values
            share_of_forest = forest_cover[0] / land_cover[0]

            # if 2020 store as 'base' year for given scenario 
            if year == 2020:
                base_value = share_of_forest
            # for all other 
            else:
                forest_cover_values.append(share_of_forest - base_value)

            beccs_seq_values.append(beccs_seq[0])   

        # Check if the beccs threshold is breached
        if any(i > beccs_threshold for i in beccs_seq_values):
            beccs_threshold_breached.append(1)
        else:
            beccs_threshold_breached.append(0)
        
        # Append the forest cover change values to the list
        forest_change_2050.append(forest_cover_values[2])
        forest_change_2100.append(forest_cover_values[-1])


    # Create a dataframe with the mean value and mean value up to 2050
    output_df = pd.DataFrame()
    output_df['scenario'] = scenario_model_list['scenario']
    output_df['model'] = scenario_model_list['model']
    output_df['forest_change_2050'] = forest_change_2050
    output_df['forest_change_2100'] = forest_change_2100
    output_df['beccs_threshold_breached'] = beccs_threshold_breached
    
    if regional is not None:
        
        # add column for region with region in each row
        output_df['region'] = region
        return output_df
    
    else:
        output_df.to_csv('outputs/environmental_metrics' + str(categories) + '.csv')

        

def make_scenario_project_list():
   

    # get data
    df = EnvSus.connAr6.query(model='*', scenario='*',
                variable='Emissions|CO2', region='World', year=2010)

    df_pd = df.as_pandas(meta_cols=True)

    df_pd = df_pd[['scenario', 'model', 'Project_study']]    
    
    # # new dataframe
    # list_df = pd.DataFrame()
    # list_df['scenario'], list_df['model'], list_df['Project_study'] = df['scenario'], df['model'], df['Project_study']
    
    # remove identical rows
    df_pd = df_pd.drop_duplicates()
    
    print(df_pd)

    # export to csv
    df_pd.to_csv('scenario_project_list.csv')
      
def data_download():
    # Main query of AR6 data, returns a IamDataFrame object
    df = EnvSus.connAr6.query(model='*', scenario='*',
                variable=['Water Consumption', 'Land Cover|Pasture', 'Land Cover|Forest',
                          'Land Cover', 'Land Cover|Cropland',
                          'Land Cover|Cropland|Energy Crops'],
                )
    # Put variables in a list
    
    # print(df.variables().unique().tolist())

    # Export all query data to a pandas dataframe
    pddf = df.as_pandas(meta_cols=True)
    
        
    # Filter out by range of values from column header
    pddf = pddf[pddf['year'].between(2020, 2100)]
    
    # Filter out by desired categories C1, C2, C3, C4 & C5
    pddf = pddf[pddf['Category'].isin(EnvSus.categories)]

    # Ensure only global data is included
    pddf = pddf[pddf['region'] == 'World']

    scenario_list = pddf['scenario'].unique().tolist()

    for variables in EnvSus.variables:
        variable_df = pddf[pddf['variable'] == variables]
        print('The number of rows for ', variables, 'is: ', len(variable_df))

    complete_scenarios = []
    
    # Make a list of the scenarios that have data on all the variables
    for scenario in scenario_list:
        scenario_df = pddf[pddf['scenario'] == scenario]
        print(len(scenario_df))
        
        include = 1
        for variables in EnvSus.variables:
            variable_df = scenario_df[scenario_df['variable'] == variables]
            print('The number of rows for ', variables, 'is: ', len(variable_df))
            if len(variable_df) == 0:
                print(scenario, 'is missing ', variables)
                include = 0
        
        if include == 1:
            complete_scenarios.append(scenario)
    
    # Filter out scenarios that don't have data on all the variables
    
    print(complete_scenarios)
    pddf = pddf[pddf['scenario'].isin(complete_scenarios)]


    print(pddf)
    pddf.to_csv('ar6_test_scenarios.csv')



    # df = pyam.IamDataFrame(data='ar6_test_scenarios.csv')
    # # Plot each of the variables in the list on subplots, with different colours for

    # Main query of AR6 data, returns a IamDataFrame object
    df = EnvSus.connAr6.query(model='*', scenario='*',
                variable=['Water Consumption', 'Land Cover|Pasture', 'Land Cover|Forest',
                          'Land Cover|Cropland','Land Cover|Cropland|Energy Crops'],
                )
    
    # Filter out scenarios that do not have all the variables
    df = df.filter(variable=EnvSus.variables)

    df = df.filter(region='World')

    # Filter out by desired categories as above:
    df = df.filter(Category=EnvSus.plotting_categories)

    # Filter out values that are not in the range of years 2025 to 2100
    df = df.filter(year=range(2025, 2101))

    fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey=False, sharex=False)
    axs = axs.flatten()
    
    # for each variable, create a subplot
    variable_int = 0
    for i in EnvSus.variables:
        print(i)    
        df.filter(variable=i).plot(color='model', legend=False, ax=axs[variable_int],
                                                        title=i,
                                                        y='value', x='year',
            )
        # set the y axis label as the unit
        axs[variable_int].set_ylabel(df.filter(variable=i).unit[0])
        
        # Set the x-axis limits
        axs[variable_int].set_xlim(2025, 2100)

        # reduce the line width
        for line in axs[variable_int].lines:
            line.set_linewidth(0.3)

        variable_int += 1
    
    fig.delaxes(axs[5])
    plt.show()

# Plot the data showing warming categories and different variables
def plot_outputs():

    
    df = pd.read_csv('ar6_test_scenarios.csv')

    # Make a list of unique scenarios
    scenario_list = df['scenario'].unique().tolist()
    categories = EnvSus.plotting_categories
    category_colours = EnvSus.plotting_category_colours

    # filter out scenarios that are not in the categories list
    df = df[df['Category'].isin(categories)]
    
    # Plot the scenarios as lines on subplots for each variable. 
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey=False, sharex=False)
    axs = axs.flatten()
    for i, variables in enumerate(EnvSus.plotting_variables):
        for scenario in scenario_list:
            
            scenario_df = df[df['scenario'] == scenario]
            scenario_df = scenario_df[scenario_df['variable'] == variables]

            # Make a unique dataframe for each model run of scenario
            scenario_models = scenario_df['model'].unique().tolist()
            for model in scenario_models:
                
                scenario_model_df = scenario_df[scenario_df['model'] == model]
                
                # Make a variable of the category for the scenario
                category = scenario_df['Category'].unique().tolist()
                category = category[0]

                # Get units for the variable
                units = scenario_df['unit'].unique().tolist()
                units = units[0]

                # Colour the lines based on the category
                if EnvSus.run_mode == 'cat':
                    # Plot the data
                    axs[i].plot(scenario_model_df['year'], scenario_model_df['value'], 
                                label=scenario, color=category_colours[category], 
                                linewidth=0.3, alpha=0.5) 
                # # Colour the lines based on the model
                # elif EnvSus.run_mode == 'model':
                #     # Plot the data so the color of the lines is based on the model
                #     axs[i].plot(scenario_model_df['year'], scenario_model_df['value'], 
                #                 label=scenario, linewidth=0.3, alpha=0.5, 
                #                 color=

        axs[i].set_xlim(2020, 2100)
        axs[i].set_title(variables)
        
        # set the y axis label as the unit
        axs[i].set_ylabel(units)

        # axs[i].legend()
    

    # make legend based on category colours
    handles = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in category_colours.values()]
    labels = category_colours.keys()
    fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.85, 0.25))

    fig.delaxes(axs[5])
    # plt.tight_layout()
    plt.show()

    # print(df)

# Plot seaborn violin plots with snapshots of the data at 2050 and 2100, across each variable and each category
def violin_plots():

    df = pd.read_csv('ar6_test_scenarios.csv')
    
    snapshot_year = [2050]
    
    # filter out scenarios that are not in the categories list
    df = df[df['Category'].isin(EnvSus.plotting_categories)]
    
    # Make a list of unique scenarios
    scenario_list = df['scenario'].unique().tolist()
    
    # Make a subplot for each variable that has three violins of the scenario results at snapshot year, by category
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey=False, sharex=False)
    axs = axs.flatten()

    # for each variable, create a subplot, first create the dataset
    variable_int = 0
    for variable in EnvSus.plotting_variables:

        # Make a dataframe for each variable
        variable_df = df[df['variable'] == variable]

        # Make a dataframe for the snapshot year
        snapshot_df = variable_df[variable_df['year'].isin(snapshot_year)]


        
        # Make a dataframe that has columns for each category with the corresponding
        # variable value

        # plot violins on the figure subplot for each category
        data= pd.DataFrame()
        category_int = 0
        for category in EnvSus.plotting_categories:
    
                category_df = snapshot_df[snapshot_df['Category'] == category]
                # reset index
                category_df = category_df.reset_index(drop=True)
                data[category_int] = category_df['value']

                # plot the violin
                sns.boxplot(data=data, ax=axs[variable_int], 
                palette=EnvSus.violin_colours)
                sns.stripplot(data=data, ax=axs[variable_int], 
                               color=".3")
                category_int += 1
        
        # put x axis labels on the figure as per the categories
        axs[variable_int].set_xticklabels(EnvSus.plotting_categories)
        # add a title to the subplot
        axs[variable_int].set_title(variable + str(snapshot_year[0]))
        
        # set y axis labels
        units = variable_df['unit'].unique().tolist()
        axs[variable_int].set_ylabel(units[0])
    
        # Add ticks on the right hand y axis as well as left
        axs[variable_int].yaxis.set_ticks_position('both')

    
    # remove 6th subplot
    
    # add legend to the figure with category colours
    

                
        variable_int += 1
    fig.delaxes(axs[5])
    plt.show()
    
        
  


        
    
    plt.show()


if __name__ == "__main__":
    main()