import numpy as np
import pyam
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
from utils import Data
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
    econ_scenarios = pd.read_csv('scenarios_investment_all_World.csv')
    # econ_data = pyam.IamDataFrame(data="cat_meta['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'].csv")
    run_mode = 'cat'
    alpha = -0.037
    beta = -0.0018
    warming_variable = 'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile' 
    present_warming = 1.25
    # AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|95.0th Percentile

def main() -> None:

    # data_download()
    # plot_outputs()
    # plot_using_pyam()
    # violin_plots()
    # assess_variable_data()
    # energy_supply_investment_score(Data.dimensions_pyamdf, 0.023, 2100, Data.model_scenarios, Data.categories)
    energy_supply_investment_analysis(0.023, 2100, EconFeas.econ_scenarios)

def assess_variable_data():

    # Main query of AR6 data, returns a IamDataFrame object
    df = EconFeas.connAr6.query(model='*', scenario='*',
                variable=['GDP|PPP', 'Investment', 'Investment|Energy Supply'])

    df_filtered = df.filter(region='World', Category=EconFeas.categories)

        
    # Filter out by range of values from column header
    df_filtered_2050 = df_filtered.filter()

    
    # Export all query data to a pandas dataframe
    pddf = df_filtered_2050.as_pandas(meta_cols=True)
    
    
    # export to csv for analysis
    pddf.to_csv('ar6_econfeas.csv')
    
    # print(df_filtered_2050, len(df_filtered_2050))

    # # Find the number of scenarios reporting each variable in the year 2100

    # print('The number of C1-C5 scenarios reporting Energy Supply investment in 2050 is', len(df_filtered_2050.filter(variable='Investment|Energy Supply')))
    # print('The number of C1-C5 scenarios reporting GDP in 2050 is', len(df_filtered_2050.filter(variable='GDP|PPP')))
    # print('The number of C1-C5 scenarios reporting total investment in 2050 is', len(df_filtered_2050.filter(variable='Investment')))

    # # print the unique model names reporting investment in 2050
    # # print(df_filtered_2050.filter(variable='Investment')['model'].unique().tolist())
    
    # # import model classification data and model types
    # model_classification = pd.read_csv('model_classification.csv')
    # model_type = model_classification['Model Type'].unique().tolist()

   
def data_download():
    # Main query of AR6 data, returns a IamDataFrame object
    df = EconFeas.connAr6.query(model='*', scenario='*',
                variable=['GDP|PPP', 'Investment', 'Investment|Energy Supply']
                )
    # Put variables in a list
    
    # print(df.variables().unique().tolist())

    # Export all query data to a pandas dataframe
    pddf = df.as_pandas(meta_cols=True)
    
        
    # Filter out by range of values from column header
    pddf = pddf[pddf['year'].between(2020, 2100)]
    
    # Filter out by desired categories C1, C2, C3, C4 & C5
    pddf = pddf[pddf['Category'].isin(EconFeas.categories)]

    # Ensure only global data is included
    pddf = pddf[pddf['region'] == 'World']

    scenario_list = pddf['scenario'].unique().tolist()

    for variables in EconFeas.variables:
        variable_df = pddf[pddf['variable'] == variables]
        print('The number of rows for ', variables, 'is: ', len(variable_df))

    complete_scenarios = []
    
    # Make a list of the scenarios that have data on all the variables
    for scenario in scenario_list:
        scenario_df = pddf[pddf['scenario'] == scenario]
        print(len(scenario_df))
        
        include = 1
        for variables in EconFeas.variables:
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
    pddf.to_csv('ar6_econfeas.csv')


def plot_using_pyam():

    # df = pyam.IamDataFrame(data='ar6_test_scenarios.csv')
    # # Plot each of the variables in the list on subplots, with different colours for

    # Main query of AR6 data, returns a IamDataFrame object
    df = EconFeas.connAr6.query(model='*', scenario='*',
                variable=['Water Consumption', 'Land Cover|Pasture', 'Land Cover|Forest',
                          'Land Cover|Cropland','Land Cover|Cropland|Energy Crops'],
                )
    
    # Filter out scenarios that do not have all the variables
    df = df.filter(variable=EconFeas.variables)

    df = df.filter(region='World')

    # Filter out by desired categories as above:
    df = df.filter(Category=EconFeas.plotting_categories)

    # Filter out values that are not in the range of years 2025 to 2100
    df = df.filter(year=range(2025, 2101))

    fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey=False, sharex=False)
    axs = axs.flatten()
    
    # for each variable, create a subplot
    variable_int = 0
    for i in EconFeas.variables:
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
    categories = EconFeas.plotting_categories
    category_colours = EconFeas.plotting_category_colours

    # filter out scenarios that are not in the categories list
    df = df[df['Category'].isin(categories)]
    
    # Plot the scenarios as lines on subplots for each variable. 
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey=False, sharex=False)
    axs = axs.flatten()
    for i, variables in enumerate(EconFeas.plotting_variables):
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
                if EconFeas.run_mode == 'cat':
                    # Plot the data
                    axs[i].plot(scenario_model_df['year'], scenario_model_df['value'], 
                                label=scenario, color=category_colours[category], 
                                linewidth=0.3, alpha=0.5) 
                # # Colour the lines based on the model
                # elif EconFeas.run_mode == 'model':
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

    df = pd.read_csv('ar6_econfeas.csv')
    
    snapshot_year = [2050]
    
    # # filter out scenarios that are not in the categories list
    # df = df[df['Category'].isin(EconFeas.plotting_categories)]
    
    # # Make a list of unique scenarios
    # scenario_list = df['scenario'].unique().tolist()
    
    # Make a subplot for each variable that has three violins of the scenario results at snapshot year, by category
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey=False, sharex=False)
    axs = axs.flatten()

    # for each variable, create a subplot, first create the dataset
    variable_int = 0
    for variable in EconFeas.variables:

        # Make a dataframe for each variable
        variable_df = df[df['variable'] == variable]

        # Make a dataframe for the snapshot year
        snapshot_df = variable_df[variable_df['year'].isin(snapshot_year)]


        # plot violins on the figure subplot for each category
        data= pd.DataFrame()
        category_int = 0
        for category in EconFeas.categories:
    
                category_df = snapshot_df[snapshot_df['Category'] == category]
                # reset index
                category_df = category_df.reset_index(drop=True)
                data[category_int] = category_df['value']

                # plot the violin
                sns.violinplot(data=data, ax=axs[variable_int], 
                               palette='tab10')
                category_int += 1
        
        # put x axis labels on the figure as per the categories
        axs[variable_int].set_xticklabels(EconFeas.categories)
        axs[variable_int].set_title(variable + str(snapshot_year[0]))
        
        # Set the ymin to zero
        axs[variable_int].set_ylim(bottom=0)
       
        # axs[variable_int].set_ylabel(snapshot_year[0])
        # axs[variable_int].set_xlabel('')
        # axs[variable_int].set_xticks([])
        # axs[variable_int].set_xticklabels(['c1', 'c3', 'c5'])
    
    # add legend to the figure with category colours
    

                
        variable_int += 1
    plt.show()
    
        
  


        
    
    plt.show()

# takes as an input a Pyam dataframe object with n number of scenarios in it. For each scenario it calculates both a binary 
def energy_supply_investment_score(pyam_df, base_value, end_year, scenario_model_list, categories):

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
    
    # Filter out the data for the required variables
    df = pyam_df.filter(variable=['Investment|Energy Supply','GDP|MER'])
    
    # Filter for the region
    df = df.filter(region='World')

    # Filter out the data for the required years
    df = df.filter(year=range(2020, end_year+1))

    df = df.filter(scenario=scenario_model_list['scenario'], model=scenario_model_list['model'])
    # print(scenario_model_list)
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

    output_df.to_csv('outputs/energy_supply_investment_score' + str(categories) + '.csv')

    
        # # calculate the ratio of the mean value to the base value
        # ratio = mean_value / base_value
        # print(ratio)
        
        # # Export the results to a .csv file
        # df.to_csv('investment_score.csv')
        
        #     return df


def energy_supply_investment_analysis(base_value, end_year, scenario_model_list, 
                                      apply_damages=True):

    """
    Function similar to the above but that has additional indicator and different
    outputs required for analysis beyond the assessment framework.
    """
    # connAr6 = pyam.iiasa.Connection(name='ar6-public', 
    #                 creds=None, 
    #                 auth_url='https://api.manager.ece.iiasa.ac.at') 
    # Filter out the data for the required variables
    df = pyam.IamDataFrame(data="cat_df['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'].csv")
    
    df = df.filter(variable=['Investment|Energy Supply','GDP|MER', EconFeas.warming_variable], 
                        region='World', year=range(2020, end_year+1), 
                        scenario=scenario_model_list['scenario'], 
                        model=scenario_model_list['model'])
    
    meta_data = pd.read_csv("cat_meta['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'].csv")

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



if __name__ == "__main__":
    main()