import numpy as np
import pyam
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import pandas as pd
import mplcyberpunk
from matplotlib import rcParams
from utils import Data
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Arial']
from main import Selection
from main import IndexBuilder
# plt.rcParams['font.size'] = 7
# plt.rcParams['axes.titlesize'] = 7
# plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.major.left'] = True
plt.rcParams['ytick.major.right'] = True
# plt.rcParams['ytick.minor.visible'] = True
#plt.rcParams['ytick.labelright'] = True
#plt.rcParams['ytick.major.size'] = 0
#plt.rcParams['ytick.major.pad'] = -56
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
"""
Title: Plotting for IAM scenario anaylsis
Author: Hamish Beath
Date: November 2023
Description:
This page contains plotting functions that can be called either stand alone or 
from the main dimension specific scripts. 

"""

def main() -> None:
    
    # Plotting.polar_bar_plot_variables(Plotting, 'stats_datasheet.csv', Plotting.dimensions, 'C1a_NZGHGs')
    # Plotting.box_plot_variables(Plotting, 'variable_categories.csv', 'robust', 'C1a_NZGHGs', [2050, 2100])
    # Plotting.single_variable_box_line(Plotting, Data.c1aR10_scenarios, 'Land Cover|Forest', 'C1a_NZGHGs', 'World',years=range(2020, 2101))
    # Plotting.plot_metrics(Plotting, Plotting.bright_modern_colors, Data.model_scenarios)
    # Plotting.create_radar_plots(Plotting, 
    #                             Data.model_scenarios, 
    #                             Selection.economic_scores, 
    #                             Selection.environment_scores, 
    #                             Selection.resource_scores, 
    #                             Selection.resilience_scores, 
    #                             Selection.robustness_scores, 
    #                             Plotting.bright_modern_colors)      
    
    # Plotting.radar_plot_scenario_archetypes(Plotting, Data.model_scenarios, Selection.archetypes, Plotting.bright_modern_colors,
    #                                         Plotting.selected_scenarios)
    # Plotting.line_plot_narrative_variables('Agricultural Demand', 2020, 2100, Plotting.c1a_data,ylim_min=None, ylim_max=None, base_normalisation=True, secondary_variable=None)
    # Plotting.energy_system_stackplot(Plotting, Data.c1a_scenarios_selected, Data.c1a_models_selected, Data.dimensions_pyamdf, Plotting.c1a_data, Plotting.energy_shares_variables)
    # Plotting.transport_stackplot(Plotting, Data.c1a_scenarios_selected, Data.c1a_models_selected, Plotting.c1a_data)
    # Plotting.land_use_stacked_shares(Plotting, Data.c1a_scenarios_selected, Data.c1a_models_selected, Plotting.c1a_data)
    Plotting.CDR_stacked_shares(Plotting, Data.c1a_scenarios_selected, Data.c1a_models_selected, Plotting.c1a_data)


class Plotting:

    dimensions = ['economic', 'environment', 'resilience', 'resource', 'robust']
    dimension_names = ['Economic', 'Environment', 'Resource', 'Resilience', 'Robust']
    dimension_colours = {'economic': 'red', 'environment': 'green', 'resilience': 'blue', 'resource': 'orange', 'robust': 'purple'}
    dimension_titles = {'economic': 'Economic Feasibility', 'environment': 'Non-climate Environmental Sustainability', 'resilience': 'Societal Resilience', 'resource': 'Resource Availability', 'robust': 'Scenario Robustness'}
    dimention_cmaps = {'economic': 'Reds', 'environment': 'Greens', 'resilience': 'Blues', 'resource': 'Oranges', 'robust': 'Purples'}

    energy_shares_variables = ['Primary Energy|Fossil|w/o CCS', 
                               'Primary Energy|Fossil|w/ CCS', 
                               'Primary Energy|Nuclear', 
                               'Primary Energy|Biomass', 
                               'Primary Energy|Non-Biomass Renewables']

    energy_shares_colours = {'Primary Energy|Fossil|w/o CCS': 'darkgrey', 
                             'Primary Energy|Fossil|w/ CCS': 'lightgrey', 
                             'Primary Energy|Nuclear': 'blue', 
                             'Primary Energy|Biomass': 'green', 
                             'Primary Energy|Non-Biomass Renewables': 'lightgreen'}


    bright_modern_colors = [
    "#FF5733",  # Bright Red
    "#9933FF",  # Lavender
    "#33FF42",  # Neon Green
    "#3357FF",  # Vivid Blue
    "#F333FF",  # Magenta
    "#33FFF3",  # Cyan
    "#33FF33",  # Bright Green
    "#FF3388",  # Pink
    "#33FF88",  # Mint
    "#8833FF",  # Purple
    "#FF8833",  # Orange
    "#3399FF",  # Sky Blue
    "#99FF33",  # Lime Green
    "#FF3399",  # Hot Pink
    "#33FF99",  # Aquamarine
    "#FF9933",  # Tangerine
    "#33FFFF",  # Bright Cyan
    "#FF33FF",  # Bright Magenta
    "#FFFF33",  # Bright Yellow
    "#FF6666",  # Soft Red
    "#66FF66",  # Soft Green
    "#6666FF",  # Soft Blue
    "#FF66FF",  # Soft Magenta
    "#66FFFF",  # Soft Cyan
    "#FFFF66",  # Soft Yellow
    "#FF6F61",  # Melon
    "#6B5B95",  # Amethyst
    "#88B04B",  # Pear
    "#F7CAC9",  # Pale Pink
    "#92A8D1",  # Soft Blue Grey
    "#955251",  # Brick Red
    "#B565A7",  # Mauve
    "#009B77",  # Persian Green
    "#DD4124",  # Flame
    "#D65076",  # Raspberry
    "#45B8AC",  # Zomp
    "#EFC050",  # Marigold
    "#5B5EA6",  # Royal Blue
    "#9B2335",  # Red Berry
    "#DFCFBE",  # Almond
    "#55B4B0",  # Turquoise
    "#E15D44",  # Terra Cotta
    "#7FCDCD",  # Pale Aqua
    "#BC243C",  # Strong Red
    "#C3447A",  # Vivid Pink
    ]

    selected_scenarios = pd.read_csv('outputs/selected_scenarios' + str(Data.categories) + '.csv')
    c1a_data = pyam.IamDataFrame('c1a_selected_scenario_narrative_data.csv')

    # Create a detailed polar bar plot that categorises 
    def polar_bar_plot_variables(self, file_name, dimensions, category):
        
        variable_data = pd.read_csv(file_name)
        variable_data = variable_data[variable_data['category_1'].isin(dimensions)]

        # print(variable_data)

        for dimension in dimensions:
            
            print(dimension)
            
            # Filter the data to only include the dimension of interest, whether it is found in category_1 or category_2
            dimension_data = variable_data[(variable_data['category_1'] == dimension) | (variable_data['category_2'] == dimension)]
            
            # Rank the variables largest to smallest on the column of world value
            dimension_data = dimension_data.sort_values(by=["('World', '" + category +"', 'percentage')"], ascending=False)
            
            with plt.style.context('fivethirtyeight'):
                fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10,10))

                angles = np.linspace(0, 2*np.pi, len(dimension_data), endpoint=False)
                indexes = list(range(0, len(dimension_data)))
                width = 2*np.pi / len(dimension_data)
                angles = [element * width for element in indexes]

                label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(dimension_data))
                bars_bg = ax.bar(x = angles, height=100, width=width, color='grey',
                        edgecolor='black', zorder=1, alpha=0.1)

                rankings = dimension_data["('World', '" + category +"', 'percentage')"].tolist()    
                rankings_r5 = dimension_data["('Asian countries except Japan', '" + category +"', 'percentage')"].tolist()                
                rankings_r10 = dimension_data["('Countries of Sub-Saharan Africa', '" + category +"', 'percentage')"].tolist()

                variables = dimension_data['variable']
                bars = ax.bar(x = angles, height=rankings, width=width, color=self.dimension_colours[dimension],
                        edgecolor='white', zorder=2, alpha=0.5)
                bars_r5 = ax.bar(x = angles, height=rankings_r5, width=width, color='black',
                        edgecolor='white', zorder=3, alpha=0.3)
                bars_r10 = ax.bar(x = angles, height=rankings_r10, width=width, color='gray',
                        edgecolor='white', zorder=4, alpha=0.3)

                for angle, height, variables in zip(angles, rankings, variables):
                    rotation_angle = np.degrees(angle)
                    if angle < np.pi:
                        rotation_angle -= 90
                    elif angle == np.pi:
                        rotation_angle -= 90
                    else:
                        rotation_angle += 90
                    ax.text(angle, 105, variables, 
                            ha='center', va='bottom', 
                            rotation=rotation_angle, rotation_mode='anchor', fontsize=9)

                ax.set_xticks([])
                ax.grid(alpha=0.1, color='white', lw=3)
                plt.ylim(0, 100)
                ax.yaxis.set_zorder(10)
                ax.yaxis.set_tick_params(zorder=10)
                
                title = self.dimension_titles[dimension] + '\n% ' + category + ' Scenarios with Variables Provided'
                # add title at the top of the plot in bold
                ax.set_title(title, fontsize=13, y=1.07, fontweight='bold')
                
                # add a legend for the different regions R5 and World
                ax.legend([bars, bars_r5, bars_r10], ['World', 'R5', 'R10'], loc='upper right', bbox_to_anchor=(1.1, 1.1))

                plt.show()

                # Save the figure
                fig.savefig('figures/' + dimension + '_polar_bar_r10.png', dpi=300, bbox_inches='tight')

    # Create a box plot showing the levels of each variable for each dimension in 
    # 2020 and 2050. The box plot contains boxes for all the variables for the 
    # dimension of interest, with 2050 and 2100 values plotted in adjacent boxes. 

    def box_plot_variables(self, file_name, dimension, category, years):
        
        # import the data
        variable_data = pd.read_csv(file_name)
        
        # filter out all variables with low R10
        variable_data = variable_data[variable_data["sufficientR10"] > 0]
        
        # Filter the data to only include the dimension of interest, whether it is found in category_1 or category_2
        dimension_data = variable_data[(variable_data['category_1'] == dimension) | (variable_data['category_2'] == dimension)]

        variables = dimension_data['variable']
        variables = variables.unique()
        variables = variables.tolist()

        # import util for getting the data 
        from utils import Utils

        df = Utils.connAr6.query(model='*', scenario='*',
            variable=variables, year=[2050,2100], region=['Countries of Sub-Saharan Africa','World']
            )
        
        df_category = df.filter(Category_subset=category)
        
        
        # Find R10 scenarios
        df_R10_scenarios = df_category.filter(region='Countries of Sub-Saharan Africa')
        df_R10_scenarios = df_R10_scenarios.filter(year=2050)
        R10_scenarios = df_R10_scenarios['scenario'].unique().tolist()
        print(R10_scenarios)


        # Filter for just R10 scenarios
        df_category = df_category.filter(scenario=R10_scenarios)

        # Filter for the world
        df_category = df_category.filter(region='World')

        # set up subplots for the number of variables
        fig, axs = plt.subplots(1, len(variables), figsize=(30,10))
        axs = axs.flatten()

        # add space between subplots
        fig.subplots_adjust(wspace=0.7)

        # loop through each variable
        for i, variable in enumerate(variables):
            

            # filter the data to only include the variable of interest
            df_variable = df_category.filter(variable=variable)
            if df_variable.empty:
                continue
            
            # make a subplot for the variable
            ax = axs[i]


            units = df_variable['unit'].unique().tolist()
            units = units[0]


            data = pd.DataFrame()
            for year in range(2050, 2150, 50):
                
                print(year)
                # get the values for the variable in the year of interest
                df_values = df_variable.filter(year=year)
                
                data[year] = df_values['value']

                # set the box plot in seaborn
            sns.boxplot(data=data, ax=ax, palette=Plotting.dimention_cmaps[dimension],showfliers=False)
            sns.stripplot(data=data, ax=ax, 
            color=".3") # get the values for the variable in the year of interest

            # set the title of the plot
            ax.set_title(variable, fontsize=10,fontweight='bold')
            
            # set the units of the y axis
            ax.set_ylabel(units)

            # Add ticks on the right hand y axis as well as left
            ax.yaxis.set_ticks_position('both')
            
        # set the title of the plot
        title = 'Box plots of ' + dimension + ' variable values for '+ category + ' scenarios in 2050 and 2100'
        fig.suptitle(title, fontsize=20)
        
        
        plt.show()


    def single_variable_box_line(self, scenarios, variable, category,region, years):
        
        # import util for getting the data 
        from utils import Utils

        df = Utils.connAr6.query(model='*', scenario=scenarios,
            variable=variable, year=years, region=region
            )
        
        df_category = df.filter(Category_subset=category)
        
        # set up subplots for the box plot and line plot.
        fig = plt.figsize=(10,15)

        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3]) 

        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])


        # set the box plot in seaborn

        units = df_category['unit'].unique().tolist()
        units = units[0]
        data = pd.DataFrame()
        for year in range(2050, 2150, 50):
            
            # get the values for the variable in the year of interest
            df_values = df_category.filter(year=year)
            
            data[year] = df_values['value']

        sns.boxplot(data=data, ax=ax1, palette='cool',showfliers=False)
        sns.stripplot(data=data, ax=ax1,color=".3")     

        # set the units of the y axis
        ax1.set_ylabel(units)
        
        # Add ticks on the right hand y axis as well as left
        ax1.yaxis.set_ticks_position('both')
        ax1.xaxis.set_ticks_position('both')
        ax1.set_xlabel('Year')
        # set up the line plot showing all scenarios for the variable
        # sns.lineplot(data=df_category,x='year', y='value', hue='scenario', ax=axs[1]) #'PuBuGn'
        pyam.plotting.line(df_category, x='year', y='value', legend=None, color='scenario', ax=ax2, 
                           cmap='winter', title=False)
        
        ax2.set_xlim(2020, 2100)

        ax2.yaxis.set_ticks_position('both')
        ax2.xaxis.set_ticks_position('both')

        # set the title of the plot

        # fig.title = 'Distribution of values for ' + variable + '2020-2100'

        plt.show()

    # plot the metrics as raw values for each dimension
    def plot_metrics(self, colours, scenario_list):
    
        
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['xtick.minor.visible'] = False
        # plt.rcParams['xtick.major.visible'] = False
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['ytick.major.left'] = True
        plt.rcParams['ytick.major.right'] = True
        plt.rcParams['ytick.minor.visible'] = True
        #plt.rcParams['ytick.labelright'] = True
        #plt.rcParams['ytick.major.size'] = 0   
        #plt.rcParams['ytick.major.pad'] = -56
        plt.rcParams['xtick.top'] = False
        plt.rcParams['ytick.right'] = True


        # plot the economic data in a box plot
        investment_info = IndexBuilder.investment_metrics
        investment_info = investment_info.reset_index()
        column_list = ['mean_value', 'mean_value_2050']
        # header_list = ['mean_value', 'mean_value_2050']
        y_labels = ['Mean Ratio of Energy Supply\n Investment (2020-2100)', 'Mean Ratio of Energy Supply\n Investment (2020-2050)']
        # # make a list of numbers accending for each scenario
        zero_list = [1] * len(investment_info)
        investment_info['x'] = zero_list
        fig_length = 3 * len(column_list)
        fig, axs = plt.subplots(len(column_list), 1, figsize=(10, fig_length))
        count = 0
        for ax, investment_column in zip(axs, column_list):
            # create a box plot for the investment data with each scenario a different color
            ax.boxplot(investment_info[investment_column], showfliers=False, vert=False)
            ax.scatter(y=investment_info['x'], x=investment_info[investment_column], c=colours[:len(investment_info)], marker='o', s=100)
            ax.set_title(y_labels[count])
            # set y min and max
            ax.set_xlim(0.4, 1.3)
            # remove x tick and label
            ax.set_yticks([])
            ax.set_ylabel('')
            count += 1
        
        # make a ledgend with a swatch of each color and the scenario name and model
        # make a list of the scenario names
        label_names = []
        for label in range(0, len(investment_info)):
            label_names.append(investment_info['scenario'][label]  + '_(' + investment_info['model'][label] + ')')
        
        # make a list of the colors
        colors = colours[:len(investment_info)]
        # make a list of the markers
        markers = ['o'] * len(investment_info)
        # make a list of the sizes
        sizes = [100] * len(investment_info)
        # make a list of the labels
        labels = label_names
        swatches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
        #create a legend
        fig.legend(handles=swatches, labels=labels, loc='lower center', ncol=3, fontsize=7, frameon=False)
        plt.savefig('figures/investment_metrics_horizontal.pdf')
        # plt.show()
        
        # plot the environmental data in a box plot
        environmental_info = IndexBuilder.environment_metrics
        column_list = ['forest_change_2050', 'forest_change_2100']
        y_labels = ['Change in Forest Land Cover (2020-2050)', 'Change in Forest Land Cover (2020-2100)']
        environmental_info['x'] = zero_list
        fig, axs = plt.subplots(len(column_list),1, figsize=(fig_length, 10))
        count = 0
        for ax, environmental_column in zip(axs, column_list):
            # create a box plot for the investment data with each scenario a different color
            ax.boxplot(environmental_info[environmental_column], showfliers=False, vert=False)
            ax.scatter(y=environmental_info['x'], x=environmental_info[environmental_column], c=colours[:len(environmental_info)], marker='o', s=100)
            ax.set_title(y_labels[count])
            # set y min and max
            ax.set_xlim(0, 0.1)
            # remove x tick and label
            ax.set_yticks([])
            ax.set_ylabel('')
            count += 1
        
        fig.legend(handles=swatches, labels=labels, loc='lower center', ncol=3, fontsize=7, frameon=False)
        # plt.savefig('figures/environmental_metrics.pdf')
        

        # plot the resource data in a box plot
        resource_info = IndexBuilder.resource_metrics
        column_list = ['Nd', 'Dy', 'Cd', 'Te', 'Se', 'In']
        y_labels = ['Neodymium', 'Dysprosium', 'Cadmium', 'Tellurium', 'Selenium', 'Indium']
        resource_info['x'] = zero_list
        fig, axs = plt.subplots(1, len(column_list), figsize=(13, 10))
        count = 0
        for ax, resource_column in zip(axs, column_list):
            # create a box plot for the investment data with each scenario a different color
            ax.boxplot(resource_info[resource_column], showfliers=False)
            ax.scatter(x=resource_info['x'], y=resource_info[resource_column], c=colours[:len(resource_info)], marker='o', s=100)
            ax.set_title(y_labels[count])
            # set y min and max
            ax.set_ylim(0.1, 1.9)
            # remove x tick and label
            ax.set_xticks([])
            ax.set_xlabel('')
            count += 1
        # add title
        fig.suptitle('Mean ratio of renewables mineral usage share (to 2050) to baseline share', fontsize=14)
        fig.legend(handles=swatches, labels=labels, loc='lower center', ncol=3, fontsize=7, frameon=False)
        # plt.show()
        
        
        # plot the resilience data in a box plot
        resilience_info = IndexBuilder.final_energy_demand
        resilience_info['diversity'] = IndexBuilder.energy_diversity['shannon_index']
        column_list = ['final_energy_demand', 'diversity']
        y_labels = ['Cumulative Final Energy Demand (EJ)', 'Shannon Diversity Index']
        resilience_info['x'] = zero_list
        fig, axs = plt.subplots(len(column_list), 1, figsize=(10, fig_length))
        count = 0
        for ax, resilience_column in zip(axs, column_list):
            # create a box plot for the investment data with each scenario a different color
            ax.boxplot(resilience_info[resilience_column], showfliers=False, vert=False)
            ax.scatter(y=resilience_info['x'], x=resilience_info[resilience_column], c=colours[:len(resilience_info)], marker='o', s=100)
            ax.set_title(y_labels[count])
            # remove x tick and label
            ax.set_yticks([])
            ax.set_ylabel('')
            count += 1

        fig.legend(handles=swatches, labels=labels, loc='lower center', ncol=3, fontsize=7, frameon=False)
        # plt.show()
        # plt.savefig('figures/resilience_metrics.pdf')

        # plot the robustness data in a box plot
        robustness_info = IndexBuilder.carbon_budgets
        robustness_info['diversity'] = IndexBuilder.energy_diversity['shannon_index']
        robustness_info['flexibility'] = IndexBuilder.energy_system_flexibility['flexibility_score']
        robustness_info['x'] = zero_list
        column_list = ['carbon_budget_share', 'diversity', 'flexibility']       
        y_labels = ['Carbon Budget Share \nused by 2030', 'Shannon Diversity Index', 'Flexibility Score']
        fig, axs = plt.subplots(len(column_list), 1, figsize=(10, fig_length))
        count = 0
        for ax, robustness_column in zip(axs, column_list):
            # create a box plot for the investment data with each scenario a different color
            ax.boxplot(robustness_info[robustness_column], showfliers=False, vert=False)
            ax.scatter(y=robustness_info['x'], x=robustness_info[robustness_column], c=colours[:len(robustness_info)], marker='o', s=100)
            ax.set_title(y_labels[count])
            # remove x tick and label
            ax.set_yticks([])
            ax.set_ylabel('')
            count += 1
        
        fig.legend(handles=swatches, labels=labels, loc='lower center', ncol=3, fontsize=7, frameon=False)
        plt.show()


    def create_radar_plots(self, scenario_model_list, economic_scores, environment_scores, resource_scores,
                           resilience_scores, robustness_scores, colours):
        
        """
        some code from https://www.pythoncharts.com/matplotlib/radar-charts/
        """

        # plt.rcParams['font.size'] = 7
        radar_data = pd.DataFrame()
        radar_data['economic'] = economic_scores['investment_score']
        # normalise the economic score
        radar_data['economic'] = radar_data['economic'] / radar_data['economic'].max()
        radar_data['environment'] = environment_scores['environmental_score']
        # normalise the environment score
        radar_data['environment'] = radar_data['environment'] / radar_data['environment'].max()
        radar_data['resource'] = resource_scores['resource_score']
        # normalise the resource score
        radar_data['resource'] = radar_data['resource'] / radar_data['resource'].max()
        radar_data['resilience'] = resilience_scores['resilience_score']
        # normalise the resilience score
        radar_data['resilience'] = radar_data['resilience'] / radar_data['resilience'].max()
        radar_data['robustness'] = robustness_scores['robustness_score']
        # normalise the robustness score
        radar_data['robustness'] = radar_data['robustness'] / radar_data['robustness'].max()
        radar_data['scenario'] = economic_scores['scenario']
        radar_data['model'] = economic_scores['model']
        print(radar_data)
        # make a fig with 15 subplots, 3 rows
        fig, axs = plt.subplots(6, 3, figsize=(13, 20), subplot_kw=dict(polar=True))

        # Number of variables we're plotting.
        categories = list(radar_data)[0:5]  # get the first 5 columns as the categories
        N = len(categories)
        print(categories)
        # make a list of the scenario 
        scenario_list = radar_data['scenario'].tolist()
        
        i = 0
        # loop through each scenario
        for scenario, model in zip(scenario_model_list['scenario'], scenario_model_list['model']): 
            # Find the right subplot

            ax = axs.flatten()[i]
            scenario_item = radar_data['scenario'][i]
            model = radar_data['model'][i]
            # Compute angle each bar is centered on:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            # The plot is circular, so we need to "complete the loop" and append the start value to the end.
            stats = radar_data.loc[radar_data['model'] == model]
            stats = stats.loc[stats['scenario'] == scenario_item, categories].values.flatten().tolist()
            stats += stats[:1]
            angles += angles[:1]

            # Draw the outline of our data.
            ax.fill(angles, stats, color=colours[i], alpha=0.25)
            ax.plot(angles, stats, color=colours[i], linewidth=2)

            # # Labels for each point
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)

            #set the y limit
            ax.set_ylim(0, 1)

            # Title for each subplot with the scenario name
            ax.set_title(f"{scenario} ({model})",size=10, y=1.1, fontweight='bold')

            # # get the data for the scenario
            # scenario_data = radar_data[radar_data['scenario'] == scenario]
            # # make a radar plot for the scenario
            # self.radar_plot(ax, scenario_data, colours[i])
            # # set the title of the plot
            # ax.set_title(scenario)
            # # remove the x and y ticks
            # ax.set_xticks([])
            # ax.set_yticks([])
            # ax.set_xlabel('')
            # ax.set_ylabel('')
            i += 1
        # Show the figure
        plt.tight_layout()
        plt.savefig('figures/radar_plots' + str(Data.categories) + '.pdf')
        plt.show()
        

    def radar_plot_scenario_archetypes(self, scenario_model_list, archetypes, colours, selected_scenarios):

        # plt.rcParams['font.size'] = 7
        # Number of variables we're plotting.
        categories = list(archetypes)[0:5]
        archetype_names = ['Warning lights', 'Resource risk', 'Sustainability struggle', 'Clear path']
        archetype_colours = ['#FBA006', '#19E5FF', '#FF03FB', '#00F982']
        # make a fig with 4 subplots, 2 rows
        fig, axs = plt.subplots(2, 2, figsize=(10, 10), subplot_kw=dict(polar=True))

        N = len(categories)

        # organise the selected scenarios by cluster 0-3
        selected_scenarios = selected_scenarios.sort_values(by='cluster')

        for archetype in range(0, 4):

            ax = axs.flatten()[archetype]
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            stats = archetypes.loc[archetypes['cluster'] == archetype, categories].values.flatten().tolist()
            stats += stats[:1]
            angles += angles[:1]
            
            stats_illustrative = selected_scenarios[selected_scenarios['cluster'] == archetype].values.flatten().tolist()
            stats_illustrative = stats_illustrative[1:6]
            stats_illustrative += stats_illustrative[:1]

            # Draw the outline of our data.
            ax.fill(angles, stats, color=archetype_colours[archetype], alpha=0.27, zorder=10)
            ax.plot(angles, stats, color=archetype_colours[archetype], linewidth=3, zorder=11)
            ax.fill(angles, stats_illustrative, color='black', alpha=0.1)
            ax.plot(angles, stats_illustrative, color='black', linewidth=1)

            # # Labels for each point
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(Plotting.dimension_names)

            #set the y limit
            ax.set_ylim(0, 1)

            # Title for each subplot with the archetype name
            ax.set_title(archetype_names[archetype],size=14, y=1.1, fontweight='bold')

        # Show the figure
        plt.tight_layout()
        plt.show()


    def line_plot_narrative_variables(variable, start_year, end_year, df,ylim_min=int, ylim_max=int, base_normalisation=False, secondary_variable=None):

        plt.rcParams['ytick.minor.visible'] = True
        plt.rcParams['ytick.labelright'] = True
        scenario_colours = ['#FBA006', '#19E5FF', '#FF03FB', '#00F982']
        names = ['Warning lights', 'Resource risk', 'Sustainability struggle', 'Eco balance']
        # set up the figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # filter for the variable
        df_variable = df.filter(variable=variable)

        # get the units
        units = df_variable['unit'].unique().tolist()
        units = units[0]

        # filter for the years
        df_variable = df_variable.filter(year=range(start_year, end_year+1))

        for i, scenario in enumerate(Data.c1a_scenarios_selected):
            
            label = names[i] + ' (' + scenario + ')' 

            # filter for the scenario
            df_scenario = df_variable.filter(model=Data.c1a_models_selected[i], scenario=scenario)
            
            if base_normalisation == True:
                # df_scenario['value'] = df_scenario['value'] / df_scenario['value'].iloc[0]
                ax.plot(df_scenario['year'], (df_scenario['value']/df_scenario['value'][0]), label=label, color=scenario_colours[i], linewidth=2)
            else:
                ax.plot(df_scenario['year'], df_scenario['value'], label=label, color=scenario_colours[i], linewidth=2)
        
            if secondary_variable != None:
                df_secondary_variable = df.filter(variable=secondary_variable)
                df_secondary_variable = df_secondary_variable.filter(model=Data.c1a_models_selected[i], scenario=scenario)
                if base_normalisation == True:
                    ax.plot(df_secondary_variable['year'], (df_secondary_variable['value']/df_secondary_variable['value'][0]), label=secondary_variable, color=scenario_colours[i], linestyle='dashed', linewidth=2)
                else:
                    ax.plot(df_secondary_variable['year'], df_secondary_variable['value'], label=secondary_variable, color=scenario_colours[i], linestyle='dashed', linewidth=2)

        # set the title of the plot
        ax.set_title(variable)
        # set the x and y axis labels
        ax.set_xlabel('Year')
        
        if base_normalisation == True:
            ax.set_ylabel('Change over time (normalised to 2020)')
        else:
            ax.set_ylabel(units)

        # set x limits
        ax.set_xlim(start_year, end_year)

        # set y limits
        ax.set_ylim(ylim_min, ylim_max)


        # add a legend
        ax.legend(frameon=False)

        plt.show()
        
    
    def energy_system_stackplot(self, illustrative_scenarios, 
                                illustrative_models,
                                indicator_df, 
                                narrative_df, 
                                energy_variables):
        
        plt.rcParams['ytick.minor.visible'] = True
        names = ['Warning lights', 'Resource risk', 'Sustainability struggle', 'Eco balance']
        print(energy_variables)
        """
        this function needs :
        - set up four sublots, one for each illustrative scenario
        - to extract all the energy data from 2020-2100 for each variable and for
        each illustrative scenario, being mindful the data is coming from two different dataframes
        - interpolate the data for each energy variable
        - stack plot the data for each energy variable
        """

        # set up the figure
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

        for i in range(0, len(illustrative_scenarios)):

            plotting_df = pd.DataFrame()
            # set the scenario and model
            scenario = illustrative_scenarios[i]
            model = illustrative_models[i]
            # filter the indicator dataframe for the scenario and model
            indicator_df_scenario = indicator_df.filter(scenario=scenario, model=model, year=range(2020, 2101), region='World')
            
            # filter the narrative dataframe for the scenario and model
            narrative_df_scenario = narrative_df.filter(scenario=scenario, model=model, year=range(2020, 2101), region='World')

            # loop through each energy variable
            for variable in energy_variables:
                print(variable)
                colours = Plotting.energy_shares_colours.values()
                
                if variable == 'Primary Energy|Fossil|w/ CCS' or variable == 'Primary Energy|Fossil|w/o CCS':
                    df = narrative_df_scenario.filter(variable=variable)
                    print(df)
                else:
                    df = indicator_df_scenario.filter(variable=variable)
                    print(df)
                # interpolate the data for the variable
                interpolated_variable = df.interpolate(range(2020, 2101))

                # add the interpolated data to the plotting dataframe
                plotting_df[variable] = interpolated_variable['value']

            # make a stack plot for the scenario
            axs.flatten()[i].stackplot(range(2020, 2101), plotting_df.T, labels=energy_variables, colors=colours, alpha=0.45, edgecolor=colours)
            
            # # add line plot over the top
            # for variable in energy_variables:
            #     axs.flatten()[i].plot(range(2020, 2101), plotting_df[variable], label=variable, color=Plotting.energy_shares_colours[variable], linewidth=2)
            title = names[i] + ' (' + scenario + ')'
            # set the title of the plot
            axs.flatten()[i].set_title(title)
            # set the x and y axis labels
            axs.flatten()[i].set_xlabel('Year')
            axs.flatten()[i].set_ylabel('EJ')

            # set x limits
            axs.flatten()[i].set_xlim(2020, 2100)

            # set y limits
            axs.flatten()[i].set_ylim(0, 1100)

        colours = list(Plotting.energy_shares_colours.values())

        # create legend
        labels = energy_variables
        swatches = [mpatches.Patch(color=colours[j], label=labels[j]) for j in range(len(labels))]
        # #create a legend
        fig.legend(handles=swatches, labels=labels, loc='lower center', ncol=2, frameon=False)
        
        # add a legend
        # fig.legend(frameon=False)
        plt.show()


    def transport_stackplot(self, illustrative_scenarios, 
                                illustrative_models,
                                narrative_df
                                ):
        
        plt.rcParams['ytick.minor.visible'] = True
        names = ['Warning lights', 'Resource risk', 'Sustainability struggle', 'Eco balance']
        transport_variables = ['Final Energy|Transportation', 'Final Energy|Transportation|Liquids|Oil']
        colours = ['#3399FF', 'darkgrey']
        
        # set up the figure
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

        for i in range(0, len(illustrative_scenarios)):

            plotting_df = pd.DataFrame()
            # set the scenario and model
            scenario = illustrative_scenarios[i]
            model = illustrative_models[i]
            narrative_df_scenario = narrative_df.filter(scenario=scenario, model=model, year=range(2020, 2101), region='World')

            # loop through each energy variable
            for variable in transport_variables:
                
                if variable == 'Final Energy|Transportation':

                    df = narrative_df_scenario.filter(variable=variable)
                    df_other = narrative_df_scenario.filter(variable='Final Energy|Transportation|Liquids|Oil')
                    interpolated_variable = df.interpolate(range(2020, 2101))
                    interpolated_variable_other = df_other.interpolate(range(2020, 2101))
                    
                    # take away the oil from the total
                    interpolated_variable = interpolated_variable.data.copy()
                    interpolated_variable_other = interpolated_variable_other.data.copy()
                    interpolated_variable['value'] = interpolated_variable['value'] - interpolated_variable_other['value']
                    plotting_df[variable] = interpolated_variable['value']

                else:
                    df = narrative_df_scenario.filter(variable=variable)
                    interpolated_variable = df.interpolate(range(2020, 2101))
                    plotting_df[variable] = interpolated_variable['value']

            # make a stack plot for the scenario
            axs.flatten()[i].stackplot(range(2020, 2101), plotting_df.T, labels=transport_variables, colors=colours, alpha=0.55, edgecolor='darkgrey')
            
            # # add line plot over the top
            # for variable in energy_variables:
            #     axs.flatten()[i].plot(range(2020, 2101), plotting_df[variable], label=variable, color=Plotting.energy_shares_colours[variable], linewidth=2)
            title = names[i] + ' (' + scenario + ')'
            # set the title of the plot
            axs.flatten()[i].set_title(title)
            # set the x and y axis labels
            axs.flatten()[i].set_xlabel('Year')
            axs.flatten()[i].set_ylabel('EJ')

            # set x limits
            axs.flatten()[i].set_xlim(2020, 2100)

            # set y limits
            axs.flatten()[i].set_ylim(0, 160)

        colours = list(Plotting.energy_shares_colours.values())

        # create legend
        labels = transport_variables
        swatches = [mpatches.Patch(color=colours[j], label=labels[j]) for j in range(len(labels))]
        # #create a legend
        fig.legend(handles=swatches, labels=labels, loc='lower center', ncol=2, frameon=False)
        
        # add a legend
        # fig.legend(frameon=False)
        plt.show()

    def land_use_stacked_shares(self, illustrative_scenarios,
                                illustrative_models,
                                narrative_df):
        
        plt.rcParams['ytick.minor.visible'] = True
        land_use_variables = ['Land Cover|Pasture', 'Land Cover|Cropland', 'Land Cover|Forest', 'Land Cover']
        colours = ['#955251', '#33FF33', '#009933', 'darkgrey']
        names = ['Warning lights', 'Resource risk', 'Sustainability struggle', 'Eco balance']
        
        # set up the figure
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

        for i in range(0, len(illustrative_scenarios)):

            plotting_df = pd.DataFrame()

            # set the scenario and model
            scenario = illustrative_scenarios[i]
            model = illustrative_models[i]

            narrative_df_scenario = narrative_df.filter(scenario=scenario, model=model, year=range(2020, 2101), region='World')
            land_cover_total = narrative_df_scenario.filter(variable='Land Cover')
            land_cover_total = land_cover_total.interpolate(range(2020, 2101)).data.copy()

            for variable in land_use_variables:
                if variable == 'Land Cover':
                    # from the plotting df calculate the remaining land cover
                    plotting_df['Land Cover|Other'] = 1 - (plotting_df['Land Cover|Pasture'] + plotting_df['Land Cover|Cropland'] + plotting_df['Land Cover|Forest'])

                else:
                    df = narrative_df_scenario.filter(variable=variable)
                    interpolated_variable = df.interpolate(range(2020, 2101)).data.copy()
                    # calculate the percentage of the land cover
                    interpolated_variable['value'] = (interpolated_variable['value'] / land_cover_total['value'])
                    plotting_df[variable] = interpolated_variable['value']

            # make a stack plot for the scenario
            axs.flatten()[i].stackplot(range(2020, 2101), plotting_df.T, labels=land_use_variables, colors=colours, alpha=0.4, edgecolor='darkgrey')

            title = names[i] + ' (' + scenario + ')'

            # set the title of the plot
            axs.flatten()[i].set_title(title)

            # set the x and y axis labels
            axs.flatten()[i].set_xlabel('Year')
            axs.flatten()[i].set_ylabel('Share of land cover')

            # set x limits
            axs.flatten()[i].set_xlim(2020, 2100)

            # set y limits
            axs.flatten()[i].set_ylim(0, 1)
            plotting_df.to_csv('land_use_outputs_' + names[i] + '.csv')
        # create legend
        labels = land_use_variables
        swatches = [mpatches.Patch(color=colours[j], label=labels[j]) for j in range(len(labels))]
        # #create a legend
        fig.legend(handles=swatches, labels=labels, loc='lower center', ncol=2, frameon=False)

        plt.show()
        
            # loop through each energy variable

    def CDR_stacked_shares(self, illustrative_scenarios,
                           illustrative_models,
                           narrative_df):
        
        plt.rcParams['ytick.minor.visible'] = True
        names = ['Problem Pathway', 'Resource Risk', 'Sustainability Struggle', 'Eco-tech Endeavour']
        CDR_variables = ['Carbon Sequestration|CCS|Biomass','Carbon Sequestration|Land Use','Carbon Sequestration|Direct Air Capture']
        colours = ['#FF6600', '#FFCC00', '#FF0000']

        # set up the figure
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

        for i in range(0, len(illustrative_scenarios)):
            
            Plotting.plotting_df = pd.DataFrame()
            # set the scenario and model
            scenario = illustrative_scenarios[i]
            model = illustrative_models[i]
            narrative_df_scenario = narrative_df.filter(scenario=scenario, model=model, year=range(2020, 2101), region='World')
            print(model)
            # loop through each CDR variable

            for variable in CDR_variables:
                print(variable)
                if variable == 'Carbon Sequestration|Direct Air Capture':
                    try:
                        df = narrative_df_scenario.filter(variable=variable)
                        interpolated_variable = df.interpolate(range(2020, 2101)).data.copy()
                        Plotting.plotting_df[variable] = interpolated_variable['value']
                    except:
                        pass
                
                else:
                    try:
                        df = narrative_df_scenario.filter(variable=variable)
                        interpolated_variable = df.interpolate(range(2020, 2101)).data.copy()
                        Plotting.plotting_df[variable] = interpolated_variable['value']
                    except:
                        narrative_df_scenario = Data.land_use_seq_data.filter(scenario=scenario, model=model, year=range(2020, 2101), region='World')
                        df = narrative_df_scenario.filter(variable='Imputed|Carbon Sequestration|Land Use')
                        interpolated_variable = df.interpolate(range(2020, 2101)).data.copy()
                        Plotting.plotting_df[variable] = interpolated_variable['value']
                
            # make a stack plot for the scenario
            axs.flatten()[i].stackplot(range(2020, 2101), Plotting.plotting_df.T, labels=CDR_variables, colors=colours, alpha=0.4, edgecolor='darkgrey')

            title = names[i] + ' (' + scenario + ')'

            # set the title of the plot
            axs.flatten()[i].set_title(title)

            # set the x and y axis labels
            axs.flatten()[i].set_xlabel('Year')
            axs.flatten()[i].set_ylabel('MtCO2 Sequestered per year')

            # set x limits
            axs.flatten()[i].set_xlim(2020, 2100)

            # set y limits
            axs.flatten()[i].set_ylim(0, 21000)
        
        # create legend
        labels = CDR_variables

        swatches = [mpatches.Patch(color=colours[j], label=labels[j]) for j in range(len(labels))]
        # #create a legend
        fig.legend(handles=swatches, labels=labels, loc='lower center', ncol=2, frameon=False)

        plt.show()


if __name__ == "__main__":
    main()
