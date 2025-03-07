import numpy as np
import pyam
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from itertools import combinations
import re
import pandas as pd
from matplotlib import rcParams
import matplotlib.lines as mlines


from constants import *
from utils.file_parser import *
from scipy import stats

# from src.analysis import Selection
# from src.analysis import IndexBuilder




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
    


    scenario_archetypes = pd.read_csv(OUTPUT_DIR + 'scenario_archetypes'+ str(CATEGORIES_DEFAULT) + '.csv')




    scenarios = read_csv(PROCESSED_DIR + 'Framework_scenarios' + str(CATEGORIES_DEFAULT))


    # Plotting.polar_bar_plot_variables(Plotting, 'stats_datasheet.csv', Plotting.dimensions, 'C1a_NZGHGs')
    # Plotting.box_plot_variables(Plotting, 'variable_categories.csv', 'robust', 'C1a_NZGHGs', [2050, 2100])
    # Plotting.single_variable_box_line(Plotting, Data.c1aR10_scenarios, 'Land Cover|Forest', 'C1a_NZGHGs', 'World',years=range(2020, 2101))
    # Plotting.plot_metrics_vertical(Plotting, scenarios,  Plotting.model_families)
    # Plotting.create_radar_plots(Plotting, 
    #                             Data.model_scenarios, 
    #                             Selection.economic_scores, 
    #                             Selection.environment_scores, 
    #                             Selection.resource_scores, 
    #                             Selection.resilience_scores, 
    #                             Selection.robustness_scores, 
    #                             Plotting.bright_modern_colors)      
    
    # clostest_cluster_centroid = pd.read_csv('outputs/closest_to_centroids' + str(Data.categories) + '.csv')
    # Plotting.radar_plot_scenario_archetypes(Plotting, Data.model_scenarios, Selection.archetypes, Plotting.bright_modern_colors,
    #                                          clostest_cluster_centroid)
    # Plotting.radar_plot_scenario_archetypes(Plotting, scenarios, scenario_archetypes, Plotting.closest_to_centroids, Plotting.clustered_scores, add_all_clustered=True)
    # Plotting.line_plot_narrative_variables(Plotting, 'Energy Service|Transportation|Freight', Selection.centroid_scenarios, 
    #                                                 Data.briefing_paper_data, base_normalisation=False, 
    #                                                 secondary_variable=None, region=Data.briefing_paper_regions[0])
    
    pyam_df = read_pyam_df(PROCESSED_DIR + 'Framework_pyam' + str(CATEGORIES_DEFAULT) + '.csv')
    # Plotting.energy_system_stackplot(Plotting,Plotting.closest_to_centroids, pyam_df, 
    #                                  Plotting.energy_shares_variables, region=None)
    
    # Plotting.transport_stackplot(Plotting, Selection.centroid_scenarios, Data.briefing_paper_data, region=Data.briefing_paper_regions[1])
    # Plotting.land_use_stacked_shares(Plotting, Selection.centroid_scenarios, Data.briefing_paper_data, region=Data.briefing_paper_regions[1])
    # Plotting.CDR_stacked_shares(Plotting, Plotting.closest_to_centroids, pyam_df, region=None)
    # Plotting.radar_plot_model_fingerprint_single_panel(Plotting, scenarios, Plotting.model_families, Plotting.model_colours, Plotting.clustered_scores)
    Plotting.regional_differences_across_scenarios(Plotting, Plotting.normalised_scores, Plotting.regional_normalised_scores_cross_regional, scenarios, Plotting.closest_to_centroids, cross_regional_norm=True)
    # Plotting.specific_dimension_regional_analysis(Plotting, Plotting.normalised_scores, Plotting.regional_normalised_scores, 
    #                                               Data.R10[3], Data.R10[6], Plotting.model_families)
    # Plotting.radar_plot_ssp_pairs(Plotting, scenarios, Plotting.gini_coefficient, Plotting.clustered_scores)
    # Plotting.radar_plot(Plotting, Data.model_scenarios, Plotting.clustered_scores)
    # Plotting.convex_hull(Plotting, Plotting.clustered_scores, 10)
    # Plotting.duplicate_scenarios_plot(Plotting, Plotting.clustered_scores)
    # Plotting.count_pairwise_low_scores(Plotting, Plotting.clustered_scores, low_score_threshold=0.2)
    # meta = read_meta_data(META_FILE)
    # Plotting.radar_plot_temp_category(Plotting, meta, Plotting.clustered_scores)

    # Plotting.parallel_coord_plot(Plotting, Plotting.clustered_scores, Plotting.archetype_colours)
    # Plotting.correlation_heatmap(Plotting, Plotting.normalised_scores)

class Plotting:

    dimensions = ['economic', 'environment', 'resilience', 'resource', 'robust', 'fairness', 'transition_speed']
    dimension_names = ['Economic', 'Environment', 'Resource', 'Resilience', 'Robust', 'Fairness', 'Transition Speed']
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

    model_families = read_csv(INPUT_DIR + 'model_family.csv')

    try:
        clustered_scores = pd.read_csv(OUTPUT_DIR + 'clustered_scores' + str(CATEGORIES_DEFAULT) + '.csv')
    except FileNotFoundError:
        print('No clustered scores file found')

    selected_scenarios = pd.read_csv(OUTPUT_DIR + 'selected_scenarios' + str(CATEGORIES_DEFAULT) + '.csv')

    model_colours = {'IMAGE': '#E69F00', 'AIM':'#090059','GCAM':'#D57501', 'MESSAGE':'#56B4E9', 'REMIND':'#009E73', 'WITCH':'#CC79A7'}

    regional_normalised_scores = pd.read_csv(OUTPUT_DIR + 'regional_normalised_dimension_scores' + str(CATEGORIES_DEFAULT) + '.csv')
    regional_normalised_scores_cross_regional = pd.read_csv(OUTPUT_DIR + 'regional_normalised_dimension_scores_cross_regional_normalisation' + str(CATEGORIES_DEFAULT) + '.csv')
    normalised_scores = pd.read_csv(OUTPUT_DIR + 'normalised_scores' + str(CATEGORIES_DEFAULT) + '.csv')
    # clustered_scores = pd.read_csv(OUTPUT_DIR + 'clustered_scores' + str(CATEGORIES_DEFAULT) + '.csv')
    
    archetype_colours = ['#FFB000', '#648FFF', '#DC267F', '#FE6100']


    categories = CATEGORIES_DEFAULT
        
    investment_metrics = pd.read_csv(OUTPUT_DIR + 'energy_supply_investment_score' + str(categories) + '.csv')
    environment_metrics = pd.read_csv(OUTPUT_DIR + 'environmental_metrics' + str(categories) + '.csv')
    resource_metrics = pd.read_csv(OUTPUT_DIR + 'material_use_ratios' + str(categories) + '.csv')
    transition_speed_metrics = pd.read_csv(OUTPUT_DIR + 'transition_speed_metrics' + str(categories) + '.csv')

    # import resilience metrics
    final_energy_demand = pd.read_csv(OUTPUT_DIR + 'final_energy_demand' + str(categories) + '.csv')    
    # energy_diversity = pd.read_csv(OUTPUT_DIR + 'shannon_diversity_index' + str(categories) + '.csv')   
    gini_coefficient = pd.read_csv(OUTPUT_DIR + 'gini_coefficient' + str(categories) + '.csv')
    # electricity_price = pd.read_csv(OUTPUT_DIR + 'electricity_prices' + str(categories) + '.csv')

    # import robustness metrics
    # energy_system_flexibility = pd.read_csv(OUTPUT_DIR + 'flexibility_scores' + str(categories) + '.csv')
    carbon_budgets = pd.read_csv(OUTPUT_DIR + 'carbon_budget_shares' + str(categories) + '.csv')
    # low_carbon_diversity = pd.read_csv(OUTPUT_DIR + 'low_carbon_shannon_diversity_index' + str(categories) + '.csv')
    # CDR_2050 = pd.read_csv(OUTPUT_DIR + 'total_CDR' + str(categories) + '.csv')

    # import fairness metrics
    between_region_gini = pd.read_csv(OUTPUT_DIR + 'between_region_gini' + str(categories) + '.csv')
    # carbon_budget_fairness = pd.read_csv(OUTPUT_DIR + 'carbon_budget_fairness' + str(categories) + '.csv')

    closest_to_centroids = pd.read_csv(OUTPUT_DIR + 'closest_to_centroids' + str(categories) + '.csv')

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
            ax.scatter(y=investment_info['x'], x=investment_info[investment_column], marker='o',c=colours[:len(investment_info)], s=50, alpha=0.5)
            # ax.scatter(y=investment_info['x'], x=investment_info[investment_column] + np.random.normal(0, 0.02, len(investment_info)), marker='o', s=50, alpha=0.5)
            ax.set_title(y_labels[count])
            # set y min and max
            ax.set_xlim(0.4, 1.3)
            # remove x tick and label
            ax.set_yticks([])
            ax.set_ylabel('')
            count += 1
        
        # make a ledgend with a swatch of each color and the scenario name and model
        # # make a list of the scenario names
        # label_names = []
        # for label in range(0, len(investment_info)):
        #     label_names.append(investment_info['scenario'][label]  + '_(' + investment_info['model'][label] + ')')
        
        # # make a list of the colors
        # colors = colours[:len(investment_info)]
        # # make a list of the markers
        # markers = ['o'] * len(investment_info)
        # # make a list of the sizes
        # sizes = [100] * len(investment_info)
        # # make a list of the labels
        # labels = label_names
        # swatches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
        # #create a legend
        # fig.legend(handles=swatches, labels=labels, loc='lower center', ncol=3, fontsize=7, frameon=False)
        # plt.savefig('figures/investment_metrics_horizontal.pdf')
        plt.show()
        
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

    # plot the metrics as raw values for each dimension
    def plot_metrics_vertical(self, scenario_list, model_families):

        
        colour_map = ['#E69F00','#56B4E9', '#009E73', '#CC79A7']
        
        
        # add a column to the scenario_list with the model family
        scenario_list['model_family'] = scenario_list['model'].map(model_families.set_index('model')['model_family'])

        # get the unique model families
        model_families = scenario_list['model_family'].unique().tolist()
        models = scenario_list['model'].unique().tolist()
    
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



        investment_info = Plotting.investment_metrics
        investment_info = investment_info.reset_index()
        environmental_info = Plotting.environment_metrics
        environmental_info = environmental_info.reset_index()
        resource_info = Plotting.resource_metrics
        final_energy_demand = Plotting.final_energy_demand
        robustness_info = Plotting.carbon_budgets
        fairness_info = Plotting.between_region_gini
        transition_speed_info = Plotting.transition_speed_metrics

        # put columns together into one dataframe
        investment_info['forest_change_2100'] = environmental_info['forest_change_2100']
        investment_info['Cd'] = resource_info['Cd']
        investment_info['final_energy_demand'] = final_energy_demand['final_energy_demand']
        investment_info['carbon_budget_share'] = robustness_info['carbon_budget_share']
        investment_info['between_region_gini'] = fairness_info['between_region_gini']
        investment_info['Share of final energy from electricity'] = transition_speed_info['Share of final energy from electricity']

        print(investment_info)

        column_list = ['mean_value', 'forest_change_2100', 'Cd', 'final_energy_demand', 'carbon_budget_share', 'between_region_gini', 'Share of final energy from electricity']
        # header_list = ['mean_value', 'mean_value_2050']
        # y_labels = ['Mean Ratio of Energy Supply\n Investment (2020-2100)', 'Mean Ratio of Energy Supply\n Investment (2020-2050)']
        y_labels = ['Mean Ratio of Energy Supply\n Investment (2020-2100)', 
                    'Change in Forest Land\n Cover (2020-2100)', 
                    'Neodymium', 
                    'Final Energy Demand (EJ)',
                    'Carbon Budget Share\n used by 2030', 
                    'Between Region Gini Coefficient',
                    'Max decade increase in the share of final energy from electricity']
        
        # # make a list of numbers accending for each scenario
        zero_list = [1] * len(investment_info)
        investment_info['x'] = zero_list
        fig_length = 3 * len(column_list)
        fig, axs = plt.subplots(1,len(column_list), figsize=(5, fig_length))
        count_column = 0
        for ax, investment_column in zip(axs, column_list):
            # create a box plot for the investment data with each scenario a different color
            ax.boxplot(investment_info[investment_column], showfliers=False)
            investment_info['model_family'] = scenario_list['model_family']
            # for model_family in model_families:

            #     colour = colours[model_families.index(model_family)]
            #     model_family_info = investment_info[investment_info['model_family'] == model_family]
            #     ax.scatter(x=model_family_info['x'], y=model_family_info[investment_column], c=colour, marker='o', s=50, alpha=0.65)
            count = 0
            for model in model_families:
                
                model_info = investment_info[investment_info['model_family'] == model]
                ax.scatter(x=model_info['x'], y=model_info[investment_column], c=colour_map[count], marker='o', s=40, alpha=0.35)
                count += 1
            # ax.scatter(x=investment_info['x'], y=investment_info[investment_column], marker='o',c=colours[:len(investment_info)], s=50, alpha=0.5)
            # ax.scatter(y=investment_info['x'], x=investment_info[investment_column] + np.random.normal(0, 0.02, len(investment_info)), marker='o', s=50, alpha=0.5)
            ax.set_title(column_list[count_column])
            # set y min and max
            # ax.set_ylim(0.4, 1.3)
            # remove x tick and label
            ax.set_xticks([])
            ax.set_xlabel('')
            count_column += 1
        
        # make a ledgend with a swatch of each color and the scenario name and model
        # # make a list of the scenario names
        # label_names = []
        # for label in range(0, len(models)):
        #     label_names.append(investment_info['model'][label])
        
        # make a list of the colors
        colors = colour_map
        # make a list of the markers
        markers = ['o'] * len(model_families)
        # make a list of the sizes
        sizes = [50] * len(model_families)
        # # make a list of the labels
        labels = model_families
        swatches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
        # #create a legend
        fig.legend(handles=swatches, labels=labels, loc='lower center', ncol=3, fontsize=7, frameon=False)
        # plt.savefig('figures/investment_metrics_horizontal.pdf')
        plt.show()
        
        # plot the environmental data in a box plot
        column_list = ['forest_change_2050', 'forest_change_2100']
        y_labels = ['Change in Forest Land Cover (2020-2050)', 'Change in Forest Land Cover (2020-2100)']
        environmental_info['x'] = zero_list
        fig, axs = plt.subplots(1,len(column_list), figsize=(5, fig_length))
        count = 0
        for ax, environmental_column in zip(axs, column_list):
            # create a box plot for the investment data with each scenario a different color
            ax.boxplot(environmental_info[environmental_column], showfliers=False)

            for model in models:

                colour = colours[models.index(model)]
                model_info = environmental_info[environmental_info['model'] == model]
                ax.scatter(x=model_info['x'], y=model_info[environmental_column], c=colour, marker='o', s=40, alpha=0.35)
            
            # ax.scatter(y=environmental_info['x'], x=environmental_info[environmental_column], c=colours[:len(environmental_info)], marker='o', s=100)
            ax.set_title(y_labels[count])
            # set y min and max
            ax.set_ylim(0, 0.1)
            # remove x tick and label
            ax.set_xticks([])
            ax.set_xlabel('')
            count += 1
        
        # fig.legend(handles=swatches, labels=labels, loc='lower center', ncol=3, fontsize=7, frameon=False)
        # plt.show()
        # plt.savefig('figures/environmental_metrics.pdf')
        

        # plot the resource data in a box plot
        resource_info = IndexBuilder.resource_metrics
        print(resource_info)
        column_list = ['Nd', 'Dy', 'Ni', 'Mn', 'Ag', 'Cd', 'Te', 'Se', 'In']
        y_labels = ['Neodymium', 'Dysprosium', 'Nickle', 'Manganese', 'Silver', 'Cadmium', 'Tellurium', 'Selenium', 'Indium']
        resource_info['x'] = zero_list
        fig, axs = plt.subplots(1, len(column_list), figsize=(13, 10))
        count = 0
        for ax, resource_column in zip(axs, column_list):
            # create a box plot for the investment data with each scenario a different color
            ax.boxplot(resource_info[resource_column], showfliers=False)
            
            for model in models:
                colour = colours[models.index(model)]
                model_info = resource_info[resource_info['model'] == model]
                ax.scatter(x=model_info['x'], y=model_info[resource_column], c=colour, marker='o', s=40, alpha=0.35)
            
            ax.set_title(y_labels[count])
            # set y min and max
            # ax.set_ylim(0.1, 22)
            # remove x tick and label
            ax.set_xticks([])
            ax.set_xlabel('')
            count += 1
        
        # add title
        fig.suptitle('Mean ratio of renewables mineral usage share (to 2050) to baseline share', fontsize=14)
        # fig.legend(handles=swatches, labels=labels, loc='lower center', ncol=3, fontsize=7, frameon=False)
        plt.show()
        
        
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
        # fig, axs = plt.subplots(3, 6, figsize=(20, 10), subplot_kw=dict(polar=True))
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
        # plt.savefig('figures/radar_plots' + str(Data.categories) + '.pdf')
        plt.show()
        

    def radar_plot_scenario_archetypes(self, scenario_model_list, archetypes, selected_scenarios, clustered_scores, add_all_clustered=False):

        # plt.rcParams['font.size'] = 7
        # Number of variables we're plotting.
        categories = list(archetypes)[0:7]
        # archetype_names = ['Warning lights', 'Resource risk', 'Sustainability struggle', 'Clear path']
        archetype_names = ['A','B', 'C', 'D']
        archetype_colours = ['#648FFF', '#DC267F', '#FE6100', '#FFB000']
        # archetype_colours = ['#FFB000', '#648FFF', '#DC267F', '#FE6100']
        # make a fig with 4 subplots, 2 rows
        fig, axs = plt.subplots(2, 2, figsize=(10, 10), subplot_kw=dict(polar=True))

        N = len(categories)

        print(selected_scenarios)
        print(archetypes)
        # organise the seected scenarios by cluster 0-3
        selected_scenarios = selected_scenarios.sort_values(by='cluster')
        print(selected_scenarios)
        for archetype in range(0, 4):

            ax = axs.flatten()[archetype]
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            stats = archetypes.loc[archetypes['cluster'] == archetype, categories].values.flatten().tolist()
            stats += stats[:1]
            angles += angles[:1]
            
            stats_illustrative = selected_scenarios[selected_scenarios['cluster'] == archetype].values.flatten().tolist()
            stats_illustrative = stats_illustrative[2:9]
            # stats_illustrative = stats_illustrative[0:7]
            stats_illustrative += stats_illustrative[:1]

            # Draw the outline of our data.
            ax.fill(angles, stats, color=archetype_colours[archetype], alpha=0.2, zorder=10)
            ax.plot(angles, stats, color=archetype_colours[archetype], linewidth=3, zorder=11)
            ax.fill(angles, stats_illustrative, color='black', alpha=0.1)
            ax.plot(angles, stats_illustrative, color='black', linewidth=1)            # Add the other scenarios in the cluster as faint outlines
            
            if add_all_clustered:

                cluster_scenarios = clustered_scores[clustered_scores['cluster'] == archetype]['scenario']
                print(cluster_scenarios)
                cluster_models = clustered_scores[clustered_scores['cluster'] == archetype]['model']

                for scenario, model in zip(cluster_scenarios, cluster_models):

                    print(scenario)
                    scenario_stats = clustered_scores.loc[(clustered_scores['scenario'] == scenario) & (clustered_scores['model'] == model), categories].values.flatten().tolist()
                    print(scenario_stats)
                    scenario_stats += scenario_stats[:1]
                    ax.plot(angles, scenario_stats, color=archetype_colours[archetype], alpha=0.2, zorder=9, linewidth=1)

            # # Labels for each point
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(Plotting.dimension_names)

            #set the y limit
            ax.set_ylim(0, 1)
            title = archetype_names[archetype] + '_' + selected_scenarios[selected_scenarios['cluster'] == archetype]['scenario'].values[0]
            # Title for each subplot with the archetype name
            ax.set_title(title,size=14, y=1.1, fontweight='bold')

        # Show the figure
        plt.tight_layout()
        plt.show()


    def radar_plot_model_fingerprint(self, scenario_model_list, model_families, model_colours, scenario_scores):

        """
        Model Fingerprint Radar Plot
        
        This function takes all the scenarios and their respective dimension scores, and 
        sorts them into model families. There is then one radar per model family with the 
        median value for each dimension plotted in bold, with every other scenario in the 
        family plotted in a lighter color to demonstrate variation within the model 
        family. 
        
        Inputs: 
        - scenario_model_list: a dataframe containing the scenario names and model names
        - model_families: a dataframe containing the model names and the model family they belong to
        - colours: a list of colors to use for the radar plots
        
        Outputs: 
        - Radar plots for each model family, showing the median value for each dimension in bold

        """


        # add a column to the scenario scores with the model family
        scenario_scores['model_family'] = scenario_scores['model'].map(model_families.set_index('model')['model_family'])
        scenario_scores.to_csv('outputs/model_families' +  str(CATEGORIES_DEFAULT) + '.csv')


        # get the unique model families
        model_families = scenario_scores['model_family'].unique().tolist()

        print(model_families, len(model_families))

        # Number of variables we're plotting.
        categories = list(scenario_scores)[2:8]

        # make a fig with 6 subplots, 2 rows
        fig, axs = plt.subplots(2, 2, figsize=(10, 10), subplot_kw=dict(polar=True))

        N = len(categories)

        for i, model_family in enumerate(model_families):

            ax = axs.flatten()[i]
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            stats = scenario_scores.loc[scenario_scores['model_family'] == model_family, categories].median().values.flatten().tolist()
            stats += stats[:1]
            angles += angles[:1]

            print(model_family)
            
            # Get model colour from dictionary
            model_colour = model_colours[model_family]

            # make colour a shade brighter
            model_colour = sns.light_palette(model_colour, n_colors=10)[8]

            # Draw the outline of our data.
            # ax.fill(angles, stats, color=colours[i], alpha=0.25)
            ax.plot(angles, stats, color=model_colour, linewidth=2, zorder=10, alpha=0.9)

            # # Labels for each point
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)

            #set the y limit
            ax.set_ylim(0, 1)

            # Title for each subplot with the model family name
            ax.set_title(model_family,size=10, y=1.1, fontweight='bold')

            model_family_data = scenario_scores.loc[scenario_scores['model_family'] == model_family]


            # add the outlines for the other scenarios in the model family
            for scenario in model_family_data.loc[model_family_data['model_family'] == model_family, 'scenario']:

                print(scenario, model_family)
                scenario_stats = model_family_data.loc[model_family_data['scenario'] == scenario, categories].values.flatten().tolist()
                scenario_angle = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
                print(scenario_stats)
                scenario_stats += scenario_stats[:1]  # Fix: Use scenario_stats instead of stats
                scenario_angle += scenario_angle[:1]
                ax.plot(scenario_angle, scenario_stats, color='black', alpha=0.1, zorder=9)

            # # Labels for each point
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(Plotting.dimension_names)

            #set the y limit
            ax.set_ylim(0, 1)

            # Title for each subplot with the model family name
            ax.set_title(model_family,size=12, y=1.1, fontweight='bold')

        # Show the figure
        plt.tight_layout()
        plt.show()


    def radar_plot_model_fingerprint_single_panel(self, scenario_model_list, model_families, model_colours, scenario_scores):

        """
        Model Fingerprint Radar Plot
        
        This function takes all the scenarios and their respective dimension scores, and 
        sorts them into model families. There is then one radar per model family with the 
        median value for each dimension plotted in bold, with every other scenario in the 
        family plotted in a lighter color to demonstrate variation within the model 
        family. 
        
        Inputs: 
        - scenario_model_list: a dataframe containing the scenario names and model names
        - model_families: a dataframe containing the model names and the model family they belong to
        - colours: a list of colors to use for the radar plots
        
        Outputs: 
        - Radar plots for each model family, showing the median value for each dimension in bold

        """
        
        colour_map = ['#E69F00','#56B4E9', '#009E73', '#CC79A7']

        # add a column to the scenario scores with the model family
        scenario_scores['model_family'] = scenario_scores['model'].map(model_families.set_index('model')['model_family'])
        scenario_scores.to_csv(OUTPUT_DIR + 'model_families' +  str(CATEGORIES_DEFAULT) + '.csv')

        # filter for EN_NPi2020_400f
        scenario_scores = scenario_scores[scenario_scores['scenario'].str.contains('EN_NPi2020_400f')]


        # get the unique model families
        model_families = scenario_scores['model_family'].unique().tolist()

        print(model_families, len(model_families))

        # Number of variables we're plotting.
        categories = list(scenario_scores)[2:9]

        # make a fig with 6 subplots, 2 rows
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(polar=True))

        N = len(categories)

        # ax = axs.flatten()
        count = 0
        for model_family in model_families:
        
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            stats = scenario_scores.loc[scenario_scores['model_family'] == model_family, categories].median().values.flatten().tolist()
            stats += stats[:1]
            angles += angles[:1]

            print(model_family)
            
            # model_colour = Plotting.bright_modern_colors[count]
            model_colour = colour_map[count]
            # Draw the outline of our data.
            # ax.fill(angles, stats, color=colours[i], alpha=0.25)
            ax.plot(angles, stats, color=model_colour, linewidth=2, zorder=10, alpha=1)

            # # Labels for each point
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)

            #set the y limit
            ax.set_ylim(0, 1)

            # Title for each subplot with the model family name
            ax.set_title(model_family,size=10, y=1.1, fontweight='bold')

            model_family_data = scenario_scores.loc[scenario_scores['model_family'] == model_family]


            # add the outlines for the other scenarios in the model family
            for scenario in model_family_data.loc[model_family_data['model_family'] == model_family, 'scenario']:

                print(scenario, model_family)
                scenario_stats = model_family_data.loc[model_family_data['scenario'] == scenario, categories].values.flatten().tolist()
                scenario_angle = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
                print(scenario_stats)
                scenario_stats += scenario_stats[:1]  # Fix: Use scenario_stats instead of stats
                scenario_angle += scenario_angle[:1]
                ax.plot(scenario_angle, scenario_stats, color=model_colour, alpha=0.2, zorder=9)

            count += 1
        # # Labels for each point
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(Plotting.dimension_names)

        #set the y limit
        ax.set_ylim(0, 1)

        # Title for each subplot with the model family name
        ax.set_title(model_family,size=12, y=1.1, fontweight='bold')

        # Show the figure
        plt.tight_layout()
        plt.show()


    def radar_plot_ssp(self, scenario_model_list, ssps, scenario_scores):

        """
        SSP fingerprint radar plot
        This plot is as above, but instead of model families, the scenarios are grouped by SSP

        Inputs:
        - scenario_model_list: a dataframe containing the scenario names and model names
        - ssps: a dataframe containing the scenario names and the SSP they belong to
        - scenario_scores: a dataframe containing the scenario names and the dimension scores

        Outputs:
        - Radar plots for each SSP, showing the median value for each dimension in bold
        """

        ssp_colours = {1:'#4ADEB5',2:'#8596FC' }
        scenario_pairs = {'IMAGE':{'SSP1_SPA1_19I_D_LB':'SSP2_SPA1_19I_D_LB',
                                   'SSP1_SPA1_19I_LIRE_LB':'SSP2_SPA1_19I_LIRE_LB',
                                   'SSP1_SPA1_19I_RE_LB':'SSP2_SPA1_19I_RE_LB'},
                            'REMIND':{'SusDev_SSP1-PkBudg900':'SusDev_SSP2-PkBudg900',
                                      'CEMICS_SSP1-1p5C-fullCDR':'CEMICS_SSP2-1p5C-fullCDR',
                                      'CEMICS_SSP1-1p5C-minCDR':'CEMICS_SSP2-1p5C-minCDR'}}



        scenario_scores['ssp'] = ssps['ssp']

        # get the unique SSPs
        ssps = scenario_scores['ssp'].unique().tolist()

        # Number of variables we're plotting.
        categories = list(scenario_scores)[2:9]
        print(categories)

        # make a fig with 2 subplots, 1 row
        fig, axs = plt.subplots(2, 1, figsize=(10, 10), subplot_kw=dict(polar=True))

        N = len(categories)

        for i, ssp in enumerate(ssps):

            ax = axs.flatten()[i]
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            stats = scenario_scores.loc[scenario_scores['ssp'] == ssp, categories].median().values.flatten().tolist()
            stats += stats[:1]
            angles += angles[:1]

            # Draw the outline of our data.
            ax.fill(angles, stats, color=ssp_colours[ssp], alpha=0.25)
            ax.plot(angles, stats, color=ssp_colours[ssp], linewidth=2)

            # # Labels for each point
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)

            #set the y limit
            ax.set_ylim(0, 1)

            # Title for each subplot with the SSP name
            ax.set_title(ssp,size=10, y=1.1, fontweight='bold')

            ssp_data = scenario_scores.loc[scenario_scores['ssp'] == ssp]
            print(ssp_data)

            for scenario, model in zip(ssp_data.loc[scenario_scores['ssp'] == ssp, 'scenario'], ssp_data.loc[scenario_scores['ssp'] == ssp, 'model']):
                
                scenario_stats = scenario_scores.loc[scenario_scores['scenario'] == scenario]
                model_scenario_stats = scenario_stats.loc[scenario_stats['model'] == model, categories].values.flatten().tolist() 
                print(model_scenario_stats)
                scenario_angle = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
                print(scenario_angle)
                model_scenario_stats += model_scenario_stats[:1]
                scenario_angle += scenario_angle[:1]
                ax.plot(scenario_angle, model_scenario_stats, color='black', alpha=0.1)

        # Show the figure
        plt.tight_layout()
        plt.show()


    def radar_plot_ssp_pairs(self, scenario_model_list, ssps, scenario_scores):

        """
        SSP fingerprint radar plot (Pairs)
        This plot is as above, but instead of model families, the scenarios are grouped by SSP

        Inputs:
        - scenario_model_list: a dataframe containing the scenario names and model names
        - ssps: a dataframe containing the scenario names and the SSP they belong to
        - scenario_scores: a dataframe containing the scenario names and the dimension scores

        Outputs:
        - Radar plots for each SSP, showing the median value for each dimension in bold
        """

        ssp_colours = {1:'#4ADEB5',2:'#8596FC' }
        # scenario_pairs = {'IMAGE':{'SSP1_SPA1_19I_D_LB':'SSP2_SPA1_19I_D_LB',
        #                            'SSP1_SPA1_19I_LIRE_LB':'SSP2_SPA1_19I_LIRE_LB',
        #                            'SSP1_SPA1_19I_RE_LB':'SSP2_SPA1_19I_RE_LB'},
        #                     'REMIND':{'SusDev_SSP1-PkBudg900':'SusDev_SSP2-PkBudg900',
        #                               'CEMICS_SSP1-1p5C-fullCDR':'CEMICS_SSP2-1p5C-fullCDR',
        #                               'CEMICS_SSP1-1p5C-minCDR':'CEMICS_SSP2-1p5C-minCDR'}}
        scenario_pairs = {'IMAGE':{'SSP1_SPA1_19I_D_LB':'SSP2_SPA1_19I_D_LB'},
                    'REMIND':{'CEMICS_SSP1-1p5C-minCDR':'CEMICS_SSP2-1p5C-minCDR'}}
        
        colours = {'IMAGE':["#E69F00", "#FF6347", "#FF0000"], 'REMIND':["#009E73"]}
        model_families = ['IMAGE', 'REMIND']

        # list of green colours
        greens = ['#00FF00', '#00EE00', '#00DD00', '#00CC00', '#00BB00', '#00AA00', '#009900', '#008800', '#007700', '#006600', '#005500', '#004400', '#003300', '#002200', '#001100', '#000000']
        oranges = ['#FFA500', '#FF9F00', '#FF9E00', '#FF9D00', '#FF9C00', '#FF9B00', '#FF9A00', '#FF9900', '#FF9800', '#FF9700', '#FF9600', '#FF9500', '#FF9400', '#FF9300', '#FF9200', '#FF9100']
        # scenario_scores['ssp'] = ssps['ssp']

        # # get the unique SSPs
        # ssps = scenario_scores['ssp'].unique().tolist()

        # Number of variables we're plotting.
        categories = list(scenario_scores)[2:9]

        # make a fig with 2 subplots, 1 row
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(polar=True))

        N = len(categories)
        

        for i in model_families:

            scenario_pairs_model = scenario_pairs[i]
            
            count = 0
            for ssp1, ssp2 in scenario_pairs_model.items():
                
                colour = colours[i][count]      

                ssp1_stats = scenario_scores.loc[scenario_scores['scenario'] == ssp1, categories].values.flatten().tolist()
                ssp2_stats = scenario_scores.loc[scenario_scores['scenario'] == ssp2, categories].values.flatten().tolist()

                angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
                # stats = scenario_scores.loc[scenario_scores['ssp'] == ssp, categories].median().values.flatten().tolist()
                ssp1_stats += ssp1_stats[:1]
                ssp2_stats += ssp2_stats[:1]
                angles += angles[:1]

                # Draw the outline of our data.
                # ax.fill(angles, stats, color=ssp_colours[ssp], alpha=0.25)
                ax.plot(angles, ssp1_stats, color=colour, linewidth=2, alpha=0.5)
                ax.plot(angles, ssp2_stats, color=colour, linewidth=2, alpha=0.5, linestyle='dashed')

                count += 1
        # # Labels for each point
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)

        #set the y limit
        ax.set_ylim(0, 1)

        # # Title for each subplot with the SSP name
        # ax.set_title(ssp,size=10, y=1.1, fontweight='bold')

        # ssp_data = scenario_scores.loc[scenario_scores['ssp'] == ssp]
        # print(ssp_data)

        # Show the figure
        plt.tight_layout()
        plt.show()


    def radar_plot_temp_category(self, meta_data, scenario_scores):
        
        """
        Temperature category radar plot
        Here all the scenarios are grouped by temperature category, with the median value for each dimension
        plotted in bold, and every other scenario in the category plotted in a lighter color to demonstrate

        Inputs:
        - scenario_model_list: a dataframe containing the scenario names and model names, temp category
        - scenario_scores: a dataframe containing the scenario names and the dimension scores

        Outputs:
        - a single radar plot with scenarios plotted based on their temperature category

        """
        category_colours = {'C1':'#117733', 'C2':'#AA4499'}

        # plt.rcParams['font.size'] = 7
        # Number of variables we're plotting.
        categories = list(scenario_scores)[2:9]

        # make a fig with 3 subplots, 1 row
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(polar=True))

        N = len(categories)

        # get the unique temperature categories
        temp_categories = ['C1', 'C2']

        # filter the metadata for world and emissions co2
        # meta_data = meta_data[(meta_data['region'] == 'Countries of Sub-Saharan Africa') & (meta_data['variable'] == 'Emissions|CO2') & (meta_data['year'] == 2030)]
        # print(meta_data)
        # set the index as the scenario and model
        # meta_data.set_index(['scenario', 'model'], inplace=True)
        scenario_scores.set_index(['scenario', 'model'], inplace=True)
        
        # merge the scenario scores with the temperature category
        scenario_scores = scenario_scores.merge(meta_data['Category'], on=['scenario', 'model'], how='left')
        print(scenario_scores)
        scenario_scores.reset_index(inplace=True)

        for i, temp_category in enumerate(temp_categories):

            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            stats = scenario_scores.loc[scenario_scores['Category'] == temp_category, categories].median().values.flatten().tolist()
            stats += stats[:1]
            angles += angles[:1]

            # Draw the outline of our data.
            # ax.fill(angles, stats, color='black', alpha=0.25)
            ax.plot(angles, stats, color=category_colours[temp_category], linewidth=3)

            # # Labels for each point
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)

            #set the y limit
            ax.set_ylim(0, 1)

            temp_category_data = scenario_scores.loc[scenario_scores['Category'] == temp_category]
            for scenario, model in zip(temp_category_data['scenario'], temp_category_data['model']):
                
                scenario_stats = temp_category_data.loc[temp_category_data['scenario'] == scenario]
                model_scenario_stats = scenario_stats.loc[scenario_stats['model'] == model, categories].values.flatten().tolist() 
                scenario_angle = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
                model_scenario_stats += model_scenario_stats[:1]
                scenario_angle += scenario_angle[:1]
                ax.plot(scenario_angle, model_scenario_stats, color=category_colours[temp_category], alpha=0.3)
        # Show the figure
        plt.tight_layout()
        plt.show()


    def radar_plot(self, scenario_model_list, scenario_scores):

        """
        Single radar with all the scenarios plotted on it
    
        Inputs:
        - scenario_model_list: a dataframe containing the scenario names and model names
        - scenario_scores: a dataframe containing the scenario names and the dimension scores

        Outputs:
        - Radar plot with all scenarios plotted on it

        """

        # Number of variables we're plotting.
        categories = list(scenario_scores)[2:8]

        # make a fig with 1 subplot
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(polar=True))

        N = len(categories)

        for scenario, model in zip(scenario_model_list['scenario'], scenario_model_list['model']):

            scenario_stats = scenario_scores.loc[scenario_scores['scenario'] == scenario]
            model_scenario_stats = scenario_stats.loc[scenario_stats['model'] == model, categories].values.flatten().tolist()

            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            model_scenario_stats += model_scenario_stats[:1]
            angles += angles[:1]

            # # Draw the outline of our data.
            # ax.fill(angles, model_scenario_stats, color='black', alpha=0.25)
            ax.plot(angles, model_scenario_stats, color='black', alpha=0.1)

        # # Labels for each point
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)

        #set the y limit
        ax.set_ylim(0, 1)

        # # Title for each subplot with the scenario name
        # ax.set_title(scenario,size=10, y=1.1, fontweight='bold')

        # Show the figure
        plt.tight_layout()
        plt.show()

          
    def line_plot_narrative_variables(self, variable, selected_scenarios, df, 
                                      base_normalisation=False, secondary_variable=None, region=None,):

        illustrative_models = selected_scenarios['model'].tolist()
        illustrative_scenarios = selected_scenarios['scenario'].tolist()
        
        if region == None:
            region = 'World'

        plt.rcParams['xtick.minor.visible'] = True
        plt.rcParams['ytick.minor.visible'] = True
        plt.rcParams['ytick.labelright'] = True
        # scenario_colours = ['#FBA006', '#19E5FF', '#FF03FB', '#00F982']
        scenario_colours = ['#FFB000', '#648FFF', '#DC267F', '#FE6100']
        names = ['A', 'B', 'C', 'D']
        # names = ['Warning lights', 'Resource risk', 'Sustainability struggle', 'Eco balance']
        
        # set up the figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # filter for the variable
        df_variable = df.filter(variable=variable)

        # get the units
        units = df_variable['unit'].unique().tolist()
        units = units[0]

        # filter for the years
        df_variable = df_variable.filter(year=range(2020, 2100+1))

        for i in range(0, len(illustrative_scenarios)):
            
            scenario = illustrative_scenarios[i]
            label = names[i] + ' (' + scenario + ')' 

            # filter for the scenario
            df_scenario = df_variable.filter(model=illustrative_models[i], scenario=scenario, region=region)
            
            if base_normalisation == True:
                # df_scenario['value'] = df_scenario['value'] / df_scenario['value'].iloc[0]
                ax.plot(df_scenario['year'], (df_scenario['value']/df_scenario['value'][0]), label=label, color=scenario_colours[i], linewidth=2)
            else:
                ax.plot(df_scenario['year'], df_scenario['value'], label=label, color=scenario_colours[i], linewidth=2)
        
            if secondary_variable != None:
                df_secondary_variable = df.filter(variable=secondary_variable)
                df_secondary_variable = df_secondary_variable.filter(model=illustrative_models[i], scenario=scenario)
                if base_normalisation == True:
                    ax.plot(df_secondary_variable['year'], (df_secondary_variable['value']/df_secondary_variable['value'][0]), label=secondary_variable, color=scenario_colours[i], linestyle='dashed', linewidth=2)
                else:
                    ax.plot(df_secondary_variable['year'], df_secondary_variable['value'], label=secondary_variable, color=scenario_colours[i], linestyle='dashed', linewidth=2)

        # set the title of the plot
        ax.set_title(variable + ' (' + region + ')')
        # set the x and y axis labels
        ax.set_xlabel('Year')
        
        if base_normalisation == True:
            ax.set_ylabel('Change over time (normalised to 2020 values)')
        else:
            ax.set_ylabel(units)

        # set x limits
        ax.set_xlim(2020, 2100)

        # set y limits
        # ax.set_ylim(ylim_min, ylim_max)


        # add a legend
        ax.legend(frameon=False)

        plt.show()
        
    
    def energy_system_stackplot(self, selected_scenarios, data_df, 
                                energy_variables, region=None):
        
        if region == None:
            region = 'World'
        
        illustrative_models = selected_scenarios['model'].tolist()
        illustrative_scenarios = selected_scenarios['scenario'].tolist()
        narrative_df = data_df.filter(variable=energy_variables, region=region, year=range(2020, 2101))
        
        plt.rcParams['ytick.minor.visible'] = True
        # names = ['Warning lights', 'Resource risk', 'Sustainability struggle', 'Eco balance']
        names = ['A', 'B', 'C', 'D']
        
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
            # indicator_df_scenario = indicator_df.filter(scenario=scenario, model=model, year=range(2020, 2101), region='World')
            # filter the narrative dataframe for the scenario and model
            narrative_df_scenario = narrative_df.filter(scenario=scenario, model=model)
            colours = Plotting.energy_shares_colours.values()
            
            # loop through each energy variable
            for variable in energy_variables:
                print(variable)
                df = narrative_df_scenario.filter(variable=variable)                
                
                # if variable == 'Primary Energy|Fossil|w/ CCS' or variable == 'Primary Energy|Fossil|w/o CCS':
                #     df = narrative_df_scenario.filter(variable=variable)
                #     print(df)
                # else:
                #     df = indicator_df_scenario.filter(variable=variable)
                #     print(df)
                # interpolate the data for the variable
                interpolated_variable = df.interpolate(range(2020, 2101))
                # add the interpolated data to the plotting dataframe
                plotting_df[variable] = interpolated_variable['value']

            # make a stack plot for the scenario
            axs.flatten()[i].stackplot(range(2020, 2101), plotting_df.T, labels=energy_variables, colors=colours, alpha=0.45, edgecolor=colours, linewidth=0.25)
            
            # # add line plot over the top
            # for variable in energy_variables:
            #     axs.flatten()[i].plot(range(2020, 2101), plotting_df[variable], label=variable, color=Plotting.energy_shares_colours[variable], linewidth=2)
            title = names[i] + ' (' + scenario + ')'
            # title = scenario
            # set the title of the plot
            axs.flatten()[i].set_title(title)
            # set the x and y axis labels
            axs.flatten()[i].set_xlabel('Year')
            axs.flatten()[i].set_ylabel('EJ')

            # set x limits
            axs.flatten()[i].set_xlim(2020, 2100)

            # set y limits
            axs.flatten()[i].set_ylim(0, 1200)

        colours = list(Plotting.energy_shares_colours.values())

        # create legend
        labels = energy_variables
        swatches = [mpatches.Patch(color=colours[j], label=labels[j]) for j in range(len(labels))]
        # #create a legend
        fig.legend(handles=swatches, labels=labels, loc='lower center', ncol=2, frameon=False)
        
        # add overall title
        fig.suptitle('Primary Energy Mix ('+ region + ')', fontsize=16)

        # add a legend
        # fig.legend(frameon=False)
        plt.show()


    def transport_stackplot(self, selected_scenarios,
                                narrative_df,
                                region=None):
        if region == None:
            region = 'World'
        
        illustrative_models = selected_scenarios['model'].tolist()
        illustrative_scenarios = selected_scenarios['scenario'].tolist()
        plt.rcParams['ytick.minor.visible'] = True
        # names = ['Warning lights', 'Resource risk', 'Sustainability struggle', 'Eco balance']
        names = ['A', 'B', 'C', 'D']
        transport_variables = ['Final Energy|Transportation', 'Final Energy|Transportation|Liquids|Oil']
        colours = ['#3399FF', 'darkgrey']
        
        # set up the figure
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

        for i in range(0, len(illustrative_scenarios)):

            plotting_df = pd.DataFrame()
            # set the scenario and model
            scenario = illustrative_scenarios[i]
            model = illustrative_models[i]
            narrative_df_scenario = narrative_df.filter(scenario=scenario, model=model, year=range(2020, 2101), region=region)

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
            axs.flatten()[i].set_ylim(0, 18)

        colours = list(Plotting.energy_shares_colours.values())

        # create legend
        labels = transport_variables
        swatches = [mpatches.Patch(color=colours[j], label=labels[j]) for j in range(len(labels))]
        # #create a legend
        fig.legend(handles=swatches, labels=labels, loc='lower center', ncol=2, frameon=False)
        
        # add overall title
        fig.suptitle('Transport Final Energy ' + '(' + region + ')', fontsize=16)
        # add a legend
        # fig.legend(frameon=False)
        plt.show()


    def land_use_stacked_shares(self, selected_scenarios,
                                narrative_df, region=None):
        if region == None:
            region = 'World'
        
        illustrative_models = selected_scenarios['model'].tolist()
        illustrative_scenarios = selected_scenarios['scenario'].tolist()
        plt.rcParams['ytick.minor.visible'] = True
        land_use_variables = ['Land Cover|Pasture', 'Land Cover|Cropland', 'Land Cover|Forest', 'Land Cover']
        colours = ['#955251', '#33FF33', '#009933', 'darkgrey']
        # names = ['Warning lights', 'Resource risk', 'Sustainability struggle', 'Eco balance']
        names = ['A', 'B', 'C', 'D']
        
        # set up the figure
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

        for i in range(0, len(illustrative_scenarios)):

            plotting_df = pd.DataFrame()

            # set the scenario and model
            scenario = illustrative_scenarios[i]
            model = illustrative_models[i]

            narrative_df_scenario = narrative_df.filter(scenario=scenario, model=model, year=range(2020, 2101), region=region)
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
        fig.suptitle('CDR Mix ('+ region + ')', fontsize=16)
        
        plt.show()
        
            # loop through each energy variable


    def CDR_stacked_shares(self, selected_scenarios, narrative_df, region=None):

        if region == None:
            region = 'World'
        
        
        illustrative_models = selected_scenarios['model'].tolist()
        illustrative_scenarios = selected_scenarios['scenario'].tolist()
        # try:
        #     narrative_df = Data.narrative_data
        # except AttributeError:
        #     narrative_df = Utils.data_download_sub(Data.narrative_variables, illustrative_models,
        #                                            illustrative_scenarios, 'World', 2100)
        #     narrative_df.to_csv('outputs/narrative_data' + str(Data.categories) + '.csv')
        
        plt.rcParams['ytick.minor.visible'] = True
        # names = ['Problem Pathway', 'Resource Risk', 'Sustainability Struggle', 'Eco-tech Endeavour']
        names = ['A', 'B', 'C', 'D']
        CDR_variables = ['Carbon Sequestration|CCS|Biomass','Carbon Sequestration|Land Use','Carbon Sequestration|Direct Air Capture']
        colours = ['#FF6600', '#FFCC00', '#FF0000']

        # set up the figure
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

        for i in range(0, len(illustrative_scenarios)):
            
            Plotting.plotting_df = pd.DataFrame()
            # set the scenario and model
            scenario = illustrative_scenarios[i]
            model = illustrative_models[i]
            narrative_df_scenario = narrative_df.filter(scenario=scenario, model=model, year=range(2020, 2101), region=region)
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
                        pass
                        # narrative_df_scenario = Data.land_use_seq_data.filter(scenario=scenario, model=model, year=range(2020, 2101), region='World')
                        # df = narrative_df_scenario.filter(variable='Imputed|Carbon Sequestration|Land Use')
                        # interpolated_variable = df.interpolate(range(2020, 2101)).data.copy()
                        # Plotting.plotting_df[variable] = interpolated_variable['value']
                
            # make a stack plot for the scenario
            axs.flatten()[i].stackplot(range(2020, 2101), Plotting.plotting_df.T, labels=CDR_variables, colors=colours, alpha=0.4, edgecolor='darkgrey', linewidth=0.25)

            # title = names[i] + ' (' + scenario + ')'
            title = names[i] + ' (' + scenario + ')'
            # set the title of the plot
            axs.flatten()[i].set_title(title)

            # set the x and y axis labels
            axs.flatten()[i].set_xlabel('Year')
            axs.flatten()[i].set_ylabel('MtCO2 Sequestered per year')

            # set x limits
            axs.flatten()[i].set_xlim(2020, 2100)

            # set y limits
            axs.flatten()[i].set_ylim(0, 14000)
        
        # create legend
        labels = CDR_variables
        fig.suptitle('CDR Mix ('+ region + ')', fontsize=16)
        swatches = [mpatches.Patch(color=colours[j], label=labels[j]) for j in range(len(labels))]
        # #create a legend
        fig.legend(handles=swatches, labels=labels, loc='lower center', ncol=2, frameon=False)

        plt.show()



    def regional_differences_across_scenarios(self, dimension_scores_global, dimension_scores_regional, scenarios_list, selected_scenarios, cross_regional_norm=None):

        
        dimensions_list = ['economic','resilience','robustness']
        
        output_df = pd.DataFrame() 
        
        for region in R10_CODES:
            
            region_df = pd.DataFrame()
            print(region)
            dimension_scores_regional_selected = dimension_scores_regional[dimension_scores_regional['region'] == region]
            dimension_scores_regional_selected = dimension_scores_regional_selected.reset_index(drop=True)
            print(dimension_scores_regional_selected)
            region_df['scenario'] = dimension_scores_regional_selected['scenario']
            region_df['model'] = dimension_scores_regional_selected['model']
            region_df['region'] = dimension_scores_regional_selected['region']
            # region_df['economic_diff'] = dimension_scores_regional_selected['economic_dimension_score'] - dimension_scores_global['economic_score']
            # region_df['environmental_diff'] = dimension_scores_regional_selected['environmental_dimension_score'] - dimension_scores_global['environmental_score']
            # region_df['resource_diff'] = dimension_scores_regional_selected['resource_dimension_score'] - dimension_scores_global['resource_score']
            # region_df['resilience_diff'] = dimension_scores_regional_selected['resilience_dimension_score'] - dimension_scores_global['resilience_score']
            # region_df['robustness_diff'] = dimension_scores_regional_selected['robustness_dimension_score'] - dimension_scores_global['robustness_score']
            region_df['economic_diff'] = dimension_scores_regional_selected['economic_dimension_score'] 

  
            region_df['resilience_diff'] = dimension_scores_regional_selected['resilience_dimension_score'] 
            region_df['robustness_diff'] = dimension_scores_regional_selected['robustness_dimension_score'] 
            print(region_df)
            output_df = pd.concat([output_df, region_df], axis=0)
        print(output_df)
        
        if cross_regional_norm == None:
            # add global scores as 'World' region
            global_df = pd.DataFrame()
            global_df['scenario'] = dimension_scores_global['scenario']
            global_df['model'] = dimension_scores_global['model']
            global_df['region'] = 'World'
            global_df['economic_diff'] = dimension_scores_global['economic_score']
            global_df['resilience_diff'] = dimension_scores_global['resilience_score']
            global_df['robustness_diff'] = dimension_scores_global['robustness_score']
            output_df = pd.concat([output_df, global_df], axis=0)

        # output_df.to_csv('regional_differences.csv')
        # set up the figure so that it is a Overlapping densities (‘ridge plot’) from seaborn
        # with the regions on the y axis and the distribution of the differences on the x axis
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

        for i, dimension in enumerate(dimensions_list):
            # sns.violinplot(x=dimension + '_diff', y='region', data=output_df, ax=axs[i], palette='terrain', linewidth=0.4, inner='quart', cut=0, fill=False, alpha=0.7)
            sns.boxplot(x=dimension + '_diff', y='region', data=output_df, ax=axs[i], palette='terrain', linewidth=0.4, showfliers=False)
            axs[i].set_title(dimension)
            axs[i].set_ylabel('')
            axs[i].set_xlabel('Dimension score')
            axs[i].set_xlim(-0.1, 1.1)
            icon_list = ['v', 'o', 's', 'p']
            
            # # Extract the y values of the middle of the violin plots
            # y_coords_list = []
            # collections = axs[i].collections  # Collection from the correct axs[i]
            
            # # Process each collection which corresponds to each region's violin plot
            # for j in range(len(output_df['region'].unique())):
            #     # Assuming each region corresponds to two paths (left and right of the violin)
            #     path = collections[j].get_paths()[0]  # Typically, each collection will have one path, but this might need adjusting
            #     vertices = path.vertices
            #     y_coords = vertices[:, 1]  # Extract y-coordinates
            #     # Find the y-coordinate of the middle of the plot
            #     y_mid = np.mean(y_coords)
            #     y_coords_list.append(y_mid)

                        # Extract y values of the middle of the violin plots
            y_coords_list = []
            region_order = axs[i].get_yticklabels() # Get the order of regions as displayed
            y_coords_list = axs[i].get_yticks() # Get the y-position of each region

            # y_coords_list.append((region, region_position))
                    
            print(y_coords_list)
            
            
            marker_colours = ['black', 'red', 'grey', 'lightgrey']
            # Add an icon for each selected scenario
            for j, scenario in enumerate(selected_scenarios['scenario']):
                scenario_diff = output_df[(output_df['scenario'] == scenario) & 
                                        (output_df['model'] == selected_scenarios['model'][j])]
                scenario_diff_values = scenario_diff[dimension + '_diff'].values.flatten()
                # Ensure that there's a y-coordinate for each value to plot
                if len(scenario_diff_values) == len(y_coords_list):
                    axs[i].scatter(scenario_diff_values, y_coords_list, marker=icon_list[j], s=50, color=marker_colours[j], alpha=0.9, edgecolor='grey', linewidth=0.3, zorder=10)
                else:
                    print("Mismatch in number of points to plot")
                #         Plot each scenario value at the corresponding y-coordinate
                # for val, region in zip(scenario_diff_values, scenario_regions):
                #     for region_name, y_mid in y_coords_list:
                #         if region == region_name:
                #             axs[i].scatter(val, y_mid, marker=icon_list[j], s=100, color='black')

        # set the y axis labels as data.R10_codes
        axs[0].set_yticklabels(R10_CODES)

        # make a list of the markers
        markers = icon_list
        # # make a list of the labels
        labels = selected_scenarios['scenario'].tolist()
        # Create a list of Line2D objects for the legend
        swatches = [mlines.Line2D([], [], color=marker_colours[i], marker=icon_list[i], linestyle='None',
                                markersize=10, label=labels[i]) for i in range(len(labels))]

        # #create a legend
        fig.legend(handles=swatches, labels=labels, loc='lower center', ncol=4, fontsize=10, frameon=False)
        # make legends for markers for the icons and their colours for the scenarios

        plt.show()


    def specific_dimension_regional_analysis(self, input_dimension_scores_global, input_dimension_scores_regional,
                                             selected_region_south, selected_region_north,
                                             model_families):

        """
        Plot for breaking down dimension scores for a specific region and comparing with global values

        Inputs:
        - dimension_scores_global: a dataframe containing the global dimension scores
        - dimension_scores_regional: a dataframe containing the regional dimension scores
        - scenarios_list: a list of scenarios to plot
        - selected_scenarios: a dataframe containing the selected scenarios
        - selected_region: the region to plot

        Outputs:
        - split sns violin plot showing the dimension and indicator scores 
        for the selected region and scenarios compared to the global scores

        """

        dimension = 'robustness'

        # import relevant dimension scores
        scores_output = pd.DataFrame()

        dimension_scores_global = input_dimension_scores_global['robustness_score']
        dimension_scores_regional_south = input_dimension_scores_regional[input_dimension_scores_regional['region'] == selected_region_south]['robustness_dimension_score']
        dimension_scores_regional_south = dimension_scores_regional_south.reset_index(drop=True)
        dimension_scores_regional_north = input_dimension_scores_regional[input_dimension_scores_regional['region'] == selected_region_north]['robustness_dimension_score']
        dimension_scores_regional_north = dimension_scores_regional_north.reset_index(drop=True)

        # add a column to the scenario scores with the model family
        to_append_global = pd.DataFrame()
        to_append_global['scores'] = dimension_scores_global
        to_append_global['region'] = 'global'

        to_append_regional_south = pd.DataFrame()
        to_append_regional_south['scores'] = dimension_scores_regional_south
        to_append_regional_south['region'] = selected_region_south

        to_append_regional_north = pd.DataFrame()
        to_append_regional_north['scores'] = dimension_scores_regional_north
        to_append_regional_north['region'] = selected_region_north

        scores_output = pd.concat([to_append_global, to_append_regional_south, to_append_regional_north], axis=0)
        scores_output['scenario'] = input_dimension_scores_regional['scenario']
        scores_output['model'] = input_dimension_scores_regional['model']
        scores_output['x_variable'] = [0] * len(scores_output)
        model_families = model_families
        print(model_families)
        scores_output['model_family'] = scores_output['model'].map(model_families.set_index('model')['model_family'])

        # Calculate median scores
        median_scores_global = scores_output[scores_output['region']=='global'].groupby('model_family')['scores'].median().reset_index()
        median_scores_regional_south = scores_output[scores_output['region']==selected_region_south].groupby('model_family')['scores'].median().reset_index()
        median_scores_regional_north = scores_output[scores_output['region']==selected_region_north].groupby('model_family')['scores'].median().reset_index()

        # import indicator scores 
        global_energy_system_flexiblity = IndexBuilder.energy_system_flexibility['flexibility_score']
        south_regional_energy_system_flexibility = IndexBuilder.regional_energy_system_flexibility[IndexBuilder.regional_energy_system_flexibility['region'] == selected_region_south]['flexibility_score']
        south_regional_energy_system_flexibility = south_regional_energy_system_flexibility.reset_index(drop=True)
        north_regional_energy_system_flexibility = IndexBuilder.regional_energy_system_flexibility[IndexBuilder.regional_energy_system_flexibility['region'] == selected_region_north]['flexibility_score']
        north_regional_energy_system_flexibility = north_regional_energy_system_flexibility.reset_index(drop=True)

        global_carbon_budgets = IndexBuilder.carbon_budgets['carbon_budget_share']
        south_regional_carbon_budgets = IndexBuilder.regional_carbon_budgets[IndexBuilder.regional_carbon_budgets['region'] == selected_region_south]['carbon_budget_share']
        south_regional_carbon_budgets = south_regional_carbon_budgets.reset_index(drop=True)
        north_regional_carbon_budgets = IndexBuilder.regional_carbon_budgets[IndexBuilder.regional_carbon_budgets['region'] == selected_region_north]['carbon_budget_share']
        north_regional_carbon_budgets = north_regional_carbon_budgets.reset_index(drop=True)

        global_low_carbon_diversity = IndexBuilder.low_carbon_diversity['shannon_index']
        south_regional_low_carbon_diversity = IndexBuilder.regional_low_carbon_diversity[IndexBuilder.regional_low_carbon_diversity['region'] == selected_region_south]['shannon_index']
        south_regional_low_carbon_diversity = south_regional_low_carbon_diversity.reset_index(drop=True)
        north_regional_low_carbon_diversity = IndexBuilder.regional_low_carbon_diversity[IndexBuilder.regional_low_carbon_diversity['region'] == selected_region_north]['shannon_index']
        north_regional_low_carbon_diversity = north_regional_low_carbon_diversity.reset_index(drop=True)

        global_CDR_2050 = IndexBuilder.regional_CDR_2050[IndexBuilder.regional_CDR_2050['region'] == 'World']['total_CDR_land']
        south_regional_CDR_2050 = IndexBuilder.regional_CDR_2050[IndexBuilder.regional_CDR_2050['region'] == selected_region_south]['total_CDR_land']
        south_regional_CDR_2050 = south_regional_CDR_2050.reset_index(drop=True)
        north_regional_CDR_2050 = IndexBuilder.regional_CDR_2050[IndexBuilder.regional_CDR_2050['region'] == selected_region_north]['total_CDR_land']
        north_regional_CDR_2050 = north_regional_CDR_2050.reset_index(drop=True)

        #minmax scale the CDR_2050 scores
        # regional_CDR_2050 = (regional_CDR_2050 - regional_CDR_2050.min()) / (regional_CDR_2050.max() - regional_CDR_2050.min())
        # global_CDR_2050 = (global_CDR_2050 - global_CDR_2050.min()) / (global_CDR_2050.max() - global_CDR_2050.min())

        # make a dictionary of the indicators and their keys
        indicators_dict = {'energy_system_flexibility': [global_energy_system_flexiblity, south_regional_energy_system_flexibility, north_regional_energy_system_flexibility],
                      'carbon_budgets': [global_carbon_budgets, south_regional_carbon_budgets, north_regional_carbon_budgets],
                      'low_carbon_diversity': [global_low_carbon_diversity, south_regional_low_carbon_diversity, north_regional_low_carbon_diversity],
                      'CDR_2050': [global_CDR_2050, south_regional_CDR_2050, north_regional_CDR_2050]}

        indicators = ['energy_system_flexibility',
                       'carbon_budgets', 
                       'low_carbon_diversity',
                         'CDR_2050']
        # indicator_keys = ['flexibility_score',
        #                   'carbon_budget_share',
        #                   'shannon_index',
        #                   'CDR_2050_score']
        
        
        plt.rcParams['ytick.minor.visible'] = True
        # set up the figure with 1 + n items from list
        # set up the figure
        fig, axs = plt.subplots(1, len(indicators)+1, figsize=(12, 10))

        # First split violin looking at dimension scores 
        # sns.violinplot(data=scores_output, x='x_variable', y="scores", ax=axs[0], hue="region",
        #        split=True, inner="quart", fill=False, cut=0, palette={selected_region: "#20DFA3", "global": ".35"}, linewidth=0.8)
        sns.boxplot(data=scores_output, x='x_variable', y="scores", ax=axs[0], hue="region",
                    palette={selected_region_south: "#A2D5AF", selected_region_north: '#DDD898', "global": "#DADAE2"}, linewidth=0.8)
        axs[0].legend().set_visible(False)
                # Position the 'global' scores just to the left and 'regional' scores just to the right of the x-axis marker
        global_x_pos = -0.1
        south_regional_x_pos = 0.1
        north_regional_x_pos = 0.15

        # Plot medians and connecting lines
        for model_family in median_scores_global['model_family'].unique():
            global_score = median_scores_global[median_scores_global['model_family'] == model_family]['scores'].values[0]
            south_regional_score = median_scores_regional_south[median_scores_regional_south['model_family'] == model_family]['scores'].values[0]
            north_regional_score = median_scores_regional_north[median_scores_regional_north['model_family'] == model_family]['scores'].values[0]

            # axs[0].scatter(x=[global_x_pos], y=[global_score], color=Plotting.model_colours[model_family], label=f'{model_family} Global' if model_family == median_scores_global['model_family'].unique()[0] else "")
            # axs[0].scatter(x=[south_regional_x_pos], y=[south_regional_score], color=Plotting.model_colours[model_family], label=f'{model_family} Regional' if model_family == median_scores_global['model_family'].unique()[0] else "")
            # axs[0].scatter(x=[north_regional_x_pos], y=[north_regional_score], color=Plotting.model_colours[model_family], label=f'{model_family} Regional' if model_family == median_scores_global['model_family'].unique()[0] else "")
            # # axs[0].plot([global_x_pos, regional_x_pos], [global_score, regional_score], color=Plotting.model_colours[model_family], linestyle='--', alpha=0.5)


        # now iterate through the indicators and plot them in the same way
        for i, indicator in enumerate(indicators):
            global_indicator = indicators_dict[indicator][0]
            south_regional_indicator = indicators_dict[indicator][1]
            north_regional_indicator = indicators_dict[indicator][2]

            indicator_output = pd.DataFrame()
            indicator_output['scores'] = global_indicator
            indicator_output['region'] = 'global'
            indicator_output['scenario'] = input_dimension_scores_regional['scenario']
            indicator_output['model'] = input_dimension_scores_regional['model']
            indicator_output['x_variable'] = [0] * len(indicator_output)

            to_append_regional_south = pd.DataFrame()
            to_append_regional_south['scores'] = south_regional_indicator
            to_append_regional_south['region'] = selected_region_south
            to_append_regional_south['scenario'] = input_dimension_scores_regional['scenario']
            to_append_regional_south['model'] = input_dimension_scores_regional['model']
            to_append_regional_south['x_variable'] = [0] * len(to_append_regional_south)

            to_append_regional_north = pd.DataFrame()
            to_append_regional_north['scores'] = north_regional_indicator
            to_append_regional_north['region'] = selected_region_north
            to_append_regional_north['scenario'] = input_dimension_scores_regional['scenario']
            to_append_regional_north['model'] = input_dimension_scores_regional['model']
            to_append_regional_north['x_variable'] = [0] * len(to_append_regional_north)

            indicator_output = pd.concat([indicator_output, to_append_regional_south, to_append_regional_north], axis=0)
            indicator_output['model_family'] = indicator_output['model'].map(model_families.set_index('model')['model_family'])

            # Calculate median scores
            median_scores_global = indicator_output[indicator_output['region']=='global'].groupby('model_family')['scores'].median().reset_index()
            median_scores_regional_south = indicator_output[indicator_output['region']==selected_region_south].groupby('model_family')['scores'].median().reset_index()
            median_scores_regional_north = indicator_output[indicator_output['region']==selected_region_north].groupby('model_family')['scores'].median().reset_index()

            # First split violin looking at dimension scores 
            # sns.violinplot(data=indicator_output, x='x_variable', y="scores", ax=axs[i+1], hue="region",
            #        split=True, inner="quart", fill=False, cut=0, palette={selected_region: "#20DFA3", "global": ".35"}, linewidth=0.8,
            #        legend=False)
            sns.boxplot(data=indicator_output, x='x_variable', y="scores", ax=axs[i+1], hue="region",
                    palette={selected_region_south: "#A2D5AF", selected_region_north: '#DDD898', "global": "#DADAE2"}, linewidth=0.8)
            axs[i].legend().set_visible(False)
            
            # Position the 'global' scores just to the left and 'regional' scores just to the right of the x-axis marker
            global_x_pos = -0.35
            south_regional_x_pos = 0.1
            north_regional_x_pos = 0.15

            # Plot medians and connecting lines
            for model_family in median_scores_global['model_family'].unique():
                global_score = median_scores_global[median_scores_global['model_family'] == model_family]['scores'].values[0]
                south_regional_score = median_scores_regional_south[median_scores_regional_south['model_family'] == model_family]['scores'].values[0]
                north_regional_score = median_scores_regional_north[median_scores_regional_north['model_family'] == model_family]['scores'].values[0]

                # axs[i+1].scatter(x=[global_x_pos], y=[global_score], color=Plotting.model_colours[model_family], label=f'{model_family} Global' if model_family == median_scores_global['model_family'].unique()[0] else "")
                # axs[i+1].scatter(x=[south_regional_x_pos], y=[south_regional_score], color=Plotting.model_colours[model_family], label=f'{model_family} Regional' if model_family == median_scores_global['model_family'].unique()
                # [0] else "")
                # axs[i+1].scatter(x=[north_regional_x_pos], y=[north_regional_score], color=Plotting.model_colours[model_family], label=f'{model_family} Regional' if model_family == median_scores_global['model_family'].unique()[0] else "")
                # axs[i+1].plot([global_x_pos, regional_x_pos], [global_score, regional_score], color=Plotting.model_colours[model_family], linestyle='--', alpha=0.5)

            # set the title of the plot
            axs[i+1].set_title(indicator)

        plt.show()


    def duplicate_scenarios_plot(self, scenario_scores):

        """
        Plot duplicate scenarios radar plots
       
        Inputs: 
        - scenario_scores: a dataframe containing the scenario scores for each dimension

        Outputs:
        - Radar plots for each duplicate scenario
        
        """
        model_colours = {'MESSAGEix-GLOBIOM_1.1':'#56B4E9', 'REMIND-MAgPIE 2.1-4.2':'#009E73', 'WITCH 5.0':'#CC79A7'}


        # Find duplicate scenarios
        duplicates = scenario_scores[scenario_scores.duplicated(subset='scenario', keep=False)]

        # get models
        models = duplicates['model'].unique().tolist()
        scenarios = duplicates['scenario'].unique().tolist()

        all_models_duplicated = []
        for scenario in scenarios:

            # filter for the scenario
            scenario_check = duplicates[duplicates['scenario'] == scenario]

            if len(scenario_check) == len(models):
                all_models_duplicated.append(scenario)
        

        categories = list(scenario_scores)[2:8]
        # set up the figure
        fig, axs = plt.subplots(1, 3, figsize=(10, 10), subplot_kw=dict(polar=True))
        N = len(categories)
        for i, scenario in enumerate(all_models_duplicated):

            ax = axs.flatten()[i]
            scenario_data = duplicates.loc[duplicates['scenario'] == scenario]
            
            # loop through each model
            for model in models:
                
                model_scenario_data = scenario_data.loc[scenario_data['model'] == model]
                stats = model_scenario_data[categories].values.flatten().tolist()
                angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
                stats += stats[:1]
                angles += angles[:1]
                ax.plot(angles, stats, color=model_colours[model], linewidth=2, alpha=0.6)

            # # Labels for each point
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)


            #set the y limit
            ax.set_ylim(0, 1)

            # Title for each subplot with the scenario name
            ax.set_title(scenario,size=12, y=1.1, fontweight='bold')

        # Show the figure
        plt.tight_layout()
        plt.show()


    def count_pairwise_low_scores(self, scenario_scores, low_score_threshold=float):
        """
        Function to count the number of scenarios scoring low on pairs of dimensions.
        
        Inputs: 
        - scenario_scores: a dataframe containing the scenario scores for each dimension
        - low_score_threshold: the threshold below which a score is considered low
        
        Outputs: 
        - pairwise_low_counts: a DataFrame containing the counts of low scores for each pair of dimensions
        """
        # Drop unnecessary columns and set index if needed
        scenario_scores = scenario_scores.set_index(['scenario', 'model'], drop=True)
        scenario_scores = scenario_scores.drop(columns=['cluster'])
        
        # Define the low score threshold for each dimension (assuming a percentile for simplicity)
        thresholds = scenario_scores.quantile(low_score_threshold)
        # print(thresholds)
        # thresholds = low_score_threshold

        # Initialize a DataFrame to store the counts
        dimensions = scenario_scores.columns
        print(dimensions)
        pairwise_counts = pd.DataFrame(0, index=dimensions, columns=dimensions)
        
        # Calculate the pairwise counts
        for dim1, dim2 in combinations(dimensions, 2):
            low_dim1 = scenario_scores[dim1] < thresholds[dim1]
            low_dim2 = scenario_scores[dim2] < thresholds[dim2]
            count = np.sum(low_dim1 & low_dim2)
            pairwise_counts.loc[dim1, dim2] = count
            pairwise_counts.loc[dim2, dim1] = count
        
        # Convert the pairwise counts matrix to a long-form DataFrame
            # Convert the pairwise counts matrix to a long-form DataFrame
        plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
        sns.set(font_scale=0.8)  # Smaller font size for better fit

        # Create heatmap
        heatmap = sns.heatmap(
            pairwise_counts, 
            annot=True, 
            cmap="plasma", 
            cbar_kws={'label': 'Count of Low Scores', 'format': '%.0f'},
            linewidths=0.1,
            square=True,
            vmax=15,  # Add lines between squares for clarity
              # Ensures each cell is square square=True
        )

        # Adjust y-axis labels to be horizontal
        heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)

        # Set labels and title
        plt.title('Pairwise Low Score Counts')
        plt.xlabel('Dimension')
        plt.ylabel('Dimension')

        # Display the plot
        plt.tight_layout()  # Adjust layout to fit everything nicely
        plt.show()
    
    # # Example usage
    # # Assuming `df` is your DataFrame containing the scenario scores with columns ['scenario', 'model', 'cluster', 'dimension1', 'dimension2', ..., 'dimension5']
    # df = pd.read_csv('your_scenario_scores.csv')
    
    # print(pairwise_low_counts)

    # make a plot of a given variable / calculated variable for all the scenarios given showing median, 25th and 75th percentiles
    def electrification_plots(pyam_df, start_year, end_year):

        """
        Function that produces timeseries plot of distribution of electrification rates for all scenarios in the pyam dataframe
        
        Inputs:
        - pyam_df: the pyam dataframe containing the data
        - start_year: the start year of the plot
        - end_year: the end year of the plot

        Outputs:
        - timeseries plot of the distribution of electrification rates for all scenarios in the pyam dataframe

        """

        print(pyam_df)

        # set the params
        plt.rcParams['ytick.minor.visible'] = True
        plt.rcParams['xtick.minor.visible'] = True

        # set up the figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # variable='Secondary Energy|Electricity'


        # filter the data for the electrification rate
        df = pyam_df.filter(year=range(start_year, end_year))


        # interpolate the data
        interpolated_data = df.interpolate(range(start_year, end_year)).data.copy()

        # create new variable for the electricity share of final energy
        df_pivot =  interpolated_data.pivot_table(
            index=["model", "scenario", "region", "year"], 
            columns="variable", 
            values="value"
        ).reset_index()

        # calculate the electrification rate
        df_pivot['Emissions intensity'] = df_pivot['Emissions|CO2'] / df_pivot['Final Energy']

        interpolated_data = df_pivot.melt(
            id_vars=["model", "scenario", "region", "year"],
            value_vars=["Emissions intensity"],  # Add other variables if needed
            var_name="variable",
            value_name="value"
)

        medians = interpolated_data.groupby('year')['value'].median()
        q25 = interpolated_data.groupby('year')['value'].quantile(0.25)
        q75 = interpolated_data.groupby('year')['value'].quantile(0.75)
        q5 = interpolated_data.groupby('year')['value'].quantile(0.05)
        q95 = interpolated_data.groupby('year')['value'].quantile(0.95)

        years = interpolated_data['year'].unique()

        # plot the data with the median line, 25th and 75th percentiles as fill between
        ax.plot(years, medians, color='orange', label='Median')
        ax.fill_between(years, q25, q75, color='orange', alpha=0.2, label='25th-75th Percentile')
        ax.fill_between(years, q5, q95, color='orange', alpha=0.1, label='5th-95th Percentile')

        # set x min and max
        ax.set_xlim(start_year, end_year-1)

        # # set the y axis label
        # ax.set_ylabel(variable)

        # set the x axis label
        ax.set_xlabel('Year')

        plt.tight_layout()
        plt.show()
        


    # function that creates a parallel coordinate plot with the normalised scores for each dimension, coloured by cluster
    # this function is currently designed for global data
    def parallel_coord_plot(self, clustered_normalised_scores, colours):

        
        # Plot Parallel Coordinates
        plt.figure(figsize=(12, 6))
        
        # Select only columns that match 'dimension' + '_score' and rename them
        dimension_columns = [col for col in clustered_normalised_scores.columns if col.endswith("_score")]
        renamed_columns = {col: re.sub(r"_score$", "", col) for col in dimension_columns}
        data_selected = clustered_normalised_scores[dimension_columns + ["cluster"]].rename(columns=renamed_columns)

        # Create parallel coordinates plot
        pd.plotting.parallel_coordinates(
            data_selected,
            'cluster',
            color=colours,
            alpha=0.5,
            axvlines=True
        )

        # Customize plot
        plt.xticks(rotation=45)
        plt.ylabel('Normalized Score')
        plt.title('Scenario Dimension Scores by Cluster')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    
    # function that creates a correlation heatmap with each score representing the correlation between the dimensions
    # this function is currently designed for global data
    def correlation_heatmap(self, normalised_scores):
        
        # set index to scenario and model
        normalised_scores = normalised_scores.set_index(['scenario', 'model'])

        # Calculate correlation matrix and p-values
        cols = normalised_scores.columns
        corr_matrix = normalised_scores.corr()
        p_matrix = pd.DataFrame(np.zeros_like(corr_matrix), columns=cols, index=cols)
        
        # Fill p-value matrix
        for i in cols:
            for j in cols:
                corr, p = stats.pearsonr(normalised_scores[i], normalised_scores[j])
                p_matrix.loc[i,j] = p

        # Create mask for significant correlations (p < 0.05)

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        
        # Plot correlation heatmap with stars for significant values
        sns.heatmap(corr_matrix, 
                    annot=True, 
                    cmap='coolwarm', 
                    vmin=-1, 
                    vmax=1,
                     # This will hide non-significant correlations
                    fmt='.2f')  # Format to 2 decimal places
        
        # Add asterisks for significance levels
        for i in range(len(cols)):
            for j in range(len(cols)):
                if p_matrix.iloc[i,j] <= 0.001:
                    plt.text(j+0.5, i+0.85, '***', ha='center', va='center')
                elif p_matrix.iloc[i,j] <= 0.01:
                    plt.text(j+0.5, i+0.85, '**', ha='center', va='center')
                elif p_matrix.iloc[i,j] <= 0.05:
                    plt.text(j+0.5, i+0.85, '*', ha='center', va='center')
                elif p_matrix.iloc[i,j] > 0.05:
                    plt.text(j+0.5, i+0.85, '', ha='center', va='center')

        plt.title('Correlation Matrix of Normalised Dimension Scores\n* p<0.05, ** p<0.01, *** p<0.001')
        plt.tight_layout()
        plt.show()







if __name__ == "__main__":
    main()
