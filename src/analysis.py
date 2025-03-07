import numpy as np
import pyam
import pandas as pd
from itertools import combinations, product
from resources import NaturalResources
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from constants import *
from utils.file_parser import *


    # try:
    #     archetypes = pd.read_csv('outputs/scenario_archetypes' + str(Data.categories) + '.csv')
    #     selected_scenarios = pd.read_csv('outputs/selected_scenarios' + str(Data.categories) + '.csv', index_col=0)
    #     centroid_scenarios = pd.read_csv('outputs/closest_to_centroids' + str(Data.categories) + '.csv', index_col=0)
    # except:
    #     print('find_scenario_archetypes function has not been run yet')


def main(categories=None, scenarios=None, run_regional=None) -> None:

    if scenarios is None:
        scenarios = read_csv(PROCESSED_DIR + 'Framework_scenarios' + str(categories))

    if categories is None:
        categories = CATEGORIES_DEFAULT
    
    try:
        print('Reading in the index data')
        investment_metrics = pd.read_csv(OUTPUT_DIR + 'energy_supply_investment_score' + str(categories) + '.csv')
        environment_metrics = pd.read_csv(OUTPUT_DIR + 'environmental_metrics' + str(categories) + '.csv')
        resource_metrics = pd.read_csv(OUTPUT_DIR + 'material_use_ratios' + str(categories) + '.csv')
        transition_speed_metrics = pd.read_csv(OUTPUT_DIR + 'transition_speed_metrics' + str(categories) + '.csv')

        # import resilience metrics
        final_energy_demand = pd.read_csv(OUTPUT_DIR + 'final_energy_demand' + str(categories) + '.csv')    
        energy_diversity = pd.read_csv(OUTPUT_DIR + 'shannon_diversity_index' + str(categories) + '.csv')   
        gini_coefficient = pd.read_csv(OUTPUT_DIR + 'gini_coefficient' + str(categories) + '.csv')
        electricity_price = pd.read_csv(OUTPUT_DIR + 'electricity_prices' + str(categories) + '.csv')

        # import robustness metrics
        energy_system_flexibility = pd.read_csv(OUTPUT_DIR + 'flexibility_scores' + str(categories) + '.csv')
        carbon_budgets = pd.read_csv(OUTPUT_DIR + 'carbon_budget_shares' + str(categories) + '.csv')
        low_carbon_diversity = pd.read_csv(OUTPUT_DIR + 'low_carbon_shannon_diversity_index' + str(categories) + '.csv')
        CDR_2050 = pd.read_csv(OUTPUT_DIR + 'total_CDR' + str(categories) + '.csv')

        # import fairness metrics
        between_region_gini = pd.read_csv(OUTPUT_DIR + 'between_region_gini' + str(categories) + '.csv')
        carbon_budget_fairness = pd.read_csv(OUTPUT_DIR + 'carbon_budget_fairness' + str(categories) + '.csv')

    except FileNotFoundError:
        print('Index data are not available yet. Please ensure you have run all the analysis scripts.')

    # economic_score(investment_metrics, categories)
    # environment_score(environment_metrics, categories)
    # resource_score(NaturalResources.minerals, resource_metrics, categories)
    # resilience_score(final_energy_demand,
    #                 energy_diversity,
    #                 gini_coefficient, electricity_price, categories)
    # fairness_score(between_region_gini, carbon_budget_fairness, categories)
    # robustness_score(energy_system_flexibility, 
    #                            low_carbon_diversity, 
    #                            carbon_budgets,
    #                            CDR_2050, categories)
    # transition_speed_score(transition_speed_metrics, categories)

    # import regional scores
    print('Calculting framework scores')
    economic_scores = read_csv(OUTPUT_DIR + 'economic_scores' + str(categories) + '.csv')
    environment_scores = read_csv(OUTPUT_DIR + 'environmental_scores' + str(categories) + '.csv')
    resource_scores = read_csv(OUTPUT_DIR + 'resource_scores' + str(categories) + '.csv')
    resilience_scores = read_csv(OUTPUT_DIR + 'resilience_scores' + str(categories) + '.csv')
    robustness_scores = read_csv(OUTPUT_DIR + 'robustness_scores' + str(categories) + '.csv')
    fairness_scores = read_csv(OUTPUT_DIR + 'fairness_scores' + str(categories) + '.csv')
    transition_speed_scores = read_csv(OUTPUT_DIR + 'transition_speed_scores' + str(categories) + '.csv')

    # combine and normalise the scores
    print('Combining and normalising scores')
    # normalise_scores(economic_scores, environment_scores, resource_scores, 
    #                         resilience_scores, robustness_scores, fairness_scores, 
    #                         transition_speed_scores,
    #                         categories)
    

    # get the scenario archetypes and illustrative scenarios
    print('Finding scenario archetypes and illustrative scenarios')
    # find_scenario_archetypes(categories, cluster_number=4)


    if run_regional == True:
        print('Now performing regional analysis')
        try:
            print('Reading in the regional index data')
            regional_investment_metrics = pd.read_csv(OUTPUT_DIR + 'energy_supply_investment_score_regional' + str(categories) + '.csv')
            regional_environment_metrics = pd.read_csv(OUTPUT_DIR + 'environmental_metrics_regional' + str(categories) + '.csv')
            regional_resource_metrics = pd.read_csv(OUTPUT_DIR + 'material_use_ratios' + str(categories) + '.csv')
            regional_transition_speed_metrics = pd.read_csv(OUTPUT_DIR + 'transition_speed_metrics_regional' + str(categories) + '.csv')

            # import resilience metrics
            regional_final_energy_demand = pd.read_csv(OUTPUT_DIR + 'final_energy_demand_regional' + str(categories) + '.csv')
            regional_energy_diversity = pd.read_csv(OUTPUT_DIR + 'shannon_diversity_index_regional' + str(categories) + '.csv')
            regional_gini_coefficient = pd.read_csv(OUTPUT_DIR + 'gini_coefficient_regional' + str(categories) + '.csv')
            regional_electricity_price = pd.read_csv(OUTPUT_DIR + 'electricity_prices_regional' + str(categories) + '.csv')
        
            # import robustness metrics
            regional_energy_system_flexibility = pd.read_csv(OUTPUT_DIR + 'flexibility_scores_regional' + str(categories) + '.csv')
            regional_carbon_budgets = pd.read_csv(OUTPUT_DIR + 'carbon_budget_shares_regional' + str(categories) + '.csv')
            regional_low_carbon_diversity = pd.read_csv(OUTPUT_DIR + 'low_carbon_shannon_diversity_index_regional' + str(categories) + '.csv')
            regional_CDR_2050 = pd.read_csv(OUTPUT_DIR + 'total_CDR_regional' + str(categories) + '.csv')

        except FileNotFoundError:
            print('Regional index data for your chosen categories not available yet. Please ensure you have run the analysis scripts for the regional data')
        print('Calculating regional scores')
        get_regional_scores(regional_investment_metrics, regional_environment_metrics,
                            regional_resource_metrics, regional_final_energy_demand,
                            regional_energy_diversity, regional_gini_coefficient,
                            regional_electricity_price, regional_energy_system_flexibility,
                            regional_low_carbon_diversity, regional_carbon_budgets, 
                            regional_CDR_2050, regional_transition_speed_metrics, categories, cross_region_norm=True)


# calculate the economic score (higher score = more economic challenges)
def economic_score(investment_scores, categories, regional=None, cross_region_norm=None):
    
    output_df = pd.DataFrame()
    output_df['model'] = investment_scores['model']
    output_df['scenario'] = investment_scores['scenario']
    energy_investment = investment_scores['mean_value']
    energy_investment_2050 = investment_scores['mean_value_2050']
    
    if cross_region_norm == None:
        # normalise the investment data
        energy_investment_normalised = (energy_investment - energy_investment.min()) / (energy_investment.max() - energy_investment.min())
        energy_investment_2050_normalised = (energy_investment_2050 - energy_investment_2050.min()) / (energy_investment_2050.max() - energy_investment_2050.min())

    else:
        
        # get the regional investment data
        regional_investment_metrics = pd.read_csv(OUTPUT_DIR + 'energy_supply_investment_score_regional' + str(categories) + '.csv')
        
        # normalise the investment data but from the min and max of all regions
        all_regional_energy_investment = regional_investment_metrics['mean_value']
        all_regional_energy_investment_2050 = regional_investment_metrics['mean_value_2050']
        energy_investment_normalised = (energy_investment - all_regional_energy_investment.min()) / (all_regional_energy_investment.max() - all_regional_energy_investment.min())
        energy_investment_2050_normalised = (energy_investment_2050 - all_regional_energy_investment_2050.min()) / (all_regional_energy_investment_2050.max() - all_regional_energy_investment_2050.min())


    output_df['investment_score'] = energy_investment_normalised
    output_df['investment_score_2050'] = energy_investment_2050_normalised
    
    if regional != None:
        output_df['Region'] = regional
        return output_df
    else:
        output_df.to_csv(OUTPUT_DIR + 'economic_scores' + str(categories) + '.csv', index=False)


# calculate the environmental score (higher score = more environmental challenges)
def environment_score(environment_metrics, categories, regional=None, cross_region_norm=None):

    output_df = pd.DataFrame()
    output_df['model'] = environment_metrics['model']
    output_df['scenario'] = environment_metrics['scenario']
    
    forest_change_2050 = -1 * environment_metrics['forest_change_2050']
    forest_change = -1 * environment_metrics['forest_change_2100']

    if cross_region_norm == None:
    # normalise the environmental data
        forest_change_normalised = (forest_change - forest_change.min()) / (forest_change.max() - forest_change.min())
        forest_change_2050_normalised = (forest_change_2050 - forest_change_2050.min()) / (forest_change_2050.max() - forest_change_2050.min())

    else:
        # get the regional investment data
        regional_environment_metrics = pd.read_csv(OUTPUT_DIR + 'environmental_metrics_regional' + str(categories) + '.csv')

        # normalise the investment data but from the min and max of all regions
        all_regional_forest_change = -1 * regional_environment_metrics['forest_change_2100']
        all_regional_forest_change_2050 = -1 * regional_environment_metrics['forest_change_2050']
        forest_change_normalised = (forest_change - all_regional_forest_change.min()) / (all_regional_forest_change.max() - all_regional_forest_change.min())
        forest_change_2050_normalised = (forest_change_2050 - all_regional_forest_change_2050.min()) / (all_regional_forest_change_2050.max() - all_regional_forest_change_2050.min())

    # add up the composite environmental score
    beccs_threshold_breached = environment_metrics['bioenergy_threshold_breached']
    output_df['environmental_score'] = forest_change_normalised + beccs_threshold_breached
    output_df['environmental_score_2050'] = forest_change_2050_normalised + beccs_threshold_breached

    if regional != None:
        output_df['Region'] = regional
        return output_df
    else:
        output_df.to_csv(OUTPUT_DIR + 'environmental_scores' + str(categories) + '.csv', index=False)


# calculate the resource score (higher score = more resource challenges)
def resource_score(minerals, resource_metrics, categories, regional=None):
   
    output_df = pd.DataFrame()
    output_df['model'] = resource_metrics['model']
    output_df['scenario'] = resource_metrics['scenario']
   
    """
    Weighting scheme:

    If a mineral has a score less than the threshold then it doesnt count:
    0-1 0 X mineral score

    """
    
    # loop through the materials and normalise the scores
    for mineral in minerals:
       
        mineral_score = resource_metrics[mineral]
        # make any values in the column that are less than 1 equal to 0
        mineral_score[mineral_score < 1] = 0
    #     mineral_score_normalised = (mineral_score - mineral_score.min()) / (mineral_score.max() - mineral_score.min())
    #     output_df[mineral] = mineral_score_normalised
        # try:
        #     output_df['total'] += mineral_score
        # except:
        #     output_df['total'] = mineral_score
        try:
            output_df['resource_score'] += mineral_score
        except:
            output_df['resource_score'] = mineral_score
    # output_df['resource_score'] = output_df['total'] / len(minerals)
    if regional != None:
        output_df['Region'] = regional
        return output_df
    
    else:
        output_df.to_csv(OUTPUT_DIR + 'resource_scores' + str(categories) + '.csv', index=False)


# calculate the resilience score (higher score = more resilience challenges)
def resilience_score(final_energy_demand, energy_diversity, gini_coefficient, electricity_price, categories, 
                     regional=None, cross_region_norm=None):
    """
    composite of 
    - final energy demand (lower better) weighting 1/4
    - energy diversity (higher better) weighting 1/4
    - gini coefficient (lower better) weighting 1/4
    - electricity price (lower better) weighting 1/4
    """
    outout_df = pd.DataFrame()
    outout_df['model'] = final_energy_demand['model']
    outout_df['scenario'] = final_energy_demand['scenario']

    # normalise the final energy demand
    if regional != None:
        final_energy_demand = final_energy_demand['energy_per_gdp']
    else:
        final_energy_demand = final_energy_demand['final_energy_demand']
    
    energy_diversity = -1 * energy_diversity['shannon_index']
    gini_coefficient = gini_coefficient['ssp_gini_coefficient']
    electricity_price = electricity_price['electricity_price']

    if cross_region_norm == None:
        # normalise the final energy demand
        final_energy_demand_normalised = (final_energy_demand - final_energy_demand.min()) / (final_energy_demand.max() - final_energy_demand.min())
        outout_df['final_energy_demand'] = final_energy_demand_normalised
        
        # normalise the energy diversity
        energy_diversity_normalised = (energy_diversity - energy_diversity.min()) / (energy_diversity.max() - energy_diversity.min())
        outout_df['energy_diversity'] = energy_diversity_normalised

        # normalise the gini coefficient
        gini_coefficient_normalised = (gini_coefficient - gini_coefficient.min()) / (gini_coefficient.max() - gini_coefficient.min())
        outout_df['gini_coefficient'] = gini_coefficient_normalised

        # normalise the electricity price
        electricity_price_normalised = (electricity_price - electricity_price.min()) / (electricity_price.max() - electricity_price.min())
        outout_df['electricity_price'] = electricity_price_normalised

    else:
        # get the regional data
        regional_final_energy_demand = pd.read_csv(OUTPUT_DIR + 'final_energy_demand_regional' + str(categories) + '.csv')
        regional_energy_diversity = pd.read_csv(OUTPUT_DIR + 'shannon_diversity_index_regional' + str(categories) + '.csv')
        regional_gini_coefficient = pd.read_csv(OUTPUT_DIR + 'gini_coefficient_regional' + str(categories) + '.csv')
        regional_electricity_price = pd.read_csv(OUTPUT_DIR + 'electricity_prices_regional' + str(categories) + '.csv')
        
        # normalise the final energy demand across all regions
        all_regional_final_energy_demand = regional_final_energy_demand['energy_per_gdp']
        final_energy_demand_normalised = (final_energy_demand - all_regional_final_energy_demand.min()) / (all_regional_final_energy_demand.max() - all_regional_final_energy_demand.min())
        outout_df['final_energy_demand'] = final_energy_demand_normalised
        
        # normalise the energy diversity across all regions
        all_regional_energy_diversity = -1 * regional_energy_diversity['shannon_index']
        energy_diversity_normalised = (energy_diversity - all_regional_energy_diversity.min()) / (all_regional_energy_diversity.max() - all_regional_energy_diversity.min())
        outout_df['energy_diversity'] = energy_diversity_normalised

        # normalise the gini coefficient across all regions
        all_regional_gini_coefficient = regional_gini_coefficient['ssp_gini_coefficient']
        gini_coefficient_normalised = (gini_coefficient - all_regional_gini_coefficient.min()) / (all_regional_gini_coefficient.max() - all_regional_gini_coefficient.min())
        outout_df['gini_coefficient'] = gini_coefficient_normalised

        # normalise the electricity price across all regions
        all_regional_electricity_price = regional_electricity_price['electricity_price']
        electricity_price_normalised = (electricity_price - all_regional_electricity_price.min()) / (all_regional_electricity_price.max() - all_regional_electricity_price.min())
        outout_df['electricity_price'] = electricity_price_normalised


    # create the composite resilience score
    outout_df['resilience_score'] = outout_df['final_energy_demand'] + outout_df['energy_diversity'] + outout_df['gini_coefficient'] + outout_df['electricity_price']
    if regional != None:
        outout_df['Region'] = regional
        return outout_df
    
    else:
        # create the composite resilience score
        outout_df.to_csv(OUTPUT_DIR + 'resilience_scores' + str(categories) + '.csv', index=False)


# calculate the robustness score (higher score = more robustness challenges)
def robustness_score(flexibility_scores, shannon_index, carbon_budgets, CDR_2050,
                     categories, regional=None, cross_region_norm=None):
    """
    composite of:
    - energy system flexibility (lower better) weighting 1/4
    - energy system diversity in low carbon options (higher better) weighting 1/4
    - carbon budget share used up by 2030 (lower better) weighting 1/4
    - total CDR by 2050 (lower better) weighting 1/4
    
    """
    output_df = pd.DataFrame()
    output_df['model'] = flexibility_scores['model']
    output_df['scenario'] = flexibility_scores['scenario']
    
    if regional != None:
        
        if cross_region_norm == None:
            CDR_2050 = CDR_2050['total_CDR_gdp']
        
        else:
            CDR_2050 = CDR_2050['total_CDR_gdp']
    else:
        CDR_2050 = CDR_2050['total_CDR']

    flexibility_scores = flexibility_scores['flexibility_score']
    shannon_index = -1 * shannon_index['shannon_index']
    carbon_budgets = carbon_budgets['carbon_budget_share']

    # check if CDR 2050 contains nan values
    if CDR_2050.isnull().values.any():
        CDR_2050 = CDR_2050.fillna(
            CDR_2050.mean())

    if cross_region_norm == None:
    
        # normalise the CDR 2050
        CDR_2050_normalised = (CDR_2050 - CDR_2050.min()) / (CDR_2050.max() - CDR_2050.min())
        output_df['CDR_2050'] = CDR_2050_normalised

        # normalise the flexibility scores
        flexibility_scores_normalised = (flexibility_scores - flexibility_scores.min()) / (flexibility_scores.max() - flexibility_scores.min())
        output_df['flexibility_scores'] = flexibility_scores_normalised

        # normalise the energy diversity
        shannon_index_normalised = (shannon_index - shannon_index.min()) / (shannon_index.max() - shannon_index.min())
        output_df['shannon_index'] = shannon_index_normalised

        # normalise the carbon budget shares
        carbon_budgets_normalised = (carbon_budgets - carbon_budgets.min()) / (carbon_budgets.max() - carbon_budgets.min())
        output_df['carbon_budgets'] = carbon_budgets_normalised

    else:
        # get the regional data
        regional_energy_system_flexibility = pd.read_csv(OUTPUT_DIR + 'flexibility_scores_regional' + str(categories) + '.csv')
        regional_low_carbon_diversity = pd.read_csv(OUTPUT_DIR + 'low_carbon_shannon_diversity_index_regional' + str(categories) + '.csv')
        regional_carbon_budgets = pd.read_csv(OUTPUT_DIR + 'carbon_budget_shares_regional' + str(categories) + '.csv')
        regional_CDR_2050 = pd.read_csv(OUTPUT_DIR + 'total_CDR_regional' + str(categories) + '.csv')
       
        # normalise the CDR 2050
        all_regional_CDR_2050 = regional_CDR_2050['total_CDR_gdp']
        CDR_2050_normalised = (CDR_2050 - all_regional_CDR_2050.min()) / (all_regional_CDR_2050.max() - all_regional_CDR_2050.min())
        # print cdr_2050 in full to see if there are any nan values

        output_df['CDR_2050'] = CDR_2050_normalised
        # normalise the flexibility scores
        all_regional_flexibility_scores = regional_energy_system_flexibility['flexibility_score']
        flexibility_scores_normalised = (flexibility_scores - all_regional_flexibility_scores.min()) / (all_regional_flexibility_scores.max() - all_regional_flexibility_scores.min())
        output_df['flexibility_scores'] = flexibility_scores_normalised

        # normalise the energy diversity
        all_regional_shannon_index = -1 * regional_low_carbon_diversity['shannon_index']
        shannon_index_normalised = (shannon_index - all_regional_shannon_index.min()) / (all_regional_shannon_index.max() - all_regional_shannon_index.min())
        output_df['shannon_index'] = shannon_index_normalised

        # normalise the carbon budget shares
        all_regional_carbon_budgets = regional_carbon_budgets['carbon_budget_share']
        carbon_budgets_normalised = (carbon_budgets - all_regional_carbon_budgets.min()) / (all_regional_carbon_budgets.max() - all_regional_carbon_budgets.min())
        output_df['carbon_budgets'] = carbon_budgets_normalised

    output_df['robustness_score'] = output_df['flexibility_scores'] + output_df['shannon_index'] + output_df['carbon_budgets'] + output_df['CDR_2050']

    if regional != None:
        output_df['Region'] = regional
        return output_df
    else:
        # create the composite robustness score
        output_df.to_csv(OUTPUT_DIR + 'robustness_scores' + str(categories) + '.csv', index=False)


# calculate the fairness score (higher score = more fairness challenges)
def fairness_score(between_region_gini, carbon_budget_fairness, categories):

    output_df = pd.DataFrame()
    output_df['model'] = between_region_gini['model']
    output_df['scenario'] = between_region_gini['scenario']

    # normalise the between region gini coefficient
    between_region_gini = between_region_gini['between_region_gini']
    between_region_gini_normalised = (between_region_gini - between_region_gini.min()) / (between_region_gini.max() - between_region_gini.min())
    output_df['between_region_gini'] = between_region_gini_normalised

    # normalise the carbon budget fairness
    carbon_budget_fairness = carbon_budget_fairness['carbon_budget_fairness']
    carbon_budget_fairness_normalised = (carbon_budget_fairness - carbon_budget_fairness.min()) / (carbon_budget_fairness.max() - carbon_budget_fairness.min())
    output_df['carbon_budget_fairness'] = carbon_budget_fairness_normalised

    # create the composite fairness score
    output_df['fairness_score'] = output_df['between_region_gini'] + output_df['carbon_budget_fairness']
    output_df.to_csv(OUTPUT_DIR + 'fairness_scores' + str(categories) + '.csv', index=False)


# calculate the transformation speed score (higher score = more transformation speed challenges)
def transition_speed_score(transition_speed_metrics, categories, regional=None, cross_region_norm=None):

    output_df = pd.DataFrame()
    output_df['model'] = transition_speed_metrics['model']
    output_df['scenario'] = transition_speed_metrics['scenario']

    final_demand_reduction = -1 * transition_speed_metrics['Final energy per cap reductions']
    electrification = transition_speed_metrics['Share of final energy from electricity']
    share_food_crop = transition_speed_metrics['Share of food demand from crops']

    if cross_region_norm == None: 

        # Normalise the metrics
        final_demand_reduction_normalised = (final_demand_reduction - final_demand_reduction.min()) / (final_demand_reduction.max() - final_demand_reduction.min())
        electrification_normalised = (electrification - electrification.min()) / (electrification.max() - electrification.min())
        share_food_crop_normalised = (share_food_crop - share_food_crop.min()) / (share_food_crop.max() - share_food_crop.min())

    else:
        # get the regional data
        regional_transition_speed_metrics = pd.read_csv(OUTPUT_DIR + 'transition_speed_metrics_regional' + str(categories) + '.csv')

        # normalise the metrics but from the min and max of all regions
        all_regional_final_demand_reduction = -1 * regional_transition_speed_metrics['Final energy per cap reductions']
        all_regional_electrification = regional_transition_speed_metrics['Share of final energy from electricity']
        all_regional_share_food_crop = regional_transition_speed_metrics['Share of food demand from crops']

        final_demand_reduction_normalised = (final_demand_reduction - all_regional_final_demand_reduction.min()) / (all_regional_final_demand_reduction.max() - all_regional_final_demand_reduction.min())
        electrification_normalised = (electrification - all_regional_electrification.min()) / (all_regional_electrification.max() - all_regional_electrification.min())
        share_food_crop_normalised = (share_food_crop - all_regional_share_food_crop.min()) / (all_regional_share_food_crop.max() - all_regional_share_food_crop.min())

    output_df['final_demand_reduction'] = final_demand_reduction_normalised
    output_df['electrification'] = electrification_normalised
    output_df['share_food_crop'] = share_food_crop_normalised

    # create the composite transformation speed score
    output_df['transition_speed_score'] = output_df['final_demand_reduction'] + output_df['electrification'] + output_df['share_food_crop']

    if regional != None:
        output_df['Region'] = regional
        return output_df
    
    else:
        output_df.to_csv(OUTPUT_DIR + 'transition_speed_scores' + str(categories) + '.csv', index=False)


def get_regional_scores(regional_investment_metrics, regional_environment_metrics, 
                        resource_metrics, regional_final_energy_demand, regional_energy_diversity,
                        regional_gini_coefficient, regional_electricity_price,
                        regional_energy_system_flexibility, regional_low_carbon_diversity,
                        regional_carbon_budgets, regional_CDR_2050, 
                        regional_transition_speed_metrics, categories,
                        cross_region_norm=None):

    # output dfs for each dimension 
    econ_output = pd.DataFrame()
    env_output = pd.DataFrame()
    resource_output = pd.DataFrame()
    resilience_output = pd.DataFrame()
    robust_output = pd.DataFrame()
    transition_speed_output = pd.DataFrame()

    for region in R10_CODES:
        
        # economic
        region_investment_scores = regional_investment_metrics[regional_investment_metrics['region'] == region]
        region_econ_scores = economic_score(region_investment_scores, categories, region, cross_region_norm)
        econ_output = pd.concat([econ_output, region_econ_scores], axis=0)
        
        # environment
        region_env_scores = regional_environment_metrics[regional_environment_metrics['region'] == region]
        region_env_scores = environment_score(region_env_scores, categories, region, cross_region_norm)
        env_output = pd.concat([env_output, region_env_scores], axis=0)
        
        # resources
        regional_resource_scores = resource_metrics
        regional_resource_scores['region'] = region
        region_resource_scores = resource_score(NaturalResources.minerals, regional_resource_scores, categories, region)
        resource_output = pd.concat([resource_output, region_resource_scores], axis=0)

        # resilience
        region_final_energy_demand = regional_final_energy_demand[regional_final_energy_demand['region'] == region]
        region_energy_diversity = regional_energy_diversity[regional_energy_diversity['region'] == region]
        region_gini_coefficient = regional_gini_coefficient[regional_gini_coefficient['region'] == region]
        region_electricity_price = regional_electricity_price[regional_electricity_price['region'] == region]
        region_resilience = resilience_score(region_final_energy_demand, region_energy_diversity, region_gini_coefficient, region_electricity_price, 
                                             categories, region, cross_region_norm)
        resilience_output = pd.concat([resilience_output, region_resilience], axis=0)
        
        # robust
        region_energy_system_flexibility = regional_energy_system_flexibility[regional_energy_system_flexibility['region'] == region]
        region_low_carbon_diversity = regional_low_carbon_diversity[regional_low_carbon_diversity['region'] == region]
        region_carbon_budgets = regional_carbon_budgets[regional_carbon_budgets['region'] == region]
        region_CDR_2050 = regional_CDR_2050[regional_CDR_2050['region'] == region]
        region_robustness = robustness_score(region_energy_system_flexibility,
                                      region_low_carbon_diversity,
                                      region_carbon_budgets,
                                      region_CDR_2050, categories, region, cross_region_norm)
        robust_output = pd.concat([robust_output, region_robustness], axis=0)        

        # transition speed
        region_transition_speed = regional_transition_speed_metrics[regional_transition_speed_metrics['region'] == region]
        region_transition_speed = transition_speed_score(region_transition_speed, categories, region, cross_region_norm)
        transition_speed_output = pd.concat([transition_speed_output, region_transition_speed], axis=0)


    if cross_region_norm != None:
        econ_output.to_csv(OUTPUT_DIR + 'economic_scores_regional_cross_region_norm' + str(categories) + '.csv', index=False)
        env_output.to_csv(OUTPUT_DIR + 'environmental_scores_regional_cross_region_norm' + str(categories) + '.csv', index=False)
        resource_output.to_csv(OUTPUT_DIR + 'resource_scores_regional_cross_region_norm' + str(categories) + '.csv', index=False)
        resilience_output.to_csv(OUTPUT_DIR + 'resilience_scores_regional_cross_region_norm' + str(categories) + '.csv', index=False)
        robust_output.to_csv(OUTPUT_DIR + 'robustness_scores_regional_cross_region_norm' + str(categories) + '.csv', index=False)
        transition_speed_output.to_csv(OUTPUT_DIR + 'transition_speed_scores_regional_cross_region_norm' + str(categories) + '.csv', index=False)
    else:
        econ_output.to_csv(OUTPUT_DIR + 'economic_scores_regional' + str(categories) + '.csv', index=False)
        env_output.to_csv(OUTPUT_DIR + 'environmental_scores_regional' + str(categories) + '.csv', index=False)
        resource_output.to_csv(OUTPUT_DIR + 'resource_scores_regional' + str(categories) + '.csv', index=False)
        resilience_output.to_csv(OUTPUT_DIR + 'resilience_scores_regional' + str(categories) + '.csv', index=False)
        robust_output.to_csv(OUTPUT_DIR + 'robustness_scores_regional' + str(categories) + '.csv', index=False)
        transition_speed_output.to_csv(OUTPUT_DIR + 'transition_speed_scores_regional' + str(categories) + '.csv', index=False)
    
    econ_output = econ_output.reset_index()
    env_output = env_output.reset_index()
    resource_output = resource_output.reset_index()
    resilience_output = resilience_output.reset_index()
    robust_output = robust_output.reset_index()
    transition_speed_output = transition_speed_output.reset_index()

    print(econ_output)


    # make a dataframe with the scores from the different dimensions
    data = pd.DataFrame({'model': econ_output['model'],
                        'scenario': econ_output['scenario'],
                        'region': econ_output['Region'], 
                        'economic_score': econ_output['investment_score'],
                        'environmental_score': env_output['environmental_score'],
                        'resource_score': resource_output['resource_score'],
                        'resilience_score': resilience_output['resilience_score'],
                        'robustness_score': robust_output['robustness_score'],
                        'transition_speed_score': transition_speed_output['transition_speed_score']}, index=None)
    
    print(data)

    # calculate the normalised dimension scores
    dimension_scores = pd.DataFrame()
    
    if cross_region_norm == None:
        for region in R10_CODES:

            region_data = data[data['region'] == region]
            region_data['economic_dimension_score'] = (region_data['economic_score'] ) #/ region_data['economic_score'].max() 
            region_data['environmental_dimension_score'] = (region_data['environmental_score'] - region_data['environmental_score'].min()) / (region_data['environmental_score'].max() - region_data['environmental_score'].min())
            region_data['resource_dimension_score'] = (region_data['resource_score'] - region_data['resource_score'].min()) / (region_data['resource_score'].max() - region_data['resource_score'].min())
            region_data['resilience_dimension_score'] = (region_data['resilience_score'] - region_data['resilience_score'].min()) / (region_data['resilience_score'].max() - region_data['resilience_score'].min())
            region_data['robustness_dimension_score'] = (region_data['robustness_score'] - region_data['robustness_score'].min()) / (region_data['robustness_score'].max() - region_data['robustness_score'].min())
            region_data['transition_speed_dimension_score'] = (region_data['transition_speed_score'] - region_data['transition_speed_score'].min()) / (region_data['transition_speed_score'].max() - region_data['transition_speed_score'].min())

            dimension_scores = pd.concat([dimension_scores, region_data], axis=0)
        
        dimension_scores.to_csv(OUTPUT_DIR + 'regional_normalised_dimension_scores' + str(categories) + '.csv', index=False)

    else:
        for region in R10_CODES:

            region_data = data[data['region'] == region]
            region_data['economic_dimension_score'] = (region_data['economic_score'] - data['economic_score'].min()) / (data['economic_score'].max() - data['economic_score'].min())
            region_data['environmental_dimension_score'] = (region_data['environmental_score'] - data['environmental_score'].min()) / (data['environmental_score'].max() - data['environmental_score'].min())
            region_data['resource_dimension_score'] = (region_data['resource_score'] - data['resource_score'].min()) / (data['resource_score'].max() - data['resource_score'].min())
            region_data['resilience_dimension_score'] = (region_data['resilience_score'] - data['resilience_score'].min()) / (data['resilience_score'].max() - data['resilience_score'].min())
            region_data['robustness_dimension_score'] = (region_data['robustness_score'] - data['robustness_score'].min()) / (data['robustness_score'].max() - data['robustness_score'].min())
            region_data['transition_speed_dimension_score'] = (region_data['transition_speed_score'] - data['transition_speed_score'].min()) / (data['transition_speed_score'].max() - data['transition_speed_score'].min())

            dimension_scores = pd.concat([dimension_scores, region_data], axis=0)


        dimension_scores.to_csv(OUTPUT_DIR + 'regional_normalised_dimension_scores_cross_regional_normalisation' + str(categories) + '.csv', index=False)


# select most dissimilar scenarios
def normalise_scores(economic_scores,
                                     environment_scores,
                                     resource_scores,
                                     resilience_scores, 
                                     robustness_scores, 
                                     fairness_scores,
                                     transition_speed_scores, 
                                     categories):
    """
    Select the most dissimilar scenarios based on the scores across the
    different dimensions. The number of scenarios to select is defined in
    the Selection class.
    
    """
    # make a dataframe with the scores from the different dimensions
    data = pd.DataFrame({'model': economic_scores['model'],
                        'scenario': economic_scores['scenario'],
                        'economic_score': economic_scores['investment_score'],
                        'environmental_score': environment_scores['environmental_score'],
                        'resource_score': resource_scores['resource_score'],
                        'resilience_score': resilience_scores['resilience_score'],
                        'robustness_score': robustness_scores['robustness_score'],
                        'fairness_score': fairness_scores['fairness_score'],
                        'transition_speed_score': transition_speed_scores['transition_speed_score']}, index=None)

    # normalise the scores
    data['economic_score'] = (data['economic_score'] ) #/ data['economic_score'].max() 
    data['environmental_score'] = (data['environmental_score'] - data['environmental_score'].min()) / (data['environmental_score'].max() - data['environmental_score'].min())
    data['resource_score'] = (data['resource_score'] - data['resource_score'].min()) / (data['resource_score'].max() - data['resource_score'].min())
    data['resilience_score'] = (data['resilience_score'] - data['resilience_score'].min()) / (data['resilience_score'].max() - data['resilience_score'].min())
    data['robustness_score'] = (data['robustness_score'] - data['robustness_score'].min()) / (data['robustness_score'].max() - data['robustness_score'].min())
    data['fairness_score'] = (data['fairness_score'] - data['fairness_score'].min()) / (data['fairness_score'].max() - data['fairness_score'].min())
    data['transition_speed_score'] = (data['transition_speed_score'] - data['transition_speed_score'].min()) / (data['transition_speed_score'].max() - data['transition_speed_score'].min())
    
    # make the model and scenario the index
    data.set_index(['model', 'scenario'], inplace=True)
    data.to_csv(OUTPUT_DIR + 'normalised_scores' + str(categories) + '.csv', index=True)

    # # Calculate pairwise Euclidean distances
    # dist_matrix = squareform(pdist(data, 'euclidean'))

    # # Generate all possible combinations of four pathways
    # pathway_indices = range(len(data))
    # combinations_of_four = list(combinations(pathway_indices, 4))

    # # Initialise variables to store the maxmium score and the best combination
    # max_score = -np.inf
    # best_combination = None

    # # Loop through all combinations of four pathways
    # for combo in combinations_of_four:
        
    #     # Extract the distances for the current combination
    #     combo_dist_matrix = dist_matrix[np.ix_(combo, combo)]

    #     score = np.sum(combo_dist_matrix)

    #     if score > max_score:
    #         max_score = score
    #         best_combination = combo
        
    # # Sample implementation for selecting 4 most dissimilar pathways (greedy approach)
    # n_select = Selection.number_of_illustrative_scenarios
    # selected_indices = [np.argmax(np.sum(dist_matrix, axis=1))]  # Start with the most dissimilar

    # for _ in range(1, n_select):
    #     # Calculate the minimum distance of remaining pathways to the selected set
    #     min_dist_to_set = np.min(dist_matrix[:, selected_indices], axis=1)
    
    #     # Select the pathway that maximizes the minimum distance to the already selected set
    #     next_index = np.argmax(min_dist_to_set)
    #     selected_indices.append(next_index)

    # # reset the index of the model_scenarios_list
    # # model_scenarios_list = model_scenarios_list.reset_index()
    # best_combination = list(best_combination)
    # # # `selected_indices` now contains the indices of the 4 most different pathways
    # selected_pathways = model_scenarios_list.iloc[best_combination]
    # print(selected_pathways)


# find scenario archetypes (K-means clustering of scenarios)
def find_scenario_archetypes(categories, cluster_number=int):
    """
    Find scenario archetypes using K-means clustering of the scenarios based on the
    scores across the different dimensions.
    
    """
    data = pd.read_csv(OUTPUT_DIR + 'normalised_scores' + str(categories) + '.csv')
    
    # set index
    data.set_index(['model', 'scenario'], inplace=True)

    # Sample implementation for finding 4 scenario archetypes (K-means clustering)
    n_clusters = cluster_number
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    data['cluster'] = kmeans.labels_
    
    data.to_csv(OUTPUT_DIR + 'clustered_scores' + str(categories) + '.csv', index=True)
    
    # Split the DataFrame into a list of DataFrames for each cluster
    clusters = [data[data['cluster'] == i] for i in data['cluster'].unique()]   

    # Generate all combinations, one scenario from each cluster
    combinations = list(product(*[cluster.iterrows() for cluster in clusters]))

    # Calculate dissimilarity for each combination and select the most dissimilar one
    most_dissimilar_combination = max(combinations, key=calculate_total_dissimilarity)

    # # Extract the selected scenarios
    # selected_scenarios = [scenario for scenario in most_dissimilar_combination]
        # Extract the selected scenarios into a new DataFrame
    selected_scenarios_df = pd.DataFrame()
    models = []
    scenario = []
    for index, scenario_row in most_dissimilar_combination:
        
        models.append(index[0])
        scenario.append(index[1])
        # Extract the row (as a DataFrame) and append it to the new DataFrame
        scenario_df = pd.DataFrame([scenario_row])
        selected_scenarios_df = pd.concat([selected_scenarios_df, scenario_df])
    
    selected_scenarios_df['model'] = models
    selected_scenarios_df['scenario'] = scenario

    # Reset index if necessary, to include 'model' and 'scenario' in the columns
    # selected_scenarios_df.reset_index(inplace=True)
    
    # Save the selected scenarios to a CSV file
    selected_scenarios_df.to_csv(OUTPUT_DIR + 'selected_scenarios' + str(categories) + '.csv', index=False)
    
    # # calculate scenario archetype scores as the centroids of each cluster
    cluster_centroids = pd.DataFrame(kmeans.cluster_centers_, columns=data.columns[:-1])
    # add in the cluster number
    cluster_centroids['cluster'] = range(0, n_clusters)
    cluster_centroids.to_csv(OUTPUT_DIR + 'scenario_archetypes' + str(categories) + '.csv', index=False)

    # return the four scenarios most similar to the cluster centroids
    closest_to_centroids = return_centroid_illustrative_scenarios(cluster_centroids, data, cluster_number)
    closest_to_centroids = closest_to_centroids.reset_index()
    closest_to_centroids.to_csv(OUTPUT_DIR + 'closest_to_centroids' + str(categories) + '.csv', index=False)

    
# Function to calculate total dissimilarity for a combination of scenarios
def calculate_total_dissimilarity(combination):
    # Extracting the scenarios' features from the combination
    # Adjusted to handle `(index, row)` tuples
    scenarios_features = [scenario[1].drop('cluster').values for scenario in combination]
    # Calculating pairwise distances and then the total sum of those distances
    total_dissimilarity = sum(pdist(scenarios_features, 'euclidean'))
    return total_dissimilarity


# Function that returns the four scenarios most similar to the cluster centroids
def return_centroid_illustrative_scenarios(cluster_centroids, data, cluster_number=int):
    
    """
    Sub function that returns the four scenarios most similar to the cluster centroids.
    
    Inputs:
    - cluster_centroids: DataFrame with the cluster centroids.
    - data: DataFrame with the scenarios and their clusters
    - cluster_number: Number of clusters
    
    Returns:
    - selected_pathways: DataFrame with the illustrative scenarios clostest to the cluster centroids
    
    """
    selected_indices = []

    for i in range(cluster_number):
        
        # Calculate the Euclidean distance between the cluster centroid and all pathways
        centroid = cluster_centroids.iloc[i].values
        distances = np.linalg.norm(data.values - centroid, axis=1)
        
        # Select the pathway closest to the centroid
        closest_indices = np.argsort(distances)[:1]
        selected_indices.extend(closest_indices)

    # Extract the selected scenarios
    selected_pathways = data.iloc[selected_indices]
    return selected_pathways




if __name__ == "__main__":
    main()