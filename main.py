import numpy as np
import pyam
import pandas as pd
from utils import Data
from itertools import combinations
from resources import NaturalResouces
from scipy.spatial.distance import pdist, squareform

class IndexBuilder:

    investment_metrics = pd.read_csv('outputs/energy_supply_investment_score.csv')
    environment_metrics = pd.read_csv('outputs/environmental_metrics.csv')
    resource_metrics = pd.read_csv('outputs/material_use_ratios.csv')
    
    # import resilience metrics
    final_energy_demand = pd.read_csv('outputs/final_energy_demand.csv')    
    energy_diversity = pd.read_csv('outputs/shannon_diversity_index.csv')   
    gini_coefficient = pd.read_csv('outputs/gini_coefficient.csv')

    # import robustness metrics
    energy_system_flexibility = pd.read_csv('outputs/flexibility_scores.csv')
    carbon_budgets = pd.read_csv('outputs/carbon_budget_shares.csv')

class Selection:

    economic_scores = pd.read_csv('outputs/economic_scores.csv')
    environment_scores = pd.read_csv('outputs/environmental_scores.csv')
    resource_scores = pd.read_csv('outputs/resource_scores.csv')
    resilience_scores = pd.read_csv('outputs/resilience_scores.csv')
    robustness_scores = pd.read_csv('outputs/robustness_scores.csv')
    number_of_illustrative_scenarios = 4

def main() -> None:

    # economic_score(IndexBuilder.investment_metrics)
    # environment_score(IndexBuilder.environment_metrics)
    # resource_score(NaturalResouces.minerals, IndexBuilder.resource_metrics)
    # resilience_score(IndexBuilder.final_energy_demand,
    #                 IndexBuilder.energy_diversity,
    #                 IndexBuilder.gini_coefficient)
    # calculate_robustness_score(IndexBuilder.energy_system_flexibility, 
    #                            IndexBuilder.energy_diversity, 
    #                            IndexBuilder.carbon_budgets)
    select_most_dissimilar_scenarios(Data.model_scenarios)


# calculate the economic score (higher score = more economic challenges)
def economic_score(investment_scores):
    
    output_df = pd.DataFrame()
    output_df['model'] = investment_scores['model']
    output_df['scenario'] = investment_scores['scenario']
    
    # normalise the investment data
    energy_investment = investment_scores['mean_value']
    energy_investment_2050 = investment_scores['mean_value_2050']
    energy_investment_normalised = (energy_investment - energy_investment.min()) / (energy_investment.max() - energy_investment.min())
    energy_investment_2050_normalised = (energy_investment_2050 - energy_investment_2050.min()) / (energy_investment_2050.max() - energy_investment_2050.min())

    output_df['investment_score'] = energy_investment_normalised
    output_df['investment_score_2050'] = energy_investment_2050_normalised
    
    output_df.to_csv('outputs/economic_scores.csv', index=False)


# calculate the environmental score (higher score = more environmental challenges)
def environment_score(environment_metrics):

    output_df = pd.DataFrame()
    output_df['model'] = environment_metrics['model']
    output_df['scenario'] = environment_metrics['scenario']

    # normalise the environmental data
    forest_change_2050 = -1 * IndexBuilder.environment_metrics['forest_change_2050']
    forest_change = -1 * IndexBuilder.environment_metrics['forest_change_2100']
    forest_change_normalised = (forest_change - forest_change.min()) / (forest_change.max() - forest_change.min())
    forest_change_2050_normalised = (forest_change_2050 - forest_change_2050.min()) / (forest_change_2050.max() - forest_change_2050.min())

    # add up the composite environmental score
    beccs_threshold_breached = IndexBuilder.environment_metrics['beccs_threshold_breached']
    output_df['environmental_score'] = forest_change_normalised + beccs_threshold_breached
    output_df['environmental_score_2050'] = forest_change_2050_normalised + beccs_threshold_breached

    output_df.to_csv('outputs/environmental_scores.csv', index=False)


# calculate the resource score (higher score = more resource challenges)
def resource_score(minerals, resource_metrics):
   
    output_df = pd.DataFrame()
    output_df['model'] = resource_metrics['model']
    output_df['scenario'] = resource_metrics['scenario']
   
    # loop through the materials and normalise the scores
    for mineral in minerals:
       
        mineral_score = resource_metrics[mineral]
        mineral_score_normalised = (mineral_score - mineral_score.min()) / (mineral_score.max() - mineral_score.min())
        output_df[mineral] = mineral_score_normalised
        try:
            output_df['total'] += mineral_score_normalised
        except:
            output_df['total'] = mineral_score_normalised
    # create a composite resource score
    output_df['resource_score'] = output_df['total'] / len(minerals)
    output_df.to_csv('outputs/resource_scores.csv', index=False)


# calculate the resilience score (higher score = more resilience challenges)
def resilience_score(final_energy_demand, energy_diversity, gini_coefficient):
    """
    composite of 
    - final energy demand (lower better) weighting 1/3
    - energy diversity (higher better) weighting 1/3
    - gini coefficient (lower better) weighting 1/3
    """
    outout_df = pd.DataFrame()
    outout_df['model'] = final_energy_demand['model']
    outout_df['scenario'] = final_energy_demand['scenario']

    # normalise the final energy demand
    final_energy_demand = final_energy_demand['final_energy_demand']
    final_energy_demand_normalised = (final_energy_demand - final_energy_demand.min()) / (final_energy_demand.max() - final_energy_demand.min())
    outout_df['final_energy_demand'] = final_energy_demand_normalised

    # normalise the energy diversity
    energy_diversity = -1 * energy_diversity['shannon_index']
    energy_diversity_normalised = (energy_diversity - energy_diversity.min()) / (energy_diversity.max() - energy_diversity.min())
    outout_df['energy_diversity'] = energy_diversity_normalised

    # normalise the gini coefficient
    gini_coefficient = gini_coefficient['ssp_gini_coefficient']
    gini_coefficient_normalised = (gini_coefficient - gini_coefficient.min()) / (gini_coefficient.max() - gini_coefficient.min())
    outout_df['gini_coefficient'] = gini_coefficient_normalised

    # create the composite resilience score
    outout_df['resilience_score'] = outout_df['final_energy_demand'] + outout_df['energy_diversity'] + outout_df['gini_coefficient']
    outout_df.to_csv('outputs/resilience_scores.csv', index=False)


# calculate the robustness score (higher score = more robustness challenges)
def calculate_robustness_score(flexibility_scores, shannon_index, carbon_budgets):
    """
    composite of:
    - energy system flexibility (lower better) weighting 1/3
    - energy system diversity (higher better) weighting 1/3
    - carbon budget share used up by 2030 (lower better) weighting 1/3
    
    """
    outout_df = pd.DataFrame()
    outout_df['model'] = flexibility_scores['model']
    outout_df['scenario'] = flexibility_scores['scenario']
    
    # normalise the flexibility scores
    flexibility_scores = flexibility_scores['flexibility_score']
    flexibility_scores_normalised = (flexibility_scores - flexibility_scores.min()) / (flexibility_scores.max() - flexibility_scores.min())
    outout_df['flexibility_scores'] = flexibility_scores_normalised

    # normalise the energy diversity
    shannon_index = -1 * shannon_index['shannon_index']
    shannon_index_normalised = (shannon_index - shannon_index.min()) / (shannon_index.max() - shannon_index.min())
    outout_df['shannon_index'] = shannon_index_normalised

    # normalise the carbon budget shares
    carbon_budgets = carbon_budgets['carbon_budget_share']
    carbon_budgets_normalised = (carbon_budgets - carbon_budgets.min()) / (carbon_budgets.max() - carbon_budgets.min())
    outout_df['carbon_budgets'] = carbon_budgets_normalised

    # create the composite robustness score
    outout_df['robustness_score'] = outout_df['flexibility_scores'] + outout_df['shannon_index'] + outout_df['carbon_budgets']
    outout_df.to_csv('outputs/robustness_scores.csv', index=False)


# select most dissimilar scenarios
def select_most_dissimilar_scenarios(model_scenarios_list):
    """
    Select the most dissimilar scenarios based on the scores across the
    different dimensions. The number of scenarios to select is defined in
    the Selection class.
    
    """
    # make a dataframe with the scores from the different dimensions
    data = pd.DataFrame({'model': Selection.economic_scores['model'],
                        'scenario': Selection.economic_scores['scenario'],
                        'economic_score': Selection.economic_scores['investment_score'],
                        'environmental_score': Selection.environment_scores['environmental_score'],
                        'resource_score': Selection.resource_scores['resource_score'],
                        'resilience_score': Selection.resilience_scores['resilience_score'],
                        'robustness_score': Selection.robustness_scores['robustness_score']}, index=None)

    # normalise the scores
    data['economic_score'] = (data['economic_score'] ) / data['economic_score'].max() 
    data['environmental_score'] = (data['environmental_score'] ) / data['environmental_score'].max()
    data['resource_score'] = (data['resource_score'] ) / data['resource_score'].max()
    data['resilience_score'] = (data['resilience_score'] ) / data['resilience_score'].max()
    data['robustness_score'] = (data['robustness_score'] ) / data['robustness_score'].max()
    
    # make the model and scenario the index
    data.set_index(['model', 'scenario'], inplace=True)

    # Calculate pairwise Euclidean distances
    dist_matrix = squareform(pdist(data, 'euclidean'))

    # Generate all possible combinations of four pathways
    pathway_indices = range(len(data))
    combinations_of_four = list(combinations(pathway_indices, 4))

    # Initialise variables to store the maxmium score and the best combination
    max_score = -np.inf
    best_combination = None

    # Loop through all combinations of four pathways
    for combo in combinations_of_four:

        # Extract the distances for the current combination
        combo_dist_matrix = dist_matrix[np.ix_(combo, combo)]

        score = np.sum(combo_dist_matrix)

        if score > max_score:
            max_score = score
            best_combination = combo
        


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
    # model_scenarios_list = model_scenarios_list.reset_index()
    best_combination = list(best_combination)
    # # `selected_indices` now contains the indices of the 4 most different pathways
    selected_pathways = model_scenarios_list.iloc[best_combination]
    print(selected_pathways)


if __name__ == "__main__":
    main()