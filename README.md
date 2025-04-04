# Integrated Assessment Model Scenario Evaluation and Exploration Tool (BETA)

This tool allows users to explore, evaluate, and select integrated assessment model scenarios. It takes Pyam dataframe inputs and performs analysis across seven dimensions to consider scenarios' relative feasibility and desirability.
The seven dimensions are:
- Economic Feasibility
- Environmental Sustainability
- Resource Use
- Societal Resilience
- Interregional Fairness
- Near-term Scenario Robustness
- Societal Transition Rate

## To run
The package is still under development, however, it can be run to perform analysis across C1 and C2 scenarios. 
1. Ensure you have the [AR6 database file]( https://data.ece.iiasa.ac.at/ar6/) downloaded and saved in the database directory.
2. Ensure that in the constants.py you have the correct directory for your AR6 database file(s)
3. From the Run.py file, run the setup function.
4. Then call the remaining scripts to run the analysis. 
