# SCI Implementation Notes

These notes document the choices made to run the IAMEE framework on the SCI scenario set.

## Scenario Selection

- Database source: SCI database via the existing scenario download utilities.
- Climate categories retained: `GW2`, `GW3`, `GW4`.
- Metadata filter: `Vetting|SCI 2025 = ok`.
- Spatial resolution retained: R10 regions plus `World`.
- R5 data were not used in the framework-ready SCI files.
- Final scenario subset: 95 scenarios after dropping indicators unavailable in SCI.

## Framework-Ready Files

- `data/processed/Framework_pyam['GW2', 'GW3', 'GW4'].csv`
- `data/processed/Framework_scenarios['GW2', 'GW3', 'GW4'].csv`
- `database/SCI_GW2_GW3_GW4_vetted_R10_no_food_price_meta_data.csv`

## SCI Variable Substitutions

The SCI database uses several variable names that differ from the AR6-oriented framework names. These substitutions were applied when creating the framework-ready SCI files:

| Framework variable | SCI variable |
| --- | --- |
| `Carbon Sequestration|CCS|Biomass` | `Carbon Capture|Geological Storage|Biomass` |
| `Carbon Sequestration|Land Use` | `Carbon Removal|Land Use` |
| `Food Demand|Crops` | `Food Availability|Crops [per capita]` |
| `Food Demand|Livestock` | `Food Availability|Livestock [per capita]` |
| `Land Cover|Forest|Natural Forest` | `Land Cover|Forest` |
| `Carbon Sequestration|Direct Air Capture` | `Carbon Removal|Geological Storage|Direct Air Capture` |

Note: `Land Cover|Forest|Primary` was checked but did not provide the required SCI coverage for the selected subset, so `Land Cover|Forest` was used for the framework's natural forest indicator.

## Dropped Indicators

Two indicators were dropped for the SCI application because the required variables were not available with sufficient coverage:

- Societal Transition Rate: dietary change indicator using `Food Demand|Crops` and `Food Demand|Livestock`.
- Societal Resilience: electricity price indicator using `Price|Secondary Energy|Electricity`.

The affected dimension scores are therefore calculated from the remaining indicators:

- Societal Transition Rate: final energy per capita reductions and electrification.
- Societal Resilience: final energy demand, energy diversity, and SSP/gini component.

## Scoring Adjustments

- Constant normalised indicators are assigned a score contribution of `0`. This is used for SCI SSP/gini normalisation where all retained scenarios have `Ssp_family = 2`, so the indicator has no scenario-discriminating information.
- Shannon diversity calculations clip negative energy components to `0` before calculating shares.
- Shannon terms with zero share are treated as `0 * log(0) = 0`.
- Resource efficiency uses the existing default wind material intensity fallback where scenario-specific onshore/offshore wind split data are unavailable for the SCI subset.

## Output Scope

The clustering and illustrative-scenario selection step was not run for the SCI application.

Generated outputs are under `data/outputs/` with the suffix `['GW2', 'GW3', 'GW4'].csv`, including:

- Metric files for economic, environmental, resource, robustness, fairness, resilience, and transition speed dimensions.
- Global score files for all dimensions.
- Regional score files for all available regional dimensions.
- Combined normalised global scores.
- Regional normalised dimension scores with within-region and cross-region normalisation.
