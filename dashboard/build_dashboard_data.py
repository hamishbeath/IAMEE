#!/usr/bin/env python3
"""Build a compact JSON bundle for the IAMEE SCI dashboard."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = Path(__file__).resolve().parent
DATA_DIR = DASHBOARD_DIR / "data"
CATEGORY_SUFFIX = "['GW2', 'GW3', 'GW4']"

GLOBAL_SCORES = ROOT / "data" / "outputs" / f"normalised_scores{CATEGORY_SUFFIX}.csv"
RESOURCE_SCORES = ROOT / "data" / "outputs" / f"resource_scores{CATEGORY_SUFFIX}.csv"
REGIONAL_WITHIN = ROOT / "data" / "outputs" / f"regional_normalised_dimension_scores{CATEGORY_SUFFIX}.csv"
REGIONAL_ACROSS = ROOT / "data" / "outputs" / f"regional_normalised_dimension_scores_cross_regional_normalisation{CATEGORY_SUFFIX}.csv"
ENERGY_INVESTMENT = ROOT / "data" / "outputs" / f"energy_supply_investment_score{CATEGORY_SUFFIX}.csv"
ENVIRONMENTAL_METRICS = ROOT / "data" / "outputs" / f"environmental_metrics{CATEGORY_SUFFIX}.csv"
MATERIAL_USE_RATIOS = ROOT / "data" / "outputs" / f"material_use_ratios{CATEGORY_SUFFIX}.csv"
FINAL_ENERGY_DEMAND = ROOT / "data" / "outputs" / f"final_energy_demand{CATEGORY_SUFFIX}.csv"
ENERGY_DIVERSITY = ROOT / "data" / "outputs" / f"shannon_diversity_index{CATEGORY_SUFFIX}.csv"
GINI_COEFFICIENT = ROOT / "data" / "outputs" / f"gini_coefficient{CATEGORY_SUFFIX}.csv"
FLEXIBILITY_SCORES = ROOT / "data" / "outputs" / f"flexibility_scores{CATEGORY_SUFFIX}.csv"
LOW_CARBON_DIVERSITY = ROOT / "data" / "outputs" / f"low_carbon_shannon_diversity_index{CATEGORY_SUFFIX}.csv"
CARBON_BUDGET_SHARES = ROOT / "data" / "outputs" / f"carbon_budget_shares{CATEGORY_SUFFIX}.csv"
TOTAL_CDR = ROOT / "data" / "outputs" / f"total_CDR{CATEGORY_SUFFIX}.csv"
BETWEEN_REGION_GINI = ROOT / "data" / "outputs" / f"between_region_gini{CATEGORY_SUFFIX}.csv"
CARBON_BUDGET_FAIRNESS = ROOT / "data" / "outputs" / f"carbon_budget_fairness{CATEGORY_SUFFIX}.csv"
TRANSITION_SPEED_METRICS = ROOT / "data" / "outputs" / f"transition_speed_metrics{CATEGORY_SUFFIX}.csv"
METADATA = ROOT / "database" / "SCI_GW2_GW3_GW4_vetted_R10_no_food_price_meta_data.csv"
FRAMEWORK_DATA = ROOT / "data" / "processed" / f"Framework_pyam{CATEGORY_SUFFIX}.csv"
MODEL_FAMILIES = ROOT / "data" / "inputs" / "model_family.csv"
OUTPUT_JSON = DATA_DIR / "dashboard-data.json"

CATEGORY_COLORS = {
    "GW2": "#a1d99b",
    "GW3": "#4460fa",
    "GW4": "#7666DA",
}

MODEL_FAMILY_COLORS = {
    "IMAGE": "#0072B2",
    "MESSAGE": "#E69F00",
    "POLES": "#009E73",
    "REMIND": "#D55E00",
    "WITCH": "#CC79A7",
}

FALLBACK_MODEL_FAMILY_COLORS = ["#56B4E9", "#8F6BB1", "#999933", "#882255"]

DIMENSIONS_GLOBAL = [
    {
        "id": "economic",
        "score": "economic_score",
        "regionalScore": "economic_dimension_score",
        "label": "Economic Feasibility",
        "shortLabel": "Economic",
        "description": "Mean modelled ratio of energy-supply investment as a share of GDP, compared with recent historical investment shares.",
        "indicators": [
            {
                "name": "Energy supply investment",
                "description": "Compares transition investment needs against regional or global GDP context.",
            }
        ],
    },
    {
        "id": "environmental",
        "score": "environmental_score",
        "regionalScore": "environmental_dimension_score",
        "label": "Environmental Sustainability",
        "shortLabel": "Environment",
        "description": "Non-climate environmental pressures across mitigation scenarios.",
        "indicators": [
            {
                "name": "Bioenergy sustainability threshold",
                "description": "Tracks whether primary energy from biomass breaches a sustainability threshold.",
                "regionalNote": "Regional thresholds are calculated by regional land area.",
            },
            {
                "name": "Natural forest change",
                "description": "Captures losses or gains in natural forest cover.",
            },
        ],
    },
    {
        "id": "resource",
        "score": "resource_score",
        "regionalScore": "resource_dimension_score",
        "label": "Resource Use",
        "shortLabel": "Resources",
        "description": "Material and mineral demand pressures linked to low-carbon technology deployment.",
        "indicators": [
            {
                "name": "Critical material use",
                "description": "Assesses renewable technology material requirements against availability assumptions.",
            }
        ],
        "globalNote": "Global-only dimension in this SCI dashboard; R10 resource-use data are not available.",
    },
    {
        "id": "resilience",
        "score": "resilience_score",
        "regionalScore": "resilience_dimension_score",
        "label": "Societal Resilience",
        "shortLabel": "Resilience",
        "description": "Mitigation pathways may make societies less resilient to shocks.",
        "indicators": [
            {
                "name": "Final energy demand",
                "description": "Compares final energy demand against GDP context.",
                "regionalNote": "Regional analysis uses final energy demand over GDP so values are comparable between regions.",
            },
            {
                "name": "Energy diversity",
                "description": "Uses primary energy diversity as a proxy for dependence on narrow supply mixes.",
            },
            {
                "name": "SSP and gini component",
                "description": "Represents socioeconomic inequality context where it has scenario-discriminating information.",
                "regionalNote": "Regional values calculate inequality for countries within each R10 region.",
            },
        ],
        "droppedIndicators": [
            {
                "name": "Electricity price",
                "description": "Dropped for this SCI run because Price|Secondary Energy|Electricity did not have sufficient coverage.",
            }
        ],
    },
    {
        "id": "robustness",
        "score": "robustness_score",
        "regionalScore": "robustness_dimension_score",
        "label": "Near-term Scenario Robustness",
        "shortLabel": "Robustness",
        "description": "Dependence on choices or conditions that can make near-term pathways harder to rely on.",
        "indicators": [
            {
                "name": "Flexibility requirements",
                "description": "Captures energy-system flexibility challenges.",
            },
            {
                "name": "Low-carbon supply diversity",
                "description": "Assesses concentration across low-carbon energy supply options.",
            },
            {
                "name": "Carbon budget pressure",
                "description": "Compares regional emissions pathways against budget shares.",
                "regionalNote": "Regional carbon budgets use country-level historical emissions and GDP data with a polluter-pays approach.",
            },
            {
                "name": "Carbon dioxide removal by 2050",
                "description": "Captures reliance on early CDR deployment.",
                "regionalNote": "Regional analysis compares total CDR over regional land area.",
            },
        ],
    },
    {
        "id": "fairness",
        "score": "fairness_score",
        "regionalScore": None,
        "label": "Interregional Fairness",
        "shortLabel": "Fairness",
        "description": "Distributional outcomes across regions.",
        "indicators": [
            {
                "name": "Between-region gini",
                "description": "Measures inequality across regional pathway outcomes.",
            },
            {
                "name": "Carbon budget fairness",
                "description": "Compares northern and southern regional shares of cumulative emissions.",
            },
        ],
        "globalOnly": True,
        "globalNote": "Global-only dimension; it is excluded from R10 regional radar views.",
    },
    {
        "id": "transition_speed",
        "score": "transition_speed_score",
        "regionalScore": "transition_speed_dimension_score",
        "label": "Societal Transition Rate",
        "shortLabel": "Transition",
        "description": "How rapidly societies and energy systems need to change.",
        "indicators": [
            {
                "name": "Final energy per capita reduction",
                "description": "Tracks the maximum decadal reduction in per-capita final energy.",
            },
            {
                "name": "Electrification rate",
                "description": "Tracks the maximum decadal increase in the share of final energy from electricity.",
            },
        ],
        "droppedIndicators": [
            {
                "name": "Dietary change",
                "description": "Dropped for this SCI run because the food-demand variables were unavailable with sufficient coverage.",
            }
        ],
    },
]

REGION_LABELS = {
    "R10AFRICA": "Africa",
    "R10CHINA+": "China+",
    "R10EUROPE": "Europe",
    "R10INDIA+": "India+",
    "R10LATIN_AM": "Latin America",
    "R10MIDDLE_EAST": "Middle East",
    "R10NORTH_AM": "North America",
    "R10PAC_OECD": "Pacific OECD",
    "R10REF_ECON": "Reforming Economies",
    "R10REST_ASIA": "Rest of Asia",
    "World": "World",
}

PRIMARY_ENERGY_VARIABLES = [
    {
        "id": "Primary Energy|Biomass",
        "label": "Biomass",
        "color": "#008000",
    },
    {
        "id": "Primary Energy|Non-Biomass Renewables",
        "label": "Non-biomass renewables",
        "color": "#7ee37f",
    },
    {
        "id": "Primary Energy|Fossil|w/o CCS",
        "label": "Fossil without CCS",
        "color": "#b0b0b0",
    },
    {
        "id": "Primary Energy|Fossil|w/ CCS",
        "label": "Fossil with CCS",
        "color": "#d9d9d9",
    },
    {
        "id": "Primary Energy|Nuclear",
        "label": "Nuclear",
        "color": "#0000ff",
    },
]

CDR_VARIABLES = [
    {
        "id": "Carbon Sequestration|CCS|Biomass",
        "label": "BECCS",
        "color": "#ff6b00",
    },
    {
        "id": "Carbon Sequestration|Land Use",
        "label": "Land use",
        "color": "#ffc400",
    },
    {
        "id": "Carbon Sequestration|Direct Air Capture",
        "label": "Direct air capture",
        "color": "#ff0000",
    },
]

MATERIAL_DEMAND_AXIS = "Material demand ratios"
MATERIAL_DEMAND_DESCRIPTION = (
    "These indicators show the ratio of historical use of each material in solar and wind technologies "
    "compared to the maximum five-year mean projected use up to 2050. Lower values indicate lower material "
    "pressure; higher values indicate larger projected use relative to historical solar and wind use."
)

RAW_INDICATOR_SPECS = [
    {
        "id": "economic",
        "source": ENERGY_INVESTMENT,
        "indicators": [
            {
                "id": "investment_ratio_2020_2100",
                "column": "mean_value",
                "label": "Mean energy-supply investment ratio (2020-2100)",
                "unit": "ratio",
                "axisLabel": "",
                "description": "This is the mean modelled ratio of energy-supply investment as a share of GDP compared with recent historical values across 2020-2100. Lower values are closer to recent historical investment shares; higher values indicate a larger investment effort relative to GDP.",
            },
            {
                "id": "investment_ratio_2020_2050",
                "column": "mean_value_2050",
                "label": "Mean energy-supply investment ratio (2020-2050)",
                "unit": "ratio",
                "axisLabel": "",
                "description": "This is the mean modelled ratio of energy-supply investment as a share of GDP compared with recent historical values across 2020-2050. Lower values indicate less near-term deviation from recent historical investment shares; higher values indicate a larger near-term investment effort relative to GDP.",
            },
        ],
    },
    {
        "id": "environmental",
        "source": ENVIRONMENTAL_METRICS,
        "indicators": [
            {
                "id": "forest_change_2050",
                "column": "forest_change_2050",
                "label": "Natural forest change by 2050",
                "unit": "fractional change",
                "axisLabel": "Forest cover change",
                "description": "This is the absolute percentage-point change in natural forest cover as a share of total land cover; for example, 0.08 means 8% more of the earth's surface is forest. Lower values indicate less forest gain or greater forest pressure by 2050; higher values indicate larger natural forest expansion.",
            },
            {
                "id": "forest_change_2100",
                "column": "forest_change_2100",
                "label": "Natural forest change by 2100",
                "unit": "fractional change",
                "axisLabel": "Forest cover change",
                "description": "This is the absolute percentage-point change in natural forest cover as a share of total land cover; for example, 0.08 means 8% more of the earth's surface is forest. Lower values indicate less forest gain or greater long-term forest pressure by 2100; higher values indicate larger natural forest expansion.",
            },
            {
                "id": "bioenergy_threshold_breached",
                "column": "bioenergy_threshold_breached",
                "label": "Bioenergy sustainability threshold breached",
                "unit": "0/1 flag",
                "axisLabel": "Bioenergy sustainability threshold",
                "visual": "binaryFlag",
                "description": "A value of 0 means the bioenergy threshold is not breached; a value of 1 means the scenario exceeds the threshold. Threshold based on Creutzig et al. (2015).",
            },
        ],
    },
    {
        "id": "resource",
        "source": MATERIAL_USE_RATIOS,
        "indicators": [
            {
                "id": "material_nd",
                "column": "Nd",
                "label": "Neodymium demand ratio",
                "unit": "ratio",
                "axisLabel": MATERIAL_DEMAND_AXIS,
                "noteLabel": MATERIAL_DEMAND_AXIS,
                "description": MATERIAL_DEMAND_DESCRIPTION,
            },
            {
                "id": "material_dy",
                "column": "Dy",
                "label": "Dysprosium demand ratio",
                "unit": "ratio",
                "axisLabel": MATERIAL_DEMAND_AXIS,
                "noteLabel": MATERIAL_DEMAND_AXIS,
                "description": MATERIAL_DEMAND_DESCRIPTION,
            },
            {
                "id": "material_ni",
                "column": "Ni",
                "label": "Nickel demand ratio",
                "unit": "ratio",
                "axisLabel": MATERIAL_DEMAND_AXIS,
                "noteLabel": MATERIAL_DEMAND_AXIS,
                "description": MATERIAL_DEMAND_DESCRIPTION,
            },
            {
                "id": "material_mn",
                "column": "Mn",
                "label": "Manganese demand ratio",
                "unit": "ratio",
                "axisLabel": MATERIAL_DEMAND_AXIS,
                "noteLabel": MATERIAL_DEMAND_AXIS,
                "description": MATERIAL_DEMAND_DESCRIPTION,
            },
            {
                "id": "material_ag",
                "column": "Ag",
                "label": "Silver demand ratio",
                "unit": "ratio",
                "axisLabel": MATERIAL_DEMAND_AXIS,
                "noteLabel": MATERIAL_DEMAND_AXIS,
                "description": MATERIAL_DEMAND_DESCRIPTION,
            },
            {
                "id": "material_cd",
                "column": "Cd",
                "label": "Cadmium demand ratio",
                "unit": "ratio",
                "axisLabel": MATERIAL_DEMAND_AXIS,
                "noteLabel": MATERIAL_DEMAND_AXIS,
                "description": MATERIAL_DEMAND_DESCRIPTION,
            },
            {
                "id": "material_te",
                "column": "Te",
                "label": "Tellurium demand ratio",
                "unit": "ratio",
                "axisLabel": MATERIAL_DEMAND_AXIS,
                "noteLabel": MATERIAL_DEMAND_AXIS,
                "description": MATERIAL_DEMAND_DESCRIPTION,
            },
            {
                "id": "material_se",
                "column": "Se",
                "label": "Selenium demand ratio",
                "unit": "ratio",
                "axisLabel": MATERIAL_DEMAND_AXIS,
                "noteLabel": MATERIAL_DEMAND_AXIS,
                "description": MATERIAL_DEMAND_DESCRIPTION,
            },
            {
                "id": "material_in",
                "column": "In",
                "label": "Indium demand ratio",
                "unit": "ratio",
                "axisLabel": MATERIAL_DEMAND_AXIS,
                "noteLabel": MATERIAL_DEMAND_AXIS,
                "description": MATERIAL_DEMAND_DESCRIPTION,
            },
        ],
    },
    {
        "id": "resilience",
        "source": FINAL_ENERGY_DEMAND,
        "indicators": [
            {
                "id": "final_energy_demand",
                "column": "final_energy_demand",
                "label": "Cumulative final energy demand (2020-2100)",
                "unit": "EJ",
                "axisLabel": "Final energy demand",
                "description": "Cumulative final-energy demand across 2020-2100. Lower values indicate lower cumulative final-energy demand; higher values indicate greater energy-service demand to satisfy through the transition.",
            },
        ],
    },
    {
        "id": "resilience",
        "source": ENERGY_DIVERSITY,
        "indicators": [
            {
                "id": "energy_diversity",
                "column": "shannon_index",
                "label": "Primary energy diversity",
                "unit": "Shannon index",
                "axisLabel": "Energy system diversity",
                "description": "Lower values indicate a more concentrated energy mix; higher values indicate a more diverse primary energy mix.",
            },
        ],
    },
    {
        "id": "resilience",
        "source": GINI_COEFFICIENT,
        "indicators": [
            {
                "id": "ssp_gini_coefficient",
                "column": "ssp_gini_coefficient",
                "label": "SSP gini coefficient",
                "unit": "Gini index",
                "axisLabel": "Between country inequality",
                "description": "Lower values indicate lower socioeconomic inequality assumptions; higher values indicate greater inequality context. Because all scenarios in this subset are SSP2, this indicator is uniform across the scenario set.",
            },
        ],
    },
    {
        "id": "robustness",
        "source": FLEXIBILITY_SCORES,
        "indicators": [
            {
                "id": "flexibility_score",
                "column": "flexibility_score",
                "label": "Energy-system flexibility score",
                "unit": "score",
                "axisLabel": "Energy system flexibility",
                "description": "Lower values indicate lower flexibility stress; higher values indicate greater flexibility requirements.",
            },
        ],
    },
    {
        "id": "robustness",
        "source": LOW_CARBON_DIVERSITY,
        "indicators": [
            {
                "id": "low_carbon_diversity",
                "column": "shannon_index",
                "label": "Low-carbon supply diversity",
                "unit": "Shannon index",
                "axisLabel": "Low-carbon energy system diversity",
                "description": "Lower values indicate a more concentrated low-carbon supply mix; higher values indicate a more diverse low-carbon supply mix.",
            },
        ],
    },
    {
        "id": "robustness",
        "source": CARBON_BUDGET_SHARES,
        "indicators": [
            {
                "id": "carbon_budget_share",
                "column": "carbon_budget_share",
                "label": "Carbon budget share used by 2030",
                "unit": "share",
                "axisLabel": "Near-term Carbon budget share used",
                "description": "Lower values indicate less near-term carbon-budget pressure; higher values indicate a larger share of the budget is used by 2030.",
            },
        ],
    },
    {
        "id": "robustness",
        "source": TOTAL_CDR,
        "indicators": [
            {
                "id": "total_cdr",
                "column": "total_CDR",
                "label": "Cumulative CDR to 2050",
                "unit": "Mt CO2",
                "axisLabel": "Carbon dioxide removal",
                "description": "Lower values indicate less reliance on early carbon dioxide removal; higher values indicate greater cumulative CDR deployment by 2050.",
            },
        ],
    },
    {
        "id": "fairness",
        "source": BETWEEN_REGION_GINI,
        "indicators": [
            {
                "id": "between_region_gini",
                "column": "between_region_gini",
                "label": "Between-region gini coefficient",
                "unit": "Gini index",
                "axisLabel": "Between R10 region inequality",
                "description": "Lower values indicate more even regional outcomes; higher values indicate greater inequality across regions.",
            },
        ],
    },
    {
        "id": "fairness",
        "source": CARBON_BUDGET_FAIRNESS,
        "indicators": [
            {
                "id": "carbon_budget_fairness",
                "column": "carbon_budget_fairness",
                "label": "Carbon budget fairness ratio",
                "unit": "ratio",
                "axisLabel": "Mitigation burden fairness",
                "description": "Lower values indicate a lower north-south carbon-budget imbalance; higher values indicate greater distributional imbalance.",
            },
        ],
    },
    {
        "id": "transition_speed",
        "source": TRANSITION_SPEED_METRICS,
        "indicators": [
            {
                "id": "final_energy_per_capita_reduction",
                "column": "Final energy per cap reductions",
                "label": "Maximum decadal final-energy per-capita reduction",
                "plotLabel": "Maximum decadal final-energy<br>per-capita reduction",
                "unit": "fraction per decade",
                "axisLabel": "Energy demand reduction",
                "description": "Lower values indicate smaller or slower reductions in per-capita final energy; higher-magnitude negative values indicate faster demand reduction.",
            },
            {
                "id": "electrification_rate",
                "column": "Share of final energy from electricity",
                "label": "Maximum decadal electrification increase",
                "unit": "share",
                "axisLabel": "Electrification Rate",
                "description": "Lower values indicate slower electrification; higher values indicate faster growth in electricity's share of final energy.",
            },
        ],
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def framework_years() -> list[str]:
    with FRAMEWORK_DATA.open(newline="", encoding="utf-8-sig") as handle:
        fieldnames = next(csv.reader(handle))
    years = sorted(int(column) for column in fieldnames if column.isdigit())
    if not years:
        return []
    return [str(year) for year in range(years[0], years[-1] + 1, 5)]


def interpolate_values(row: dict[str, str], target_years: list[str]) -> list[float | None]:
    numeric_years = [int(year) for year in target_years]
    known = [
        (year, value)
        for year in numeric_years
        if (value := parse_float(row.get(str(year)))) is not None
    ]
    values = []
    for year in numeric_years:
        direct = parse_float(row.get(str(year)))
        if direct is not None:
            values.append(direct)
            continue

        lower = next(((known_year, value) for known_year, value in reversed(known) if known_year < year), None)
        upper = next(((known_year, value) for known_year, value in known if known_year > year), None)
        if lower and upper:
            lower_year, lower_value = lower
            upper_year, upper_value = upper
            fraction = (year - lower_year) / (upper_year - lower_year)
            values.append(lower_value + fraction * (upper_value - lower_value))
        else:
            values.append(None)
    return values


def scenario_key(model: str, scenario: str) -> str:
    return f"{model}|||{scenario}"


def infer_model_family(model: str) -> str:
    if model.startswith("MESSAGE"):
        return "MESSAGE"
    if model.startswith("REMIND"):
        return "REMIND"
    if model.startswith("WITCH"):
        return "WITCH"
    if model.startswith("IMAGE"):
        return "IMAGE"
    if model.startswith("POLES"):
        return "POLES"
    if model.startswith("AIM"):
        return "AIM"
    if model.startswith("GCAM"):
        return "GCAM"
    return model.split()[0]


def load_metadata() -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, str]]:
    metadata_rows = read_csv(METADATA)
    family_rows = read_csv(MODEL_FAMILIES)
    family_map = {row["model"]: row["model_family"] for row in family_rows}

    metadata: dict[tuple[str, str], dict[str, str]] = {}
    for row in metadata_rows:
        model = row["Model"]
        scenario = row["Scenario"]
        metadata[(model, scenario)] = {
            "category": row.get("Category", ""),
            "categoryName": row.get("Category_name", ""),
            "categorySubset": row.get("Category_subset", ""),
            "vetting": row.get("Vetting", ""),
            "projectStudy": row.get("Project_study", ""),
            "sspFamily": row.get("Ssp_family", ""),
            "version": row.get("version", ""),
            "modelFamily": family_map.get(model, infer_model_family(model)),
        }
    return metadata, family_map


def load_resource_score_overrides() -> dict[tuple[str, str], float]:
    overrides = {}
    if not RESOURCE_SCORES.exists():
        return overrides
    for row in read_csv(RESOURCE_SCORES):
        score = parse_float(row.get("resource_score"))
        if score is not None:
            overrides[(row["model"], row["scenario"])] = score
    return overrides


def decorate_score_row(
    row: dict[str, str],
    metadata: dict[tuple[str, str], dict[str, str]],
    resource_overrides: dict[tuple[str, str], float],
    regional: bool = False,
) -> dict:
    model = row["model"]
    scenario = row["scenario"]
    meta = metadata.get((model, scenario), {})
    item = {
        "id": scenario_key(model, scenario),
        "model": model,
        "scenario": scenario,
        "category": meta.get("category", ""),
        "categoryName": meta.get("categoryName", ""),
        "categorySubset": meta.get("categorySubset", ""),
        "modelFamily": meta.get("modelFamily", infer_model_family(model)),
        "sspFamily": meta.get("sspFamily", ""),
        "scores": {},
    }
    if regional:
        item["region"] = row["region"]
        for dimension in DIMENSIONS_GLOBAL:
            regional_score = dimension.get("regionalScore")
            if dimension["id"] == "resource" and (model, scenario) in resource_overrides:
                item["scores"][dimension["id"]] = resource_overrides[(model, scenario)]
            elif regional_score and regional_score in row:
                item["scores"][dimension["id"]] = parse_float(row.get(regional_score))
    else:
        for dimension in DIMENSIONS_GLOBAL:
            if dimension["id"] == "resource" and (model, scenario) in resource_overrides:
                item["scores"][dimension["id"]] = resource_overrides[(model, scenario)]
            else:
                item["scores"][dimension["id"]] = parse_float(row.get(dimension["score"]))
    return item


def decorate_scenario_row(model: str, scenario: str, metadata: dict[tuple[str, str], dict[str, str]]) -> dict:
    meta = metadata.get((model, scenario), {})
    return {
        "id": scenario_key(model, scenario),
        "model": model,
        "scenario": scenario,
        "category": meta.get("category", ""),
        "categoryName": meta.get("categoryName", ""),
        "categorySubset": meta.get("categorySubset", ""),
        "modelFamily": meta.get("modelFamily", infer_model_family(model)),
        "sspFamily": meta.get("sspFamily", ""),
    }


def load_scores(metadata: dict[tuple[str, str], dict[str, str]]) -> tuple[list[dict], dict[str, list[dict]]]:
    resource_overrides = load_resource_score_overrides()
    global_scores = [decorate_score_row(row, metadata, resource_overrides) for row in read_csv(GLOBAL_SCORES)]
    regional_scores = {
        "within": [decorate_score_row(row, metadata, resource_overrides, regional=True) for row in read_csv(REGIONAL_WITHIN)],
        "across": [decorate_score_row(row, metadata, resource_overrides, regional=True) for row in read_csv(REGIONAL_ACROSS)],
    }
    return global_scores, regional_scores


def load_raw_indicator_dimensions(
    metadata: dict[tuple[str, str], dict[str, str]],
    selected_scenarios: set[tuple[str, str]],
) -> list[dict]:
    dimensions = {
        dimension["id"]: {
            "id": dimension["id"],
            "label": dimension["label"],
            "shortLabel": dimension["shortLabel"],
            "description": dimension["description"],
            "indicators": [],
        }
        for dimension in DIMENSIONS_GLOBAL
    }
    indicator_lookup: dict[str, dict] = {}

    for spec in RAW_INDICATOR_SPECS:
        dimension = dimensions[spec["id"]]
        source_rows = read_csv(spec["source"])
        for indicator in spec["indicators"]:
            indicator_item = {
                "id": indicator["id"],
                "label": indicator["label"],
                "plotLabel": indicator.get("plotLabel", indicator["label"]),
                "noteLabel": indicator.get("noteLabel", indicator["label"]),
                "unit": indicator["unit"],
                "axisLabel": indicator.get("axisLabel", indicator["unit"]),
                "visual": indicator.get("visual", "boxplot"),
                "description": indicator["description"],
                "values": [],
            }
            dimension["indicators"].append(indicator_item)
            indicator_lookup[indicator["id"]] = indicator_item

        for row in source_rows:
            model = row.get("model")
            scenario = row.get("scenario")
            if not model or not scenario or (model, scenario) not in selected_scenarios:
                continue
            scenario_item = decorate_scenario_row(model, scenario, metadata)
            for indicator in spec["indicators"]:
                value = parse_float(row.get(indicator["column"]))
                if value is None:
                    continue
                indicator_lookup[indicator["id"]]["values"].append(
                    {
                        **scenario_item,
                        "value": value,
                    }
                )

    return [dimension for dimension in dimensions.values() if dimension["indicators"]]


def load_normalised_score_table(
    metadata: dict[tuple[str, str], dict[str, str]],
    resource_overrides: dict[tuple[str, str], float],
) -> list[dict]:
    rows = []
    for row in read_csv(GLOBAL_SCORES):
        model = row["model"]
        scenario = row["scenario"]
        item = decorate_scenario_row(model, scenario, metadata)
        item["scores"] = {}
        for dimension in DIMENSIONS_GLOBAL:
            if dimension["id"] == "resource" and (model, scenario) in resource_overrides:
                item["scores"][dimension["id"]] = resource_overrides[(model, scenario)]
            else:
                item["scores"][dimension["id"]] = parse_float(row.get(dimension["score"]))
        rows.append(item)
    return rows


def load_timeseries(metadata: dict[tuple[str, str], dict[str, str]], years: list[str]) -> list[dict]:
    variable_lookup = {item["id"]: item for item in PRIMARY_ENERGY_VARIABLES + CDR_VARIABLES}
    rows = []
    for row in read_csv(FRAMEWORK_DATA):
        variable = row["Variable"]
        if variable not in variable_lookup:
            continue
        model = row["Model"]
        scenario = row["Scenario"]
        meta = metadata.get((model, scenario))
        if not meta:
            continue
        rows.append(
            {
                "id": scenario_key(model, scenario),
                "model": model,
                "scenario": scenario,
                "region": row["Region"],
                "category": meta.get("category", ""),
                "modelFamily": meta.get("modelFamily", infer_model_family(model)),
                "variable": variable,
                "unit": row.get("Unit", ""),
                "values": interpolate_values(row, years),
            }
        )
    return rows


def build_payload() -> dict:
    metadata, _ = load_metadata()
    global_scores, regional_scores = load_scores(metadata)
    years = framework_years()
    timeseries = load_timeseries(metadata, years)
    selected_scenarios = {(row["model"], row["scenario"]) for row in global_scores}
    resource_overrides = load_resource_score_overrides()
    raw_indicator_dimensions = load_raw_indicator_dimensions(metadata, selected_scenarios)
    normalised_score_table = load_normalised_score_table(metadata, resource_overrides)

    categories = sorted({row["category"] for row in global_scores if row["category"]})
    model_families = sorted({row["modelFamily"] for row in global_scores if row["modelFamily"]})
    regions = sorted({row["region"] for row in regional_scores["within"]})
    model_family_groups = [
        {
            "id": family,
            "label": family,
            "color": MODEL_FAMILY_COLORS.get(
                family,
                FALLBACK_MODEL_FAMILY_COLORS[index % len(FALLBACK_MODEL_FAMILY_COLORS)],
            ),
        }
        for index, family in enumerate(model_families)
    ]

    return {
        "metadata": {
            "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": "IAMEE SCI GW2/GW3/GW4 vetted R10 run",
            "scenarioCount": len(global_scores),
            "regionalRowCount": len(regional_scores["within"]),
            "timeseriesRowCount": len(timeseries),
            "rawIndicatorCount": sum(len(dimension["indicators"]) for dimension in raw_indicator_dimensions),
            "normalisedScoreRowCount": len(normalised_score_table),
            "categories": [
                {"id": category, "label": category, "color": CATEGORY_COLORS.get(category, "#777777")}
                for category in categories
            ],
            "categoryColors": CATEGORY_COLORS,
            "modelFamilies": model_families,
            "modelFamilyGroups": model_family_groups,
            "modelFamilyColors": {item["id"]: item["color"] for item in model_family_groups},
            "regions": [{"id": region, "label": REGION_LABELS.get(region, region)} for region in regions],
            "years": years,
            "dimensionsGlobal": DIMENSIONS_GLOBAL,
            "dimensionsRegional": [dimension for dimension in DIMENSIONS_GLOBAL if not dimension.get("globalOnly")],
            "normalisations": [
                {
                    "id": "within",
                    "label": "Within-region normalisation",
                    "description": "Each R10 region is normalised against its own scenario distribution.",
                },
                {
                    "id": "across",
                    "label": "Across-region normalisation",
                    "description": "All R10 region-scenario rows share one normalisation range.",
                },
            ],
            "variables": {
                "primaryEnergy": PRIMARY_ENERGY_VARIABLES,
                "cdr": CDR_VARIABLES,
            },
            "implementationNotes": {
                "droppedIndicators": [
                    "Dietary change in Societal Transition Rate",
                    "Electricity price in Societal Resilience",
                ],
                "resourceScoresSource": str(RESOURCE_SCORES.relative_to(ROOT)),
                "regionalResourceNote": "The resource dimension is displayed as a global score in regional views.",
                "scenarioSelection": "GW2, GW3, and GW4 scenarios with SCI vetting ok, using R10 regions plus World.",
                "timeseriesInterpolation": "Primary-energy and CDR time series are linearly interpolated to a 5-year grid before charting.",
            },
        },
        "globalScores": global_scores,
        "regionalScores": regional_scores,
        "timeseries": timeseries,
        "dataTab": {
            "dimensions": raw_indicator_dimensions,
            "normalisedScores": normalised_score_table,
        },
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    OUTPUT_JSON.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=True), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Global scenarios: {payload['metadata']['scenarioCount']}")
    print(f"Regional rows: {payload['metadata']['regionalRowCount']}")
    print(f"Timeseries rows: {payload['metadata']['timeseriesRowCount']}")
    print(f"Raw indicators: {payload['metadata']['rawIndicatorCount']}")


if __name__ == "__main__":
    main()
