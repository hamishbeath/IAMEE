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
        "description": "Relative scale of investment and macroeconomic effort associated with the transition pathway.",
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
        "description": "Non-climate environmental pressures that accompany the pathway.",
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
        "description": "Potential stress placed on energy systems and society by the pathway.",
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
        "description": "Distributional outcomes across regions in emissions and mitigation burden.",
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
        "description": "How rapidly societies and energy systems need to change in the pathway.",
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


def load_scores(metadata: dict[tuple[str, str], dict[str, str]]) -> tuple[list[dict], dict[str, list[dict]]]:
    resource_overrides = load_resource_score_overrides()
    global_scores = [decorate_score_row(row, metadata, resource_overrides) for row in read_csv(GLOBAL_SCORES)]
    regional_scores = {
        "within": [decorate_score_row(row, metadata, resource_overrides, regional=True) for row in read_csv(REGIONAL_WITHIN)],
        "across": [decorate_score_row(row, metadata, resource_overrides, regional=True) for row in read_csv(REGIONAL_ACROSS)],
    }
    return global_scores, regional_scores


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
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    OUTPUT_JSON.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=True), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Global scenarios: {payload['metadata']['scenarioCount']}")
    print(f"Regional rows: {payload['metadata']['regionalRowCount']}")
    print(f"Timeseries rows: {payload['metadata']['timeseriesRowCount']}")


if __name__ == "__main__":
    main()
