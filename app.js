const plotConfig = {
  displayModeBar: false,
  responsive: true,
  scrollZoom: false,
  doubleClick: false,
};

let DATA = null;
let state = null;

const $ = (selector) => document.querySelector(selector);
const fallbackGroupColors = ["#0072b2", "#e69f00", "#009e73", "#d55e00", "#cc79a7", "#56b4e9", "#8f6bb1"];

function numberOrZero(value) {
  return Number.isFinite(value) ? value : 0;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function median(values) {
  const clean = values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  if (!clean.length) return null;
  const middle = Math.floor(clean.length / 2);
  if (clean.length % 2) return clean[middle];
  return (clean[middle - 1] + clean[middle]) / 2;
}

function mean(values) {
  const clean = values.filter((value) => Number.isFinite(value));
  if (!clean.length) return null;
  return clean.reduce((total, value) => total + value, 0) / clean.length;
}

function formatScore(value) {
  return Number.isFinite(value) ? value.toFixed(2) : "n/a";
}

function formatRawValue(value) {
  if (!Number.isFinite(value)) return "n/a";
  const abs = Math.abs(value);
  if (abs !== 0 && (abs >= 10000 || abs < 0.01)) return value.toExponential(2);
  if (abs >= 100) return value.toFixed(1);
  if (abs >= 10) return value.toFixed(2);
  return value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}

function wrapLabel(label, maxChars) {
  if (String(label).includes("<br>")) return String(label);
  const words = String(label).split(" ");
  const lines = [];
  let line = "";
  words.forEach((word) => {
    if (!line) {
      line = word;
    } else if ((line + " " + word).length <= maxChars) {
      line += ` ${word}`;
    } else {
      lines.push(line);
      line = word;
    }
  });
  if (line) lines.push(line);
  return lines.join("<br>");
}

function regionLabel(region) {
  const item = DATA.metadata.regions.find((entry) => entry.id === region);
  return item ? item.label : region;
}

function modelFamilyGroups() {
  if (DATA.metadata.modelFamilyGroups) return DATA.metadata.modelFamilyGroups;
  return DATA.metadata.modelFamilies.map((family, index) => ({
    id: family,
    label: family,
    color: fallbackGroupColors[index % fallbackGroupColors.length],
  }));
}

function groupDefinitions(groupBy = "category") {
  return groupBy === "modelFamily" ? modelFamilyGroups() : DATA.metadata.categories;
}

function groupValue(row, groupBy = "category") {
  return groupBy === "modelFamily" ? row.modelFamily : row.category;
}

function groupColor(groupBy, value) {
  const item = groupDefinitions(groupBy).find((entry) => entry.id === value);
  return item ? item.color : "#777777";
}

function rowsByGroup(rows, groupBy = "category") {
  return groupDefinitions(groupBy).map((group) => ({
    group,
    rows: rows.filter((row) => groupValue(row, groupBy) === group.id),
  }));
}

function activeRegionalRows() {
  return DATA.regionalScores[state.regionalNormalisation];
}

function activeTradeoffDimensions() {
  return state.tradeoffRegion === "World"
    ? DATA.metadata.dimensionsGlobal
    : DATA.metadata.dimensionsRegional;
}

function rowMatches(row, categorySet, familySet) {
  return categorySet.has(row.category) && familySet.has(row.modelFamily);
}

function getScore(row, dimension) {
  const value = row.scores[dimension.id];
  return Number.isFinite(value) ? value : null;
}

function thresholdFor(dimensionId) {
  return Number.isFinite(state.thresholds[dimensionId]) ? state.thresholds[dimensionId] : 1;
}

function tradeoffRowsBeforeLimits() {
  const rows = state.tradeoffRegion === "World"
    ? DATA.globalScores
    : DATA.regionalScores[state.tradeoffNormalisation].filter((row) => row.region === state.tradeoffRegion);
  return rows.filter((row) => rowMatches(row, state.tradeoffCategories, state.tradeoffFamilies));
}

function tradeoffRowsAfterLimits() {
  const dimensions = activeTradeoffDimensions();
  return tradeoffRowsBeforeLimits().filter((row) =>
    dimensions.every((dimension) => {
      const score = getScore(row, dimension);
      return !Number.isFinite(score) || score <= thresholdFor(dimension.id);
    })
  );
}

function createRadarTraces(rows, dimensions, options = {}) {
  const theta = dimensions.map((dimension) => dimension.shortLabel);
  const closedTheta = [...theta, theta[0]];
  const traces = [];
  const showScenarioPoints = options.showScenarios !== false;
  const showScenarioLines = options.showScenarioLines !== false;
  const scenarioOpacity = options.scenarioOpacity ?? 0.42;
  const scenarioLineOpacity = options.scenarioLineOpacity ?? 0.12;
  const scenarioPointOpacity = options.scenarioPointOpacity ?? (showScenarioLines ? scenarioLineOpacity : scenarioOpacity);
  const summaryStatistic = options.summaryStatistic === "mean" ? "mean" : "median";
  const summaryFunction = summaryStatistic === "mean" ? mean : median;
  const groupBy = options.groupBy || "category";
  const minSummaryGroupSize = options.minSummaryGroupSize || 1;

  if (showScenarioPoints && showScenarioLines) {
    rows.forEach((row) => {
      const r = dimensions.map((dimension) => numberOrZero(getScore(row, dimension)));
      traces.push({
        type: "scatterpolar",
        mode: "lines",
        r: [...r, r[0]],
        theta: closedTheta,
        line: { color: groupColor(groupBy, groupValue(row, groupBy)), width: options.scenarioLineWidth || 0.8 },
        opacity: scenarioLineOpacity,
        hoverinfo: "skip",
        showlegend: false,
      });
    });
  }

  if (showScenarioPoints) {
    rowsByGroup(rows, groupBy).forEach(({ group, rows: groupRows }) => {
      if (!groupRows.length) return;
      const r = [];
      const pointTheta = [];
      const text = [];
      groupRows.forEach((row) => {
        dimensions.forEach((dimension) => {
          const score = getScore(row, dimension);
          r.push(numberOrZero(score));
          pointTheta.push(dimension.shortLabel);
          text.push(
            `${row.model}<br>${row.scenario}<br>Temperature: ${row.category}<br>Model family: ${row.modelFamily}<br>${dimension.label}: ${formatScore(score)}`
          );
        });
      });
      traces.push({
        type: "scatterpolar",
        mode: "markers",
        r,
        theta: pointTheta,
        marker: {
          color: group.color,
          size: options.pointSize || 6,
          opacity: scenarioPointOpacity,
          line: { color: "#ffffff", width: 0.45 },
        },
        name: `${group.label} scenarios (${groupRows.length})`,
        hoverinfo: "text",
        text,
      });
    });
  }

  rowsByGroup(rows, groupBy).forEach(({ group, rows: groupRows }) => {
    if (!groupRows.length) return;
    if (groupRows.length < minSummaryGroupSize) return;
    const r = dimensions.map((dimension) => summaryFunction(groupRows.map((row) => getScore(row, dimension))));
    traces.push({
      type: "scatterpolar",
      mode: "lines+markers",
      r: [...r.map(numberOrZero), numberOrZero(r[0])],
      theta: closedTheta,
      line: { color: group.color, width: options.medianWidth || 3 },
      marker: { color: group.color, size: options.markerSize || 6 },
      fill: options.fillMedians ? "toself" : "none",
      fillcolor: group.color,
      opacity: 0.95,
      name: `${group.label} ${summaryStatistic} (${groupRows.length})`,
      hoverinfo: "text",
      text: closedTheta.map((label, index) => {
        const value = index === dimensions.length ? r[0] : r[index];
        return `${group.label} ${summaryStatistic}<br>${groupRows.length} scenarios<br>${label}: ${formatScore(value)}`;
      }),
    });
  });

  return traces;
}

function radarLayout(title = "", options = {}) {
  return {
    title: title ? { text: title, font: { size: 14 } } : undefined,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    dragmode: false,
    margin: { t: title ? 48 : 24, r: 64, b: options.showLegend === false ? 28 : 58, l: 64 },
    showlegend: options.showLegend !== false,
    legend: { orientation: "h", y: -0.08, x: 0, font: { size: 11 } },
    polar: {
      bgcolor: "rgba(0,0,0,0)",
      radialaxis: {
        visible: true,
        range: [0, 1],
        fixedrange: true,
        tickvals: [0, 0.25, 0.5, 0.75, 1],
        tickfont: { size: 10, color: "#5d6776" },
        gridcolor: "#d6ddd9",
        linecolor: "#a9b5af",
      },
      angularaxis: {
        gridcolor: "#d6ddd9",
        linecolor: "#a9b5af",
        fixedrange: true,
        tickfont: { size: 14, color: "#1c2430" },
      },
    },
    font: { family: "Inter, system-ui, sans-serif", color: "#1c2430" },
  };
}

function renderEmptyPlot(id, message) {
  Plotly.newPlot(
    id,
    [],
    {
      annotations: [
        {
          text: message,
          x: 0.5,
          y: 0.5,
          xref: "paper",
          yref: "paper",
          showarrow: false,
          font: { color: "#5d6776", size: 14 },
        },
      ],
      xaxis: { visible: false },
      yaxis: { visible: false },
      dragmode: false,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
    },
    plotConfig
  );
}

function renderRadar(id, rows, dimensions, options = {}) {
  if (!rows.length) {
    renderEmptyPlot(id, "No scenarios match the current filters");
    return;
  }
  Plotly.newPlot(id, createRadarTraces(rows, dimensions, options), radarLayout(options.title || "", options), plotConfig);
}

function setDimensionInfo(dimensionId) {
  const dimension = DATA.metadata.dimensionsGlobal.find((item) => item.id === dimensionId) || DATA.metadata.dimensionsGlobal[0];
  document.querySelectorAll(".dimension-button").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.dimension === dimension.id);
  });
  $("#dimensionInfo").dataset.dimension = dimension.id;

  const indicators = dimension.indicators
    .map((indicator) => `<li title="${indicator.description}"><strong>${indicator.name}</strong><span>${indicator.description}</span></li>`)
    .join("");
  const dropped = (dimension.droppedIndicators || [])
    .map((indicator) => `<li class="dropped" title="${indicator.description}"><strong>${indicator.name} dropped</strong><span>${indicator.description}</span></li>`)
    .join("");
  const regionalNotes = dimension.indicators.filter((indicator) => indicator.regionalNote);
  const regionalFootnotes = regionalNotes.length
    ? `<div class="regional-footnotes" aria-label="Regional calculation notes">${regionalNotes
        .map((indicator) => `<p><strong>${indicator.name}:</strong> ${indicator.regionalNote}</p>`)
        .join("")}</div>`
    : "";
  const globalNote = dimension.globalNote
    ? `<div class="regional-footnotes" aria-label="Global-only note"><p><strong>Global-only dimension:</strong> ${dimension.globalNote}</p></div>`
    : "";

  $("#dimensionInfo").innerHTML = `
    <h3>${dimension.label}</h3>
    <p>${dimension.description}</p>
    <ul class="indicator-list">${indicators}${dropped}</ul>
    ${globalNote}
    ${regionalFootnotes}
  `;
}

function renderAboutSummarySwitch() {
  document.querySelectorAll("#aboutSummarySwitch [data-summary-stat]").forEach((button) => {
    const active = button.dataset.summaryStat === state.aboutSummaryStatistic;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
    button.onclick = () => {
      state.aboutSummaryStatistic = button.dataset.summaryStat;
      renderAboutSummarySwitch();
      renderAboutRadar();
    };
  });
}

function renderAboutRadar() {
  renderRadar("aboutRadar", DATA.globalScores, DATA.metadata.dimensionsGlobal, {
    summaryStatistic: state.aboutSummaryStatistic,
    scenarioOpacity: 0.68,
    scenarioLineOpacity: 0.32,
    pointSize: 5.8,
    medianWidth: 2.2,
    markerSize: 5,
    fillMedians: false,
    showLegend: false,
  });
}

function renderAbout() {
  $("#aboutScenarioCount").textContent = `${DATA.metadata.scenarioCount} scenarios | R10 plus World`;
  $("#aboutCategoryLegend").innerHTML = DATA.metadata.categories
    .map(
      (category) =>
        `<span class="category-pill"><span class="category-dot" style="--dot:${category.color}"></span>${category.label}</span>`
    )
    .join("");
  const buttons = $("#dimensionButtons");
  buttons.innerHTML = "";
  DATA.metadata.dimensionsGlobal.forEach((dimension) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "dimension-button";
    button.dataset.dimension = dimension.id;
    button.title = dimension.description;
    const droppedCount = (dimension.droppedIndicators || []).length;
    button.innerHTML = `
      <strong>${dimension.label}</strong>
      <span>${dimension.indicators.length} active indicator${dimension.indicators.length === 1 ? "" : "s"}${droppedCount ? `, ${droppedCount} dropped` : ""}</span>
    `;
    button.addEventListener("focus", () => setDimensionInfo(dimension.id));
    button.addEventListener("click", () => setDimensionInfo(dimension.id));
    buttons.appendChild(button);
  });
  setDimensionInfo(DATA.metadata.dimensionsGlobal[0].id);
  renderAboutSummarySwitch();
  renderAboutRadar();
}

function renderCheckboxGroup(container, items, selectedSet, onChange) {
  container.innerHTML = "";
  items.forEach((item) => {
    const label = document.createElement("label");
    label.className = "chip-check";
    label.title = item.title || item.label;

    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = selectedSet.has(item.id);
    input.addEventListener("change", () => {
      if (input.checked) {
        selectedSet.add(item.id);
      } else if (selectedSet.size > 1) {
        selectedSet.delete(item.id);
      } else {
        input.checked = true;
        return;
      }
      onChange();
    });

    const text = document.createElement("span");
    text.textContent = item.label;
    if (item.color) {
      text.style.borderBottom = `3px solid ${item.color}`;
    }

    label.append(input, text);
    container.appendChild(label);
  });
}

function renderTradeoffGroupBySwitch() {
  document.querySelectorAll("#tradeoffGroupBy [data-group-by]").forEach((button) => {
    const active = button.dataset.groupBy === state.tradeoffGroupBy;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
    button.onclick = () => {
      state.tradeoffGroupBy = button.dataset.groupBy;
      renderTradeoffGroupBySwitch();
      renderTradeoffPlots();
    };
  });
}

function populateSelect(select, items, selectedValue, onChange) {
  select.innerHTML = "";
  items.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.label;
    if (item.id === selectedValue) option.selected = true;
    select.appendChild(option);
  });
  select.onchange = () => onChange(select.value);
}

function renderTradeoffControls() {
  populateSelect(
    $("#tradeoffRegion"),
    [{ id: "World", label: "World" }, ...DATA.metadata.regions],
    state.tradeoffRegion,
    (value) => {
      state.tradeoffRegion = value;
      renderTradeoffs();
    }
  );

  populateSelect($("#tradeoffNormalisation"), DATA.metadata.normalisations, state.tradeoffNormalisation, (value) => {
    state.tradeoffNormalisation = value;
    renderTradeoffs();
  });
  $("#tradeoffNormalisation").disabled = state.tradeoffRegion === "World";

  renderTradeoffGroupBySwitch();
  renderCheckboxGroup($("#tradeoffCategories"), DATA.metadata.categories, state.tradeoffCategories, renderTradeoffs);
  renderCheckboxGroup(
    $("#tradeoffFamilies"),
    modelFamilyGroups(),
    state.tradeoffFamilies,
    renderTradeoffs
  );
}

function renderThresholdControls() {
  const container = $("#thresholdControls");
  container.innerHTML = "";
  activeTradeoffDimensions().forEach((dimension) => {
    const wrapper = document.createElement("label");
    wrapper.className = "threshold-control";
    const value = thresholdFor(dimension.id);
    wrapper.innerHTML = `
      <span class="threshold-label">
        ${dimension.shortLabel}
        <span class="threshold-value">${value.toFixed(2)}</span>
      </span>
      <input type="range" min="0" max="1" step="0.01" value="${value}" aria-label="${dimension.label} challenge limit" />
    `;
    const input = wrapper.querySelector("input");
    const valueNode = wrapper.querySelector(".threshold-value");
    input.addEventListener("input", () => {
      state.thresholds[dimension.id] = Number(input.value);
      valueNode.textContent = Number(input.value).toFixed(2);
      renderTradeoffPlots();
    });
    container.appendChild(wrapper);
  });
}

function renderTradeoffs() {
  renderTradeoffControls();
  renderThresholdControls();
  renderTradeoffPlots();
}

function renderTradeoffPlots() {
  const before = tradeoffRowsBeforeLimits();
  const after = tradeoffRowsAfterLimits();
  const dimensions = activeTradeoffDimensions();
  const scope = state.tradeoffRegion === "World" ? "World" : regionLabel(state.tradeoffRegion);
  const groupLabel = state.tradeoffGroupBy === "modelFamily" ? "model family" : "temperature category";
  const sparseGroups = rowsByGroup(after, state.tradeoffGroupBy)
    .filter(({ rows }) => rows.length > 0 && rows.length < 10)
    .map(({ group, rows }) => `${group.label} (${rows.length})`);
  const sparseNote = sparseGroups.length
    ? ` Median lines hidden for groups with fewer than 10 scenarios: ${sparseGroups.join(", ")}.`
    : "";
  $("#tradeoffSummary").textContent = `${after.length} of ${before.length} scenarios shown for ${scope}; medians by ${groupLabel}.${sparseNote}`;
  renderRadar("tradeoffRadar", after, dimensions, {
    groupBy: state.tradeoffGroupBy,
    scenarioOpacity: 0.24,
    scenarioLineOpacity: 0.24,
    medianWidth: 1.8,
    markerSize: 5.2,
    fillMedians: false,
    minSummaryGroupSize: 10,
  });
  renderStackPlot("energyStack", "primaryEnergy", after, state.tradeoffRegion, "#energyUnit", before);
  renderStackPlot("cdrStack", "cdr", after, state.tradeoffRegion, "#cdrUnit", before);
}

function timeseriesRowsFor(group, selectedRows, region) {
  const selectedIds = new Set(selectedRows.map((row) => row.id));
  const variables = DATA.metadata.variables[group].map((item) => item.id);
  return DATA.timeseries.filter(
    (row) => selectedIds.has(row.id) && row.region === region && variables.includes(row.variable)
  );
}

function stackedMedianSeries(group, selectedRows, region) {
  const rows = timeseriesRowsFor(group, selectedRows, region);
  const years = DATA.metadata.years;
  const variables = DATA.metadata.variables[group];
  const series = variables.map((variable) => {
    const variableRows = rows.filter((row) => row.variable === variable.id);
    const y = years.map((_, yearIndex) => {
      const values = variableRows.map((row) => row.values[yearIndex]).filter((value) => Number.isFinite(value));
      const value = median(values);
      return Number.isFinite(value) ? Math.max(0, value) : 0;
    });
    return { variable, y };
  });
  return { rows, years, series };
}

function stackedMax(series) {
  if (!series.length) return 0;
  return Math.max(
    ...DATA.metadata.years.map((_, yearIndex) =>
      series.reduce((total, item) => total + numberOrZero(item.y[yearIndex]), 0)
    )
  );
}

function variableLegendLabel(variable) {
  const labels = {
    "Primary Energy|Non-Biomass Renewables": "Non-biomass renew.",
    "Primary Energy|Fossil|w/o CCS": "Fossil w/o CCS",
    "Primary Energy|Fossil|w/ CCS": "Fossil w/ CCS",
    "Carbon Sequestration|Direct Air Capture": "DAC",
  };
  return labels[variable.id] || variable.label;
}

function renderStackPlot(id, group, selectedRows, region, unitSelector, axisRows = selectedRows) {
  const { rows, years, series } = stackedMedianSeries(group, selectedRows, region);
  if (!rows.length) {
    $(unitSelector).textContent = "";
    renderEmptyPlot(id, "No timeseries data match the current filters");
    return;
  }

  const axisSeries = stackedMedianSeries(group, axisRows, region).series;
  const axisMax = Math.max(stackedMax(axisSeries), stackedMax(series));
  const yAxis = {
    gridcolor: "#d6ddd9",
    rangemode: "tozero",
    fixedrange: true,
    title: rows.find((row) => row.unit)?.unit || "",
  };
  if (axisMax > 0) {
    yAxis.range = [0, axisMax * 1.04];
  }

  const traces = series.map(({ variable, y }) => ({
    type: "scatter",
    mode: "lines",
    name: variableLegendLabel(variable),
    x: years,
    y,
    stackgroup: "one",
    line: { color: variable.color, width: 1.4 },
    hovertemplate: `${variable.label}<br>%{x}: %{y:.2f}<extra></extra>`,
  }));

  const firstUnit = rows.find((row) => row.unit)?.unit || "";
  $(unitSelector).textContent = "";

  Plotly.newPlot(
    id,
    traces,
    {
      height: 430,
      margin: { t: 12, r: 18, b: 78, l: 54 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      hovermode: "x unified",
      showlegend: true,
      legend: {
        orientation: "h",
        y: -0.24,
        x: 0,
        font: { size: 9 },
        itemsizing: "constant",
        itemwidth: 30,
      },
      dragmode: false,
      xaxis: { gridcolor: "#d6ddd9", zeroline: false, fixedrange: true },
      yaxis: yAxis,
      font: { family: "Inter, system-ui, sans-serif", color: "#1c2430" },
    },
    plotConfig
  );
}

function dataDimensions() {
  return DATA.dataTab?.dimensions || [];
}

function selectedDataDimension() {
  return dataDimensions().find((dimension) => dimension.id === state.dataDimension) || dataDimensions()[0];
}

function stableJitter(seed) {
  let hash = 0;
  for (let index = 0; index < seed.length; index += 1) {
    hash = (hash * 31 + seed.charCodeAt(index)) % 1000003;
  }
  return ((hash % 1000) / 1000 - 0.5) * 0.36;
}

function indicatorAxisGroups(indicators) {
  const groups = [];
  const lookup = new Map();
  indicators.forEach((indicator) => {
    const key = indicator.axisLabel ?? indicator.unit ?? indicator.id;
    if (!lookup.has(key)) {
      lookup.set(key, {
        id: (key || indicator.unit || indicator.id).replace(/[^a-z0-9]+/gi, "-").replace(/^-|-$/g, "").toLowerCase() || "axis",
        label: key,
        unit: indicator.unit || "not specified",
        indicators: [],
      });
      groups.push(lookup.get(key));
    }
    lookup.get(key).indicators.push(indicator);
  });
  return groups;
}

function renderBinaryFlagPlot(plotId, indicators, showLegend) {
  const indicator = indicators[0];
  const groups = rowsByGroup(indicator.values, state.dataGroupBy).filter(({ rows }) => rows.length);
  const yLabels = groups.map(({ group }) => group.label);
  const noCounts = groups.map(({ rows }) => rows.filter((row) => row.value < 0.5).length);
  const yesCounts = groups.map(({ rows }) => rows.filter((row) => row.value >= 0.5).length);
  const totals = groups.map(({ rows }) => rows.length);
  const noText = noCounts.map((count, index) => `${count} (${Math.round((count / totals[index]) * 100)}%)`);
  const yesText = yesCounts.map((count, index) => `${count} (${Math.round((count / totals[index]) * 100)}%)`);
  const commonTrace = {
    type: "bar",
    orientation: "h",
    y: yLabels,
    textposition: "inside",
    insidetextanchor: "middle",
    hoverinfo: "text",
  };

  Plotly.newPlot(
    plotId,
    [
      {
        ...commonTrace,
        name: "No",
        x: noCounts,
        text: noText,
        marker: { color: "#d6ddd9", line: { color: "#ffffff", width: 1 } },
        hovertext: groups.map(({ group }, index) => `${group.label}<br>No: ${noText[index]} scenarios`),
      },
      {
        ...commonTrace,
        name: "Yes",
        x: yesCounts,
        text: yesText,
        marker: { color: "rgba(68, 96, 250, 0.56)", line: { color: "#ffffff", width: 1 } },
        hovertext: groups.map(({ group }, index) => `${group.label}<br>Yes: ${yesText[index]} scenarios`),
      },
    ],
    {
      height: Math.max(250, groups.length * 46 + 132),
      margin: { t: showLegend ? 46 : 10, r: 28, b: 48, l: 150 },
      barmode: "stack",
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      dragmode: false,
      showlegend: showLegend,
      legend: {
        orientation: "h",
        traceorder: "normal",
        y: 1.12,
        x: 0,
        yanchor: "bottom",
        font: { size: 11 },
      },
      xaxis: {
        title: "Number of scenarios",
        gridcolor: "#d6ddd9",
        rangemode: "tozero",
        fixedrange: true,
      },
      yaxis: {
        automargin: true,
        fixedrange: true,
      },
      font: { family: "Inter, system-ui, sans-serif", color: "#1c2430" },
    },
    plotConfig
  );
}

function renderDataGroupBySwitch() {
  document.querySelectorAll("#dataGroupBy [data-data-group-by]").forEach((button) => {
    const active = button.dataset.dataGroupBy === state.dataGroupBy;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
    button.onclick = () => {
      state.dataGroupBy = button.dataset.dataGroupBy;
      renderDataGroupBySwitch();
      renderDataIndicatorPlot();
    };
  });
}

function renderDataControls() {
  const dimensions = dataDimensions();
  if (!dimensions.length) return;
  if (!dimensions.some((dimension) => dimension.id === state.dataDimension)) {
    state.dataDimension = dimensions[0].id;
  }
  populateSelect($("#dataDimension"), dimensions, state.dataDimension, (value) => {
    state.dataDimension = value;
    renderDataIndicatorPlot();
  });
  renderDataGroupBySwitch();
}

function renderDataAxisPlot(plotId, indicators, unit, showLegend) {
  if (indicators.length === 1 && indicators[0].visual === "binaryFlag") {
    renderBinaryFlagPlot(plotId, indicators, showLegend);
    return;
  }

  const narrowPlot = window.matchMedia("(max-width: 640px)").matches;
  const tickValues = indicators.map((_, index) => indicators.length - index - 1);
  const tickText = indicators.map((indicator) => wrapLabel(indicator.plotLabel || indicator.label, narrowPlot ? 20 : 34));
  const traces = indicators.map((indicator, index) => {
    const yValue = indicators.length - index - 1;
    return {
      type: "box",
      orientation: "h",
      x: indicator.values.map((row) => row.value),
      y: indicator.values.map(() => yValue),
      name: indicator.label,
      boxpoints: false,
      fillcolor: "rgba(214, 221, 217, 0.52)",
      line: { color: "#75827a", width: 1.2 },
      marker: { color: "#75827a" },
      width: 0.46,
      hoverinfo: "skip",
      showlegend: false,
    };
  });

  rowsByGroup(DATA.globalScores, state.dataGroupBy).forEach(({ group }) => {
    const x = [];
    const y = [];
    const text = [];
    indicators.forEach((indicator, index) => {
      const yValue = indicators.length - index - 1;
      indicator.values
        .filter((row) => groupValue(row, state.dataGroupBy) === group.id)
        .forEach((row) => {
          x.push(row.value);
          y.push(yValue + stableJitter(`${row.id}|${indicator.id}`));
          text.push(
            `<b>${escapeHtml(row.model)}</b><br>${escapeHtml(row.scenario)}<br>` +
              `${escapeHtml(indicator.label)}: ${formatRawValue(row.value)} ${escapeHtml(indicator.unit || "")}<br>` +
              `Temperature: ${escapeHtml(row.category || "n/a")}<br>Model family: ${escapeHtml(row.modelFamily || "n/a")}`
          );
        });
    });
    if (!x.length) return;
    traces.push({
      type: "scatter",
      mode: "markers",
      x,
      y,
      name: `${group.label} scenarios`,
      marker: {
        color: group.color,
        size: narrowPlot ? 6 : 7,
        opacity: 0.62,
        line: { color: "#ffffff", width: 0.45 },
      },
      hoverinfo: "text",
      text,
      cliponaxis: false,
      showlegend: showLegend,
    });
  });

  Plotly.newPlot(
    plotId,
    traces,
    {
      height: Math.max(narrowPlot ? 320 : 250, indicators.length * (narrowPlot ? 86 : 62) + (showLegend ? 190 : 135)),
      margin: narrowPlot
        ? { t: showLegend ? 46 : 10, r: 12, b: 54, l: 150 }
        : { t: showLegend ? 46 : 10, r: 28, b: 54, l: 235 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      dragmode: false,
      hovermode: "closest",
      showlegend: showLegend,
      legend: {
        orientation: "h",
        y: 1.12,
        x: 0,
        yanchor: "bottom",
        font: { size: 11 },
        itemsizing: "constant",
      },
      xaxis: {
        title: unit === "not specified" ? "Raw indicator value" : `Raw indicator value (${unit})`,
        gridcolor: "#d6ddd9",
        zerolinecolor: "#a9b5af",
        fixedrange: true,
      },
      yaxis: {
        tickmode: "array",
        tickvals: tickValues,
        ticktext: tickText,
        range: [-0.75, indicators.length - 0.25],
        fixedrange: true,
        gridcolor: "#eef2ef",
        automargin: true,
      },
      font: { family: "Inter, system-ui, sans-serif", color: "#1c2430" },
    },
    plotConfig
  );
}

function renderDataIndicatorPlot() {
  const dimension = selectedDataDimension();
  if (!dimension) {
    renderEmptyPlot("dataIndicatorPlot", "No raw indicator data are available");
    $("#dataIndicatorNotes").innerHTML = "";
    return;
  }

  const indicators = dimension.indicators.filter((indicator) => indicator.values?.length);
  $("#dataDimensionTitle").textContent = `${dimension.label} Raw Indicators`;
  $("#dataDimensionSummary").textContent = dimension.description;
  $("#dataIndicatorCount").textContent = `${indicators.length} indicator${indicators.length === 1 ? "" : "s"}`;

  if (!indicators.length) {
    renderEmptyPlot("dataIndicatorPlot", "No raw indicator values are available for this dimension");
    $("#dataIndicatorNotes").innerHTML = "";
    return;
  }

  const plotContainer = $("#dataIndicatorPlot");
  Plotly.purge(plotContainer);
  plotContainer.innerHTML = "";
  indicatorAxisGroups(indicators).forEach((axisGroup, index) => {
    const plotId = `data-axis-${dimension.id}-${axisGroup.id}-${index}`;
    const isBinaryFlag = axisGroup.indicators.length === 1 && axisGroup.indicators[0].visual === "binaryFlag";
    const heading = axisGroup.label || "";
    const section = document.createElement("section");
    section.className = "data-axis-section";
    section.innerHTML = `
      ${
        heading
          ? `<div class="axis-heading">
              <strong>${escapeHtml(heading)}</strong>
              <span>${axisGroup.indicators.length} indicator${axisGroup.indicators.length === 1 ? "" : "s"}</span>
            </div>`
          : ""
      }
      <div id="${plotId}" class="plot data-axis-plot"></div>
    `;
    plotContainer.appendChild(section);
    renderDataAxisPlot(plotId, axisGroup.indicators, axisGroup.unit, index === 0 || isBinaryFlag);
  });

  const noteItems = Array.from(
    indicators.reduce((notes, indicator) => {
      const title = indicator.noteLabel || indicator.label;
      const key = `${title}|${indicator.unit || ""}|${indicator.description}`;
      if (!notes.has(key)) {
        notes.set(key, {
          title,
          unit: indicator.unit,
          description: indicator.description,
        });
      }
      return notes;
    }, new Map()).values()
  );

  $("#dataIndicatorNotes").innerHTML = noteItems
    .map(
      (note) => `
        <article class="indicator-note">
          <div>
            <strong>${escapeHtml(note.title)}</strong>
            <span>Unit: ${escapeHtml(note.unit || "not specified")}</span>
          </div>
          <p>${escapeHtml(note.description)}</p>
        </article>
      `
    )
    .join("");
}

function renderScoreTable() {
  const rows = DATA.dataTab?.normalisedScores || [];
  const query = state.dataSearch.trim().toLowerCase();
  const filteredRows = rows.filter((row) => {
    if (!query) return true;
    return [row.model, row.scenario, row.category, row.modelFamily, row.sspFamily]
      .filter(Boolean)
      .some((value) => value.toLowerCase().includes(query));
  });
  const dimensions = DATA.metadata.dimensionsGlobal;
  const headerCells = [
    "Model",
    "Scenario",
    "Temperature",
    "Model family",
    ...dimensions.map((dimension) => dimension.shortLabel),
  ];
  const bodyRows = filteredRows
    .map(
      (row) => `
        <tr>
          <td>${escapeHtml(row.model)}</td>
          <td>${escapeHtml(row.scenario)}</td>
          <td>${escapeHtml(row.category || "n/a")}</td>
          <td>${escapeHtml(row.modelFamily || "n/a")}</td>
          ${dimensions.map((dimension) => `<td class="numeric">${formatScore(row.scores?.[dimension.id])}</td>`).join("")}
        </tr>
      `
    )
    .join("");

  $("#scoreTable").innerHTML = `
    <p class="table-status">${filteredRows.length} of ${rows.length} scenarios shown.</p>
    <table>
      <thead>
        <tr>${headerCells.map((cell) => `<th>${escapeHtml(cell)}</th>`).join("")}</tr>
      </thead>
      <tbody>${bodyRows || `<tr><td colspan="${headerCells.length}">No scenarios match the search.</td></tr>`}</tbody>
    </table>
  `;
}

function renderData() {
  renderDataControls();
  const search = $("#scoreTableSearch");
  search.value = state.dataSearch;
  search.oninput = () => {
    state.dataSearch = search.value;
    renderScoreTable();
  };
  renderDataIndicatorPlot();
  renderScoreTable();
}

function renderRegionalControls() {
  populateSelect($("#regionalNormalisation"), DATA.metadata.normalisations, state.regionalNormalisation, (value) => {
    state.regionalNormalisation = value;
    renderRegional();
  });
  renderCheckboxGroup($("#regionalCategories"), DATA.metadata.categories, state.regionalCategories, renderRegional);
  renderCheckboxGroup(
    $("#regionalFamilies"),
    DATA.metadata.modelFamilies.map((family) => ({ id: family, label: family })),
    state.regionalFamilies,
    renderRegional
  );
  renderCheckboxGroup($("#regionalRegions"), DATA.metadata.regions, state.selectedRegions, renderRegional);
}

function regionalRowsFor(region) {
  return activeRegionalRows().filter(
    (row) => row.region === region && rowMatches(row, state.regionalCategories, state.regionalFamilies)
  );
}

function renderRegional() {
  renderRegionalControls();
  const grid = $("#regionalGrid");
  grid.innerHTML = "";

  const selectedRegions = DATA.metadata.regions.filter((region) => state.selectedRegions.has(region.id));
  if (!selectedRegions.length) {
    grid.innerHTML = `<div class="method-note"><p>Select at least one R10 region.</p></div>`;
    return;
  }

  selectedRegions.forEach((region) => {
    const rows = regionalRowsFor(region.id);
    const card = document.createElement("article");
    card.className = "region-card";
    const plotId = `regional-${region.id.replace(/[^a-z0-9]/gi, "-")}`;
    card.innerHTML = `
      <h3>${region.label}</h3>
      <span class="muted">${rows.length} scenarios match the filters</span>
      <div id="${plotId}" class="plot"></div>
    `;
    grid.appendChild(card);
    renderRadar(plotId, rows, DATA.metadata.dimensionsRegional, {
      showScenarioLines: false,
      scenarioOpacity: 0.34,
      fillMedians: false,
      showLegend: false,
      medianWidth: 2.5,
      markerSize: 5,
      pointSize: 4.5,
    });
  });
}

function setActiveScreen(screenId) {
  document.querySelectorAll(".screen").forEach((screen) => {
    screen.classList.toggle("is-active", screen.id === screenId);
  });
  document.querySelectorAll("[data-screen-link]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.screenLink === screenId);
  });
  if (window.location.hash !== `#${screenId}`) {
    history.replaceState(null, "", `#${screenId}`);
  }
  if (screenId === "data" && DATA && state) {
    renderData();
  }
  setTimeout(() => {
    window.dispatchEvent(new Event("resize"));
  }, 0);
}

function bindNavigation() {
  document.querySelectorAll("[data-screen-link]").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.preventDefault();
      setActiveScreen(button.dataset.screenLink);
    });
  });
  const initial = window.location.hash.replace("#", "");
  if (initial === "tradeoffs") {
    setActiveScreen("explore");
  } else if (["about", "explore", "data"].includes(initial)) {
    setActiveScreen(initial);
  } else if (initial === "regional") {
    setActiveScreen("about");
  }
}

function initState() {
  state = {
    aboutSummaryStatistic: "median",
    tradeoffRegion: "World",
    tradeoffNormalisation: "within",
    tradeoffGroupBy: "category",
    tradeoffCategories: new Set(DATA.metadata.categories.map((category) => category.id)),
    tradeoffFamilies: new Set(DATA.metadata.modelFamilies),
    thresholds: Object.fromEntries(DATA.metadata.dimensionsGlobal.map((dimension) => [dimension.id, 1])),
    regionalNormalisation: "within",
    regionalCategories: new Set(DATA.metadata.categories.map((category) => category.id)),
    regionalFamilies: new Set(DATA.metadata.modelFamilies),
    selectedRegions: new Set(DATA.metadata.regions.map((region) => region.id)),
    dataDimension: DATA.dataTab?.dimensions?.[0]?.id || "",
    dataGroupBy: "category",
    dataSearch: "",
  };
}

function bindActions() {
  $("#resetThresholds").addEventListener("click", () => {
    DATA.metadata.dimensionsGlobal.forEach((dimension) => {
      state.thresholds[dimension.id] = 1;
    });
    renderTradeoffs();
  });
}

function showLoadError(error) {
  const node = $("#loadError");
  node.hidden = false;
  node.textContent = `Could not load dashboard/data/dashboard-data.json. Run a local server from the dashboard directory, then refresh. ${error.message}`;
}

fetch("data/dashboard-data.json", { cache: "no-store" })
  .then((response) => {
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    return response.json();
  })
  .then((payload) => {
    DATA = payload;
    initState();
    bindNavigation();
    bindActions();
    renderAbout();
    renderTradeoffs();
    renderData();
  })
  .catch(showLoadError);
