const plotConfig = {
  displayModeBar: false,
  responsive: true,
};

let DATA = null;
let state = null;

const $ = (selector) => document.querySelector(selector);
const fallbackGroupColors = ["#0072b2", "#e69f00", "#009e73", "#d55e00", "#cc79a7", "#56b4e9", "#8f6bb1"];

function numberOrZero(value) {
  return Number.isFinite(value) ? value : 0;
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
        opacity: options.scenarioLineOpacity ?? 0.12,
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
          opacity: scenarioOpacity,
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
    margin: { t: title ? 48 : 24, r: 64, b: options.showLegend === false ? 28 : 58, l: 64 },
    showlegend: options.showLegend !== false,
    legend: { orientation: "h", y: -0.08, x: 0, font: { size: 11 } },
    polar: {
      bgcolor: "rgba(0,0,0,0)",
      radialaxis: {
        visible: true,
        range: [0, 1],
        tickvals: [0, 0.25, 0.5, 0.75, 1],
        tickfont: { size: 10, color: "#5d6776" },
        gridcolor: "#d6ddd9",
        linecolor: "#a9b5af",
      },
      angularaxis: {
        gridcolor: "#d6ddd9",
        linecolor: "#a9b5af",
        tickfont: { size: 11, color: "#1c2430" },
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
    scenarioLineOpacity: 0.15,
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
    scenarioLineOpacity: 0.22,
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
  const yAxis = { gridcolor: "#d6ddd9", rangemode: "tozero", title: rows.find((row) => row.unit)?.unit || "" };
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
      xaxis: { gridcolor: "#d6ddd9", zeroline: false },
      yaxis: yAxis,
      font: { family: "Inter, system-ui, sans-serif", color: "#1c2430" },
    },
    plotConfig
  );
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
  } else if (["about", "explore"].includes(initial)) {
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
  })
  .catch(showLoadError);
