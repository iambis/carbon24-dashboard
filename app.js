let rawData = [];
let filteredData = [];
let clusterCol = null;
let algorithmCol = "algorithm";
let currentAlgorithm = null;

const DATA_PATH = "data/all_models_dashboard.csv";

const preferredNumericColumns = [
  "relative_energy",
  "relative_energy_meV",
  "density",
  "volume_per_atom",
  "mean_bond_length",
  "std_bond_length",
  "mean_coordination",
  "std_coordination",
  "lattice_anisotropy",
  "angle_std",
  "PC1",
  "PC2",
];

function setStatus(text) {
  document.getElementById("statusText").textContent = text;
}

function toNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function isFiniteNumber(v) {
  return typeof v === "number" && Number.isFinite(v);
}

function mean(arr) {
  const valid = arr.filter(isFiniteNumber);
  if (valid.length === 0) return null;
  return valid.reduce((a, b) => a + b, 0) / valid.length;
}

function std(arr) {
  const valid = arr.filter(isFiniteNumber);
  if (valid.length <= 1) return null;
  const m = mean(valid);
  const variance =
    valid.reduce((s, x) => s + Math.pow(x - m, 2), 0) / (valid.length - 1);
  return Math.sqrt(variance);
}

function median(arr) {
  const valid = arr.filter(isFiniteNumber).sort((a, b) => a - b);
  if (valid.length === 0) return null;
  const mid = Math.floor(valid.length / 2);
  if (valid.length % 2 === 0) {
    return (valid[mid - 1] + valid[mid]) / 2;
  }
  return valid[mid];
}

function unique(arr) {
  return [...new Set(arr)];
}
function populateAlgorithmOptions() {
  const algoSelect = document.getElementById("algorithmSelect");

  if (!algoSelect) return;

  algoSelect.innerHTML = "";

  if (!rawData || rawData.length === 0 || !(algorithmCol in rawData[0])) {
    const option = document.createElement("option");
    option.value = "All";
    option.textContent = "All";
    algoSelect.appendChild(option);
    currentAlgorithm = "All";
    return;
  }

  const algorithms = unique(rawData.map((d) => d[algorithmCol]))
    .filter((v) => v !== null && v !== undefined)
    .sort();

  algorithms.forEach((algo) => {
    const option = document.createElement("option");
    option.value = algo;
    option.textContent = algo;
    algoSelect.appendChild(option);
  });

  currentAlgorithm = algorithms[0];

  algoSelect.value = currentAlgorithm;

  algoSelect.addEventListener("change", () => {
    currentAlgorithm = algoSelect.value;

    initClusterCheckboxes();
    updateDashboard();
  });
}
function detectClusterColumn(data) {
  const candidates = [
    "cluster_label",
    "cluster_kmeans",
    "cluster_hierarchical",
    "cluster_hdbscan",
    "cluster_gmm",
    "cluster",
    "label",
    "labels",
  ];

  const cols = Object.keys(data[0]);

  for (const c of candidates) {
    if (cols.includes(c)) return c;
  }

  const clusterLike = cols.find((c) => c.toLowerCase().includes("cluster"));
  return clusterLike || null;
}

function preprocessData(data) {
  if (!data || data.length === 0) return [];

  clusterCol = detectClusterColumn(data);

  const cols = Object.keys(data[0]);

  data.forEach((row) => {
    cols.forEach((col) => {
      const n = toNumber(row[col]);
      if (n !== null && row[col] !== "") {
        row[col] = n;
      }
    });

    if ("relative_energy" in row && !("relative_energy_meV" in row)) {
      const e = toNumber(row.relative_energy);
      row.relative_energy_meV = e !== null ? e * 1000 : null;
    }

    if (clusterCol && row[clusterCol] !== undefined) {
      row[clusterCol] = String(row[clusterCol]);
    }
  });

  return data;
}

function getNumericColumns(data) {
  if (!data || data.length === 0) return [];

  const cols = Object.keys(data[0]);

  return cols.filter((col) => {
    const sample = data
      .slice(0, 100)
      .map((d) => d[col])
      .filter((v) => v !== null && v !== "");
    if (sample.length === 0) return false;
    const numericCount = sample.filter(
      (v) => typeof v === "number" && Number.isFinite(v),
    ).length;
    return numericCount / sample.length >= 0.8;
  });
}

function populateColorOptions() {
  const colorSelect = document.getElementById("colorBy");
  colorSelect.innerHTML = "";

  const options = [];

  if (clusterCol) {
    options.push({
      value: "cluster",
      label: "Cluster",
    });
  }

  const preferred = [
    "relative_energy_meV",
    "relative_energy",
    "density",
    "volume_per_atom",
    "mean_coordination",
    "mean_bond_length",
    "crystal_system",
    "space_group_number",
  ];

  preferred.forEach((col) => {
    if (rawData.length > 0 && col in rawData[0]) {
      options.push({
        value: col,
        label: col,
      });
    }
  });

  options.forEach((opt) => {
    const option = document.createElement("option");
    option.value = opt.value;
    option.textContent = opt.label;
    colorSelect.appendChild(option);
  });
}

function populateFeatureBoxOptions() {
  const featureSelect = document.getElementById("featureBoxSelect");
  featureSelect.innerHTML = "";

  const features = [
    "density",
    "volume_per_atom",
    "mean_bond_length",
    "std_bond_length",
    "mean_coordination",
    "std_coordination",
    "lattice_anisotropy",
    "angle_std",
    "relative_energy_meV",
  ].filter((c) => rawData.length > 0 && c in rawData[0]);

  features.forEach((col) => {
    const option = document.createElement("option");
    option.value = col;
    option.textContent = col;
    featureSelect.appendChild(option);
  });
}

function initClusterCheckboxes() {
  const box = document.getElementById("clusterCheckboxes");
  box.innerHTML = "";

  if (!clusterCol) {
    box.innerHTML = "<p>Không tìm thấy cột cluster.</p>";
    return;
  }

  let dataForClusters = rawData.slice();

  if (
    algorithmCol &&
    currentAlgorithm &&
    currentAlgorithm !== "All" &&
    rawData.length > 0 &&
    algorithmCol in rawData[0]
  ) {
    dataForClusters = dataForClusters.filter(
      (d) => String(d[algorithmCol]) === String(currentAlgorithm),
    );
  }

  const clusters = unique(dataForClusters.map((d) => d[clusterCol]))
    .filter((v) => v !== null && v !== undefined)
    .sort((a, b) => Number(a) - Number(b));

  clusters.forEach((c) => {
    const div = document.createElement("div");
    div.className = "checkbox-item";

    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = c;
    input.checked = true;
    input.className = "cluster-check";

    const label = document.createElement("label");
    label.textContent = `Cluster ${c}`;

    div.appendChild(input);
    div.appendChild(label);
    box.appendChild(div);
  });

  document.querySelectorAll(".cluster-check").forEach((chk) => {
    chk.addEventListener("change", updateDashboard);
  });
}

function initNumericFilters() {
  const energyInput = document.getElementById("energyMax");
  const densityInput = document.getElementById("densityMin");

  if (rawData.length > 0 && "relative_energy_meV" in rawData[0]) {
    const energies = rawData
      .map((d) => d.relative_energy_meV)
      .filter(isFiniteNumber);
    if (energies.length > 0) {
      energyInput.value = Math.ceil(Math.max(...energies));
    }
  }

  if (rawData.length > 0 && "density" in rawData[0]) {
    const densities = rawData.map((d) => d.density).filter(isFiniteNumber);
    if (densities.length > 0) {
      densityInput.value = Math.floor(Math.min(...densities) * 100) / 100;
    }
  }
}

function initControls() {
  populateAlgorithmOptions();
  populateColorOptions();
  populateFeatureBoxOptions();
  initClusterCheckboxes();
  initNumericFilters();

  document
    .getElementById("colorBy")
    .addEventListener("change", updateDashboard);
  document
    .getElementById("featureBoxSelect")
    .addEventListener("change", updateDashboard);
  document
    .getElementById("energyMax")
    .addEventListener("input", updateDashboard);
  document
    .getElementById("densityMin")
    .addEventListener("input", updateDashboard);

  document.getElementById("resetBtn").addEventListener("click", () => {
    document.querySelectorAll(".cluster-check").forEach((chk) => {
      chk.checked = true;
    });

    initNumericFilters();

    const colorSelect = document.getElementById("colorBy");
    if (colorSelect.options.length > 0) {
      colorSelect.selectedIndex = 0;
    }

    updateDashboard();
  });

  document
    .getElementById("downloadBtn")
    .addEventListener("click", downloadFilteredCSV);
}

function getSelectedClusters() {
  return Array.from(document.querySelectorAll(".cluster-check"))
    .filter((chk) => chk.checked)
    .map((chk) => String(chk.value));
}

function applyFilters() {
  const selectedClusters = getSelectedClusters();

  let data = rawData.slice();
  if (
    algorithmCol &&
    currentAlgorithm &&
    currentAlgorithm !== "All" &&
    data.length > 0 &&
    algorithmCol in data[0]
  ) {
    data = data.filter(
      (d) => String(d[algorithmCol]) === String(currentAlgorithm),
    );
  }

  if (clusterCol && selectedClusters.length > 0) {
    data = data.filter((d) => selectedClusters.includes(String(d[clusterCol])));
  }

  const energyMaxValue = document.getElementById("energyMax").value;
  const densityMinValue = document.getElementById("densityMin").value;

  if (
    energyMaxValue !== "" &&
    rawData.length > 0 &&
    "relative_energy_meV" in rawData[0]
  ) {
    const energyMax = Number(energyMaxValue);
    if (Number.isFinite(energyMax)) {
      data = data.filter((d) => {
        if (!isFiniteNumber(d.relative_energy_meV)) return false;
        return d.relative_energy_meV <= energyMax;
      });
    }
  }

  if (densityMinValue !== "" && rawData.length > 0 && "density" in rawData[0]) {
    const densityMin = Number(densityMinValue);
    if (Number.isFinite(densityMin)) {
      data = data.filter((d) => {
        if (!isFiniteNumber(d.density)) return false;
        return d.density >= densityMin;
      });
    }
  }

  filteredData = data;
}

function updateStats() {
  document.getElementById("numStructures").textContent = filteredData.length;

  if (clusterCol) {
    const nClusters = unique(filteredData.map((d) => d[clusterCol])).length;
    document.getElementById("numClusters").textContent = nClusters;
  } else {
    document.getElementById("numClusters").textContent = "-";
  }

  if (filteredData.length > 0 && "relative_energy_meV" in filteredData[0]) {
    const mEnergy = mean(filteredData.map((d) => d.relative_energy_meV));
    document.getElementById("meanEnergy").textContent =
      mEnergy !== null ? `${mEnergy.toFixed(2)} meV` : "-";
  } else {
    document.getElementById("meanEnergy").textContent = "-";
  }

  if (filteredData.length > 0 && "density" in filteredData[0]) {
    const mDensity = mean(filteredData.map((d) => d.density));
    document.getElementById("meanDensity").textContent =
      mDensity !== null ? mDensity.toFixed(3) : "-";
  } else {
    document.getElementById("meanDensity").textContent = "-";
  }
}

function getHoverText(row) {
  const fields = [
    "row_index",
    clusterCol,
    "relative_energy_meV",
    "density",
    "volume_per_atom",
    "mean_bond_length",
    "mean_coordination",
    "crystal_system",
    "space_group_number",
  ];

  return fields
    .filter((f) => f && f in row)
    .map((f) => `${f}: ${row[f]}`)
    .join("<br>");
}

function plotPCA() {
  const plotDiv = document.getElementById("pcaPlot");

  if (filteredData.length === 0) {
    Plotly.purge(plotDiv);
    plotDiv.innerHTML = "<p>Không có dữ liệu sau khi lọc.</p>";
    return;
  }

  if (!("PC1" in filteredData[0]) || !("PC2" in filteredData[0])) {
    plotDiv.innerHTML = `
            <p>
                File CSV chưa có cột PC1 và PC2. 
                Hãy xuất lại file KMeans có PCA hoặc thêm PC1/PC2 vào CSV.
            </p>
        `;
    return;
  }

  const colorBy = document.getElementById("colorBy").value;

  let traces = [];

  if (colorBy === "cluster" && clusterCol) {
    const clusters = unique(filteredData.map((d) => d[clusterCol])).sort(
      (a, b) => Number(a) - Number(b),
    );

    traces = clusters.map((c) => {
      const clusterData = filteredData.filter(
        (d) => String(d[clusterCol]) === String(c),
      );

      return {
        x: clusterData.map((d) => d.PC1),
        y: clusterData.map((d) => d.PC2),
        mode: "markers",
        type: "scattergl",
        name: `Cluster ${c}`,
        text: clusterData.map(getHoverText),
        hoverinfo: "text",
        marker: {
          size: 7,
          opacity: 0.8,
        },
      };
    });
  } else {
    const colorValues = filteredData.map((d) => d[colorBy]);

    traces = [
      {
        x: filteredData.map((d) => d.PC1),
        y: filteredData.map((d) => d.PC2),
        mode: "markers",
        type: "scattergl",
        text: filteredData.map(getHoverText),
        hoverinfo: "text",
        marker: {
          size: 7,
          opacity: 0.8,
          color: colorValues,
          colorscale: "Viridis",
          showscale: true,
          colorbar: {
            title: colorBy,
          },
        },
        name: colorBy,
      },
    ];
  }

  const layout = {
    title: `PCA Projection - colored by ${colorBy}`,
    xaxis: { title: "PC1" },
    yaxis: { title: "PC2" },
    hovermode: "closest",
    margin: { l: 60, r: 30, t: 60, b: 60 },
    legend: {
      orientation: "v",
    },
  };

  Plotly.newPlot(plotDiv, traces, layout, { responsive: true });
}

function plotEnergyBox() {
  const plotDiv = document.getElementById("energyBoxPlot");

  if (
    filteredData.length === 0 ||
    !clusterCol ||
    !("relative_energy_meV" in filteredData[0])
  ) {
    Plotly.purge(plotDiv);
    plotDiv.innerHTML = "<p>Không đủ dữ liệu để vẽ energy boxplot.</p>";
    return;
  }

  const clusters = unique(filteredData.map((d) => d[clusterCol])).sort(
    (a, b) => Number(a) - Number(b),
  );

  const traces = clusters.map((c) => {
    const clusterData = filteredData.filter(
      (d) => String(d[clusterCol]) === String(c),
    );

    return {
      y: clusterData.map((d) => d.relative_energy_meV).filter(isFiniteNumber),
      type: "box",
      name: `Cluster ${c}`,
      boxpoints: "outliers",
    };
  });

  const layout = {
    title: "Relative Energy by Cluster",
    yaxis: { title: "Relative energy (meV)" },
    xaxis: { title: "Cluster" },
    margin: { l: 70, r: 30, t: 60, b: 60 },
  };

  Plotly.newPlot(plotDiv, traces, layout, { responsive: true });
}

function plotDensityEnergy() {
  const plotDiv = document.getElementById("densityEnergyPlot");

  if (
    filteredData.length === 0 ||
    !clusterCol ||
    !("density" in filteredData[0]) ||
    !("relative_energy_meV" in filteredData[0])
  ) {
    Plotly.purge(plotDiv);
    plotDiv.innerHTML = "<p>Không đủ dữ liệu để vẽ density-energy.</p>";
    return;
  }

  const clusters = unique(filteredData.map((d) => d[clusterCol])).sort(
    (a, b) => Number(a) - Number(b),
  );

  const traces = clusters.map((c) => {
    const clusterData = filteredData.filter(
      (d) => String(d[clusterCol]) === String(c),
    );

    return {
      x: clusterData.map((d) => d.density),
      y: clusterData.map((d) => d.relative_energy_meV),
      mode: "markers",
      type: "scatter",
      name: `Cluster ${c}`,
      text: clusterData.map(getHoverText),
      hoverinfo: "text",
      marker: {
        size: 7,
        opacity: 0.8,
      },
    };
  });

  const layout = {
    title: "Density vs Relative Energy",
    xaxis: { title: "Density" },
    yaxis: { title: "Relative energy (meV)" },
    margin: { l: 70, r: 30, t: 60, b: 60 },
  };

  Plotly.newPlot(plotDiv, traces, layout, { responsive: true });
}

function plotFeatureBox() {
  const plotDiv = document.getElementById("featureBoxPlot");
  const feature = document.getElementById("featureBoxSelect").value;

  if (
    filteredData.length === 0 ||
    !clusterCol ||
    !feature ||
    !(feature in filteredData[0])
  ) {
    Plotly.purge(plotDiv);
    plotDiv.innerHTML = "<p>Không đủ dữ liệu để vẽ feature boxplot.</p>";
    return;
  }

  const clusters = unique(filteredData.map((d) => d[clusterCol])).sort(
    (a, b) => Number(a) - Number(b),
  );

  const traces = clusters.map((c) => {
    const clusterData = filteredData.filter(
      (d) => String(d[clusterCol]) === String(c),
    );

    return {
      y: clusterData.map((d) => d[feature]).filter(isFiniteNumber),
      type: "box",
      name: `Cluster ${c}`,
      boxpoints: "outliers",
    };
  });

  const layout = {
    title: `${feature} by Cluster`,
    yaxis: { title: feature },
    xaxis: { title: "Cluster" },
    margin: { l: 70, r: 30, t: 60, b: 60 },
  };

  Plotly.newPlot(plotDiv, traces, layout, { responsive: true });
}

function computeClusterProfile() {
  if (!clusterCol || filteredData.length === 0) return [];

  const profileFeatures = [
    "volume_per_atom",
    "density",
    "mean_bond_length",
    "mean_coordination",
    "relative_energy_meV",
  ].filter((c) => c in filteredData[0]);

  const clusters = unique(filteredData.map((d) => d[clusterCol])).sort(
    (a, b) => Number(a) - Number(b),
  );

  return clusters.map((c) => {
    const clusterData = filteredData.filter(
      (d) => String(d[clusterCol]) === String(c),
    );

    const row = {
      cluster: c,
      n_samples: clusterData.length,
    };

    profileFeatures.forEach((f) => {
      const values = clusterData.map((d) => d[f]).filter(isFiniteNumber);
      row[`${f}_mean`] = mean(values);
      row[`${f}_std`] = std(values);
      row[`${f}_median`] = median(values);
    });

    return row;
  });
}

function renderClusterProfile() {
  const div = document.getElementById("clusterProfile");
  const profile = computeClusterProfile();

  if (profile.length === 0) {
    div.innerHTML = "<p>Không có dữ liệu cluster profile.</p>";
    return;
  }

  const cols = Object.keys(profile[0]);

  let html = "<div class='table-wrapper'><table class='profile-table'>";
  html += "<thead><tr>";

  cols.forEach((c) => {
    html += `<th>${c}</th>`;
  });

  html += "</tr></thead><tbody>";

  profile.forEach((row) => {
    html += "<tr>";

    cols.forEach((c) => {
      let value = row[c];

      if (typeof value === "number" && Number.isFinite(value)) {
        value = value.toFixed(4);
      }

      html += `<td>${value ?? ""}</td>`;
    });

    html += "</tr>";
  });

  html += "</tbody></table></div>";

  div.innerHTML = html;
}

function renderDataTable() {
  const table = document.getElementById("dataTable");
  const thead = table.querySelector("thead");
  const tbody = table.querySelector("tbody");

  thead.innerHTML = "";
  tbody.innerHTML = "";

  if (filteredData.length === 0) {
    tbody.innerHTML = "<tr><td>Không có dữ liệu.</td></tr>";
    return;
  }

  const preferredCols = [
    "row_index",
    clusterCol,
    "relative_energy",
    "relative_energy_meV",
    "density",
    "volume_per_atom",
    "mean_bond_length",
    "mean_coordination",
    "crystal_system",
    "space_group_number",
    "PC1",
    "PC2",
  ].filter((c) => c && c in filteredData[0]);

  const cols =
    preferredCols.length > 0
      ? preferredCols
      : Object.keys(filteredData[0]).slice(0, 12);

  const headerRow = document.createElement("tr");

  cols.forEach((c) => {
    const th = document.createElement("th");
    th.textContent = c;
    headerRow.appendChild(th);
  });

  thead.appendChild(headerRow);

  const maxRows = Math.min(300, filteredData.length);

  for (let i = 0; i < maxRows; i++) {
    const row = filteredData[i];
    const tr = document.createElement("tr");

    cols.forEach((c) => {
      const td = document.createElement("td");
      let value = row[c];

      if (typeof value === "number" && Number.isFinite(value)) {
        value = value.toFixed(5);
      }

      td.textContent = value ?? "";
      tr.appendChild(td);
    });

    tbody.appendChild(tr);
  }
}

function convertToCSV(data) {
  if (!data || data.length === 0) return "";

  const cols = Object.keys(data[0]);

  const escapeCSV = (value) => {
    if (value === null || value === undefined) return "";
    const str = String(value);
    if (str.includes(",") || str.includes("\n") || str.includes('"')) {
      return `"${str.replace(/"/g, '""')}"`;
    }
    return str;
  };

  const header = cols.join(",");
  const rows = data.map((row) => {
    return cols.map((col) => escapeCSV(row[col])).join(",");
  });

  return [header, ...rows].join("\n");
}

function downloadFilteredCSV() {
  if (!filteredData || filteredData.length === 0) {
    alert("Không có dữ liệu để download.");
    return;
  }

  const csv = convertToCSV(filteredData);
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });

  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");

  link.setAttribute("href", url);
  link.setAttribute("download", "carbon24_filtered_dashboard_data.csv");
  link.style.visibility = "hidden";

  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  URL.revokeObjectURL(url);
}

function updateDashboard() {
  applyFilters();

  updateStats();

  plotPCA();
  plotPCA3D();
  plotEnergyBox();
  plotDensityEnergy();
  plotFeatureBox();

  renderClusterProfile();
  renderDataTable();

  setStatus(
    `Algorithm: ${currentAlgorithm || "N/A"} | Đã tải ${rawData.length} dòng, đang hiển thị ${filteredData.length} dòng.`,
  );
}
function plotPCA3D() {
  const plotDiv = document.getElementById("pca3DPlot");

  if (!plotDiv) return;

  if (filteredData.length === 0) {
    Plotly.purge(plotDiv);
    plotDiv.innerHTML = "<p>Không có dữ liệu sau khi lọc.</p>";
    return;
  }

  if (
    !("PC1" in filteredData[0]) ||
    !("PC2" in filteredData[0]) ||
    !("PC3" in filteredData[0])
  ) {
    plotDiv.innerHTML = `
            <p>
                File CSV chưa có PC1, PC2, PC3. 
                Hãy tạo PCA 3D trước bằng Python.
            </p>
        `;
    return;
  }

  const colorBy = document.getElementById("colorBy").value;

  let traces = [];

  if (colorBy === "cluster" && clusterCol) {
    const clusters = unique(filteredData.map((d) => d[clusterCol])).sort(
      (a, b) => Number(a) - Number(b),
    );

    traces = clusters.map((c) => {
      const clusterData = filteredData.filter(
        (d) => String(d[clusterCol]) === String(c),
      );

      return {
        x: clusterData.map((d) => d.PC1),
        y: clusterData.map((d) => d.PC2),
        z: clusterData.map((d) => d.PC3),
        mode: "markers",
        type: "scatter3d",
        name: `Cluster ${c}`,
        text: clusterData.map(getHoverText),
        hoverinfo: "text",
        marker: {
          size: 4,
          opacity: 0.75,
        },
      };
    });
  } else {
    const colorValues = filteredData.map((d) => d[colorBy]);

    traces = [
      {
        x: filteredData.map((d) => d.PC1),
        y: filteredData.map((d) => d.PC2),
        z: filteredData.map((d) => d.PC3),
        mode: "markers",
        type: "scatter3d",
        text: filteredData.map(getHoverText),
        hoverinfo: "text",
        marker: {
          size: 4,
          opacity: 0.75,
          color: colorValues,
          colorscale: "Viridis",
          showscale: true,
          colorbar: {
            title: colorBy,
          },
        },
        name: colorBy,
      },
    ];
  }

  const layout = {
    title: `3D PCA Projection - colored by ${colorBy}`,
    scene: {
      xaxis: { title: "PC1" },
      yaxis: { title: "PC2" },
      zaxis: { title: "PC3" },
    },
    margin: { l: 0, r: 0, t: 50, b: 0 },
    legend: {
      orientation: "v",
    },
  };

  Plotly.newPlot(plotDiv, traces, layout, { responsive: true });
}
function loadCSVFromPath(path) {
  setStatus(`Đang tải file: ${path}`);

  Papa.parse(path, {
    download: true,
    header: true,
    skipEmptyLines: true,
    dynamicTyping: false,

    complete: function (results) {
      if (results.errors && results.errors.length > 0) {
        console.warn("PapaParse errors:", results.errors);
      }

      if (!results.data || results.data.length === 0) {
        setStatus("File CSV rỗng hoặc không đọc được.");
        return;
      }

      rawData = preprocessData(results.data);

      if (!clusterCol) {
        setStatus("Không tìm thấy cột cluster. Hãy kiểm tra file CSV.");
        alert("Không tìm thấy cột cluster trong CSV.");
        return;
      }

      initControls();
      updateDashboard();
    },

    error: function (error) {
      console.error(error);
      setStatus("Không tải được CSV từ data/. Hãy thử upload CSV thủ công.");
    },
  });
}

function loadCSVFromUpload(file) {
  if (!file) return;

  setStatus(`Đang đọc file upload: ${file.name}`);

  Papa.parse(file, {
    header: true,
    skipEmptyLines: true,
    dynamicTyping: false,

    complete: function (results) {
      if (results.errors && results.errors.length > 0) {
        console.warn("PapaParse errors:", results.errors);
      }

      if (!results.data || results.data.length === 0) {
        setStatus("File upload rỗng hoặc không đọc được.");
        return;
      }

      rawData = preprocessData(results.data);

      if (!clusterCol) {
        setStatus("Không tìm thấy cột cluster trong file upload.");
        alert("Không tìm thấy cột cluster trong CSV.");
        return;
      }

      initControls();
      updateDashboard();

      setStatus(`Đã tải file upload: ${file.name}`);
    },

    error: function (error) {
      console.error(error);
      setStatus("Lỗi khi đọc file upload.");
    },
  });
}

function attachUploadHandler() {
  const uploadInput = document.getElementById("csvUpload");

  uploadInput.addEventListener("change", function (event) {
    const file = event.target.files[0];

    if (!file) return;

    loadCSVFromUpload(file);
  });
}

function checkRequiredColumns() {
  if (!rawData || rawData.length === 0) return;

  const cols = Object.keys(rawData[0]);

  const recommended = [
    clusterCol,
    "relative_energy",
    "relative_energy_meV",
    "density",
    "volume_per_atom",
    "mean_coordination",
    "PC1",
    "PC2",
  ].filter(Boolean);

  const missing = recommended.filter((c) => !cols.includes(c));

  if (missing.length > 0) {
    console.warn("Missing recommended columns:", missing);
  }
}

function initializeApp() {
  attachUploadHandler();

  loadCSVFromPath(DATA_PATH);
}

document.addEventListener("DOMContentLoaded", initializeApp);
