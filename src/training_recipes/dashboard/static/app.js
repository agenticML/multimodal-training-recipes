const state = {
  task: null,
  architecture: null,
  runId: null,
  inFlight: false,
  eventSource: null,
};

const taskList = document.getElementById("task-list");
const archList = document.getElementById("arch-list");
const archHint = document.getElementById("arch-hint");
const selection = document.getElementById("selection");
const runBtn = document.getElementById("run-btn");
const statusLine = document.getElementById("status-line");
const logsEl = document.getElementById("logs");
const summaryPanel = document.getElementById("summary-panel");
const summaryEl = document.getElementById("summary");
const assetsPanel = document.getElementById("assets-panel");
const fileTree = document.getElementById("file-tree");
const fileView = document.getElementById("file-view");
const assetChip = document.getElementById("asset-chip");

function setStatus(text) {
  statusLine.textContent = text;
}

function updateRunButton() {
  const supported = Boolean(state.architecture && state.architecture.dummy_supported);
  runBtn.disabled = !supported || state.inFlight;
}

function renderSelection() {
  if (!state.task || !state.architecture) {
    selection.textContent = "No architecture selected";
    return;
  }
  const mark = state.architecture.dummy_supported ? "" : " (not runnable)";
  selection.textContent = `${state.task} / ${state.architecture.name}${mark}`;
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail || response.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

function renderTasks(tasks) {
  taskList.replaceChildren();
  for (const name of tasks) {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = name;
    button.addEventListener("click", () => selectTask(name, button));
    item.appendChild(button);
    taskList.appendChild(item);
  }
}

async function selectTask(name, button) {
  state.task = name;
  state.architecture = null;
  for (const node of taskList.querySelectorAll("button")) {
    node.classList.toggle("selected", node === button);
  }
  renderSelection();
  updateRunButton();
  archHint.textContent = "Loading";
  const data = await fetchJson(`/api/tasks/${encodeURIComponent(name)}`);
  renderArchitectures(data.architectures);
}

function renderArchitectures(architectures) {
  archList.replaceChildren();
  archHint.textContent = architectures.length ? "" : "No architectures";
  for (const spec of architectures) {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.disabled = !spec.dummy_supported;
    const label = document.createElement("span");
    label.textContent = spec.name;
    const meta = document.createElement("span");
    meta.className = "meta";
    meta.textContent = spec.dummy_supported
      ? `${spec.family} · ${spec.class}`
      : `${spec.family} · ${spec.library || "custom/mlx"} · not runnable`;
    button.appendChild(label);
    button.appendChild(meta);
    if (spec.dummy_supported) {
      button.addEventListener("click", () => selectArchitecture(spec, button));
    }
    item.appendChild(button);
    archList.appendChild(item);
  }
}

function selectArchitecture(spec, button) {
  state.architecture = spec;
  for (const node of archList.querySelectorAll("button")) {
    node.classList.toggle("selected", node === button);
  }
  renderSelection();
  updateRunButton();
}

function appendLog(line) {
  const atBottom = logsEl.scrollHeight - logsEl.scrollTop - logsEl.clientHeight < 24;
  logsEl.textContent += (logsEl.textContent ? "\n" : "") + line;
  if (atBottom) {
    logsEl.scrollTop = logsEl.scrollHeight;
  }
}

function renderSummary(summary) {
  summaryEl.replaceChildren();
  if (!summary) {
    summaryPanel.classList.add("hidden");
    return;
  }
  summaryPanel.classList.remove("hidden");
  const keys = [
    "task",
    "architecture",
    "profile",
    "steps",
    "final_loss",
    "model_class",
    "parameter_count",
    "elapsed_seconds",
  ];
  const seen = new Set();
  for (const key of keys.concat(Object.keys(summary))) {
    if (seen.has(key) || !(key in summary)) {
      continue;
    }
    seen.add(key);
    const dt = document.createElement("dt");
    dt.textContent = key;
    const dd = document.createElement("dd");
    dd.textContent = String(summary[key]);
    summaryEl.appendChild(dt);
    summaryEl.appendChild(dd);
  }
}

function renderTree(groups) {
  fileTree.replaceChildren();
  fileView.textContent = "Select a file";
  const names = Object.keys(groups);
  if (!names.length) {
    fileTree.textContent = "No captured code assets";
    return;
  }
  for (const source of names) {
    const heading = document.createElement("h3");
    heading.textContent = source;
    fileTree.appendChild(heading);
    for (const file of groups[source]) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = file.path;
      button.addEventListener("click", () => {
        for (const node of fileTree.querySelectorAll("button")) {
          node.classList.toggle("selected", node === button);
        }
        loadFile(file.path);
      });
      fileTree.appendChild(button);
    }
  }
}

async function loadFile(path) {
  fileView.textContent = "Loading";
  const data = await fetchJson(
    `/api/runs/${encodeURIComponent(state.runId)}/assets/file?path=${encodeURIComponent(path)}`,
  );
  if (data.binary) {
    fileView.textContent = data.note || "binary file skipped";
    return;
  }
  fileView.textContent = data.content || "";
  if (data.truncated && data.note) {
    fileView.textContent += `\n\n[${data.note}]`;
  }
}

async function loadAssets() {
  const data = await fetchJson(`/api/runs/${encodeURIComponent(state.runId)}/assets`);
  assetsPanel.classList.remove("hidden");
  renderTree(data.groups || {});
}

function closeEvents() {
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
}

function listen(runId) {
  closeEvents();
  const source = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events`);
  state.eventSource = source;
  source.addEventListener("log", (event) => {
    appendLog(JSON.parse(event.data));
  });
  source.addEventListener("complete", async (event) => {
    source.close();
    state.eventSource = null;
    state.inFlight = false;
    updateRunButton();
    const payload = JSON.parse(event.data);
    if (payload.status === "finished") {
      setStatus("Finished");
      renderSummary(payload.summary);
      try {
        await loadAssets();
      } catch (error) {
        fileView.textContent = error.message;
      }
    } else {
      setStatus(payload.error || "Failed");
    }
  });
  source.onerror = () => {
    if (!state.inFlight) {
      return;
    }
    setStatus("Log stream interrupted");
  };
}

async function startRun() {
  if (!state.task || !state.architecture || state.inFlight) {
    return;
  }
  state.inFlight = true;
  updateRunButton();
  logsEl.textContent = "";
  summaryPanel.classList.add("hidden");
  assetsPanel.classList.add("hidden");
  setStatus("Starting");
  try {
    const run = await fetchJson("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        task: state.task,
        architecture: state.architecture.name,
      }),
    });
    state.runId = run.id;
    setStatus(`Running ${run.id}`);
    listen(run.id);
  } catch (error) {
    state.inFlight = false;
    updateRunButton();
    setStatus(error.message);
  }
}

async function boot() {
  const meta = await fetchJson("/api/meta");
  assetChip.textContent = `asset dir = ${meta.asset_dir}`;
  const tasks = await fetchJson("/api/tasks");
  renderTasks(tasks.tasks || []);
}

runBtn.addEventListener("click", startRun);
boot().catch((error) => setStatus(error.message));
