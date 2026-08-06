"use strict";

const storageKey = "dspy-custom-lm-workbench-v1";
const shell = document.querySelector("[data-app-shell]");
const showcase = document.querySelector("[data-showcase]");
const panels = [...document.querySelectorAll("[data-gate-panel]")];
const gateButtons = [...document.querySelectorAll("[data-gate-button]")];
const checkboxes = [...document.querySelectorAll("[data-check]")];
const noteFields = [...document.querySelectorAll("[data-note]")];
const progressSegments = [...document.querySelectorAll("[data-progress-strip] span")];
const progressCopy = document.querySelector("[data-progress-copy]");
const evidenceList = document.querySelector("[data-evidence-list]");
const evidenceSummary = document.querySelector("[data-evidence-summary]");
const liveRegion = document.querySelector("[data-live]");

const evidence = [
  ["Workload needs are written", "Normal dspy.LM was evaluated", "One missing seam justifies custom work"],
  ["Smallest native success is captured", "Native failures are classified", "Secrets are removed from evidence"],
  ["Every capability has a probe", "Limits are recorded", "Each row has one honest classification"],
  ["Request fields have destinations", "Response fields have DSPy homes", "Errors and retries have one owner"],
  ["EchoLM works first", "Sync and async return LMResponse", "Provider runtime stays out of state"],
  ["Real DSPy modules pass", "Cache, state, usage, and errors pass", "Blocked features remain explicit"],
  ["Operational contract is written", "All uv quality gates pass", "Clean artifacts install and run"],
];

const defaultState = { gate: 0, checks: {}, notes: {} };

function readState() {
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey));
    if (saved && Number.isInteger(saved.gate) && saved.checks && saved.notes) return saved;
  } catch {}
  return structuredClone(defaultState);
}

let state = readState();

function labelResponsiveTables() {
  document.querySelectorAll(".table-scroll table").forEach((table) => {
    const headers = [...table.querySelectorAll("thead th")].map((header) =>
      header.textContent.trim(),
    );
    table.querySelectorAll("tbody tr").forEach((row) => {
      [...row.children].forEach((cell, index) => {
        cell.dataset.mobileLabel = headers[index] || "Field";
      });
    });
  });
}

function saveState(message) {
  try {
    localStorage.setItem(storageKey, JSON.stringify(state));
  } catch {}
  if (message && liveRegion) liveRegion.textContent = message;
}

function gateChecks(index) {
  const panel = panels[index];
  return [...panel.querySelectorAll("[data-check]")];
}

function noteHasEvidence(field) {
  const value = String(field.value).trim();
  return value.length > 0 && !(field instanceof HTMLSelectElement && value === "Unclassified");
}

function gateEvidenceCounts(index) {
  const checks = gateChecks(index);
  const notes = [...panels[index].querySelectorAll("[data-note]")];
  return {
    complete: checks.filter((checkbox) => checkbox.checked).length + notes.filter(noteHasEvidence).length,
    total: checks.length + notes.length,
  };
}

function gateIsComplete(index) {
  const counts = gateEvidenceCounts(index);
  return counts.total > 0 && counts.complete === counts.total;
}

function renderEvidence(index) {
  if (!evidenceList || !evidenceSummary) return;
  evidenceList.replaceChildren();
  const complete = gateIsComplete(index);
  const counts = gateEvidenceCounts(index);
  const completedItems = complete
    ? evidence[index].length
    : Math.floor((counts.complete / counts.total) * evidence[index].length);
  evidence[index].forEach((label, itemIndex) => {
    const item = document.createElement("div");
    item.className = "evidence-item";
    item.textContent = label;
    if (itemIndex < completedItems) item.dataset.complete = "";
    evidenceList.append(item);
  });
  evidenceSummary.textContent = complete
    ? "This gate has enough recorded evidence to advance."
    : "Complete the evidence tasks inside the current gate.";
}

function renderProgress() {
  const completed = panels.filter((_, index) => gateIsComplete(index)).length;
  if (progressCopy) progressCopy.textContent = `${completed} of 7 gates complete`;
  progressSegments.forEach((segment, index) => {
    delete segment.dataset.complete;
    delete segment.dataset.current;
    if (gateIsComplete(index)) segment.dataset.complete = "";
    if (index === state.gate) segment.dataset.current = "";
  });
  gateButtons.forEach((button, index) => {
    if (gateIsComplete(index)) button.dataset.complete = "";
    else delete button.dataset.complete;
  });
}

function activateGate(index, moveFocus = true) {
  state.gate = Math.max(0, Math.min(index, panels.length - 1));
  panels.forEach((panel, panelIndex) => {
    panel.hidden = panelIndex !== state.gate;
  });
  gateButtons.forEach((button, buttonIndex) => {
    if (buttonIndex === state.gate) button.setAttribute("aria-current", "step");
    else button.removeAttribute("aria-current");
  });
  const previous = document.querySelector("[data-previous]");
  const next = document.querySelector("[data-next]");
  if (previous) previous.disabled = state.gate === 0;
  if (next) next.textContent = state.gate === panels.length - 1 ? "Review completion" : "Next gate";
  renderEvidence(state.gate);
  renderProgress();
  saveState(`Gate ${state.gate + 1} of 7 opened.`);
  if (moveFocus) document.querySelector("#workspace")?.focus({ preventScroll: true });
}

function restoreInputs() {
  checkboxes.forEach((checkbox) => {
    checkbox.checked = Boolean(state.checks[checkbox.dataset.check]);
  });
  noteFields.forEach((field) => {
    if (Object.hasOwn(state.notes, field.dataset.note)) field.value = state.notes[field.dataset.note];
  });
}

gateButtons.forEach((button, index) => {
  button.addEventListener("click", () => activateGate(index));
  button.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      activateGate(index);
      return;
    }
    if (!['ArrowDown', 'ArrowRight', 'ArrowUp', 'ArrowLeft'].includes(event.key)) return;
    event.preventDefault();
    const direction = event.key === 'ArrowDown' || event.key === 'ArrowRight' ? 1 : -1;
    const target = (index + direction + gateButtons.length) % gateButtons.length;
    gateButtons[target].focus();
  });
});

checkboxes.forEach((checkbox) => {
  checkbox.addEventListener("change", () => {
    state.checks[checkbox.dataset.check] = checkbox.checked;
    saveState(checkbox.checked ? "Evidence marked complete." : "Evidence reopened.");
    renderEvidence(state.gate);
    renderProgress();
  });
});

noteFields.forEach((field) => {
  field.addEventListener("input", () => {
    state.notes[field.dataset.note] = field.value;
    saveState("Notes saved locally.");
    renderEvidence(state.gate);
    renderProgress();
  });
});

document.querySelector("[data-previous]")?.addEventListener("click", () => activateGate(state.gate - 1));
document.querySelector("[data-next]")?.addEventListener("click", () => {
  if (state.gate < panels.length - 1) activateGate(state.gate + 1);
  else renderEvidence(state.gate);
});

document.querySelectorAll("[data-copy]").forEach((button) => {
  button.addEventListener("click", async () => {
    const source = document.querySelector(`[data-copy-source="${button.dataset.copy}"]`);
    if (!source) return;
    try {
      await navigator.clipboard.writeText(source.textContent.trim());
      button.textContent = "Copied";
      saveState("Template copied to clipboard.");
      window.setTimeout(() => { button.textContent = "Copy again"; }, 1200);
    } catch {
      source.focus?.();
      saveState("Clipboard unavailable. Select the template text to copy it.");
    }
  });
});

document.querySelector("[data-reset]")?.addEventListener("click", () => {
  if (!window.confirm("Clear every local note and checkmark in this workbench?")) return;
  state = structuredClone(defaultState);
  try {
    localStorage.removeItem(storageKey);
  } catch {}
  noteFields.forEach((field) => { field.value = ""; });
  checkboxes.forEach((checkbox) => { checkbox.checked = false; });
  activateGate(0);
  saveState("Local workbench data cleared.");
});

const showcaseMode = new URLSearchParams(window.location.search).has("showcase");
labelResponsiveTables();
if (showcaseMode) {
  shell.hidden = true;
  showcase.hidden = false;
} else {
  restoreInputs();
  activateGate(state.gate, false);
}
