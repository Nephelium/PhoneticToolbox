"use strict";

const state = {
  items: [],
  activeId: null,
  loadSequence: 0,
  navigationSequence: 0,
  loadController: null,
  audioBuffer: null,
  audioContext: null,
  source: null,
  playStartTime: 0,
  playOffset: 0,
  textgrid: null,
  visibleStart: 0,
  visibleDuration: 3.2,
  selected: null,
  selectedBoundary: null,
  copiedWord: "",
  copiedLabIndex: null,
  drag: null,
  lastMouseTime: 0,
  dirty: false,
  lipDirty: false,
  lastSavedOutput: "",
  undoStack: [],
  referenceTextGrid: null,
  waveBaseImage: null,
  specBaseImage: null,
  phoneDict: null,
  searchResults: [],
  searchIndex: -1,
  labSequence: [],
  labWords: new Set(),
  selectedIndices: [],
  lipData: null,
  lipOffset: 0,
  showLipOpen: true,
  showLipWidth: true,
  wordTierName: localStorage.getItem("pt_wordTier") || "words",
  phoneTierName: localStorage.getItem("pt_phoneTier") || "phones",
};

const els = {
  fileList: document.getElementById("fileList"),
  filter: document.getElementById("filter"),
  wave: document.getElementById("waveCanvas"),
  spectrogram: document.getElementById("spectrogramCanvas"),
  grid: document.getElementById("gridCanvas"),
  progress: document.getElementById("progressCanvas"),
  toggleSidebar: document.getElementById("toggleSidebarBtn"),
  status: document.getElementById("status"),
  play: document.getElementById("playBtn"),
  save: document.getElementById("saveBtn"),
  autoPhones: document.getElementById("autoPhonesBtn"),
  suffix: document.getElementById("suffixInput"),
  visible: document.getElementById("visibleInput"),
  referenceFile: document.getElementById("referenceFile"),
  clearReference: document.getElementById("clearReferenceBtn"),
  spliceMode: document.getElementById("spliceMode"),
  spliceStart: document.getElementById("spliceStart"),
  spliceEnd: document.getElementById("spliceEnd"),
  applySplice: document.getElementById("applySpliceBtn"),
  fitStart: document.getElementById("fitStart"),
  fitEnd: document.getElementById("fitEnd"),
  fitTrimMs: document.getElementById("fitTrimMs"),
  fitIntensity: document.getElementById("fitIntensityBtn"),
  folderPath: document.getElementById("folderPath"),
  scan: document.getElementById("scanBtn"),
  saveLip: document.getElementById("saveLipBtn"),
  lipOffsetInput: document.getElementById("lipOffsetInput"),
  lipUnit: document.getElementById("lipUnit"),
  lipLeft: document.getElementById("lipLeftBtn"),
  lipRight: document.getElementById("lipRightBtn"),
  toggleLipOpen: document.getElementById("toggleLipOpenBtn"),
  toggleLipWidth: document.getElementById("toggleLipWidthBtn"),
  wordTierInput: document.getElementById("wordTierInput"),
  phoneTierInput: document.getElementById("phoneTierInput"),
};

let lipRepeatTimer = null;
const LIP_REPEAT_DELAY = 150;
const LIP_REPEAT_RATE = 40;

function setStatus(message) {
  els.status.textContent = message;
}

function markDirty() {
  state.dirty = true;
}

function markLipDirty() {
  state.lipDirty = true;
}

function saveUndoState() {
  if (!state.textgrid) return;
  state.undoStack.push(structuredClone(state.textgrid.tiers));
  if (state.undoStack.length > 50) state.undoStack.shift();
}

function resizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(rect.width * dpr));
  const height = Math.max(1, Math.floor(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function resizeAll() {
  [els.wave, els.spectrogram, els.grid, els.progress].forEach(resizeCanvas);
  drawAll();
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function visibleEnd() {
  return state.visibleStart + state.visibleDuration;
}

function duration() {
  return state.audioBuffer ? state.audioBuffer.duration : (state.textgrid?.xmax || 0);
}

function timeToX(time, canvas) {
  return ((time - state.visibleStart) / state.visibleDuration) * canvas.width;
}

function xToTime(x, canvas) {
  return state.visibleStart + (x / canvas.width) * state.visibleDuration;
}

function parseTextGrid(text) {
  const lines = text.split(/\r?\n/);
  const tg = { xmin: 0, xmax: 0, tiers: [] };
  let seenItem = false;
  let inIntervalTier = false;
  let waitingForName = false;
  let tier = null;
  let interval = null;
  for (const raw of lines) {
    const line = raw.trim();
    if (line.startsWith("xmax =") && !seenItem) {
      tg.xmax = Number(line.split("=")[1].trim());
      continue;
    }
    if (line.startsWith("xmin =") && !seenItem) {
      tg.xmin = Number(line.split("=")[1].trim());
      continue;
    }
    if (line.startsWith("item [")) {
      seenItem = true;
      continue;
    }
    if (line.startsWith("class =")) {
      inIntervalTier = line.includes('"IntervalTier"');
      waitingForName = inIntervalTier;
      tier = null;
      interval = null;
      continue;
    }
    if (waitingForName && line.startsWith("name =")) {
      const name = quotedValue(raw);
      tier = { name, intervals: [] };
      tg.tiers.push(tier);
      waitingForName = false;
      continue;
    }
    if (!inIntervalTier || !tier) continue;
    if (line.startsWith("intervals [")) {
      interval = {};
      continue;
    }
    if (!interval) continue;
    if (line.startsWith("xmin =")) {
      interval.xmin = Number(line.split("=")[1].trim());
    } else if (line.startsWith("xmax =")) {
      interval.xmax = Number(line.split("=")[1].trim());
    } else if (line.startsWith("text =")) {
      interval.text = quotedValue(raw);
      if (Number.isFinite(interval.xmin) && Number.isFinite(interval.xmax)) {
        tier.intervals.push(interval);
      }
      interval = null;
    }
  }
  normalizeTextGrid(tg);
  return tg;
}

function quotedValue(line) {
  const first = line.indexOf('"');
  const last = line.lastIndexOf('"');
  if (first < 0 || last <= first) return "";
  return line.slice(first + 1, last).replace(/""/g, '"');
}

function escapeTextGrid(text) {
  return String(text || "").replace(/"/g, '""');
}

function serializeTextGrid(tg) {
  normalizeTextGrid(tg);
  const xmax = duration() || tg.xmax;
  const lines = [
    'File type = "ooTextFile"',
    'Object class = "TextGrid"',
    "",
    "xmin = 0",
    `xmax = ${fmt(xmax)}`,
    "tiers? <exists>",
    `size = ${tg.tiers.length}`,
    "item []:",
  ];
  tg.tiers.forEach((tier, tierIndex) => {
    const intervals = fillGaps(tier.intervals, xmax);
    lines.push(`    item [${tierIndex + 1}]:`);
    lines.push('        class = "IntervalTier"');
    lines.push(`        name = "${escapeTextGrid(tier.name)}"`);
    lines.push("        xmin = 0");
    lines.push(`        xmax = ${fmt(xmax)}`);
    lines.push(`        intervals: size = ${intervals.length}`);
    intervals.forEach((interval, intervalIndex) => {
      lines.push(`        intervals [${intervalIndex + 1}]:`);
      lines.push(`            xmin = ${fmt(interval.xmin)}`);
      lines.push(`            xmax = ${fmt(interval.xmax)}`);
      lines.push(`            text = "${escapeTextGrid(interval.text)}"`);
    });
  });
  return lines.join("\n") + "\n";
}

function fmt(value) {
  return Number(value).toFixed(6).replace(/0+$/, "").replace(/\.$/, ".0");
}

function fillGaps(intervals, xmax) {
  const out = [];
  let cursor = 0;
  [...intervals].sort((a, b) => a.xmin - b.xmin || a.xmax - b.xmax).forEach((item) => {
    let start = clamp(item.xmin, 0, xmax);
    let end = clamp(item.xmax, 0, xmax);
    if (end <= start + 1e-7) return;
    if (start > cursor + 1e-6) out.push({ xmin: cursor, xmax: start, text: "" });
    if (start < cursor) start = cursor;
    if (end > start + 1e-7) {
      out.push({ xmin: start, xmax: end, text: item.text || "" });
      cursor = end;
    }
  });
  if (cursor < xmax - 1e-6) out.push({ xmin: cursor, xmax, text: "" });
  return out.length ? out : [{ xmin: 0, xmax, text: "" }];
}

function nonEmpty(intervals) {
  return intervals.filter((item) => item.text && item.xmax > item.xmin + 1e-7);
}

function intervalCenter(item) {
  return (item.xmin + item.xmax) / 2;
}

function intervalOverlap(aStart, aEnd, bStart, bEnd) {
  return Math.max(0, Math.min(aEnd, bEnd) - Math.max(aStart, bStart));
}

function intervalBelongsToWindow(item, start, end) {
  const length = Math.max(1e-7, item.xmax - item.xmin);
  return start <= intervalCenter(item) && intervalCenter(item) <= end || intervalOverlap(start, end, item.xmin, item.xmax) >= length * 0.5;
}

function normalizeTextGrid(tg) {
  const xmax = duration() || tg.xmax || 0;
  tg.tiers.forEach((tier) => {
    tier.intervals = fillGaps(tier.intervals, xmax);
  });
}

function tierByName(name) {
  return state.textgrid?.tiers.find((tier) => tier.name === name) || null;
}

function wordTierName() {
  return state.wordTierName || "words";
}

function phoneTierName() {
  return state.phoneTierName || "phones";
}

function wordTier() {
  return tierByName(state.wordTierName || "words");
}

function phoneTier() {
  return tierByName(state.phoneTierName || "phones");
}

async function loadList() {
  const request = beginItemLoad();
  const res = await fetch(`/api/list?_t=${Date.now()}`, { signal: request.signal });
  const data = await res.json();
  if (!isCurrentLoad(request)) return;
  state.items = data.items;
  renderFileList();
  if (state.items.length) {
    await loadItemSafely(state.items[0].id);
  } else if (data.root) {
    setStatus(`该文件夹下未找到 wav/TextGrid 配对：${data.root}`);
  } else {
    setStatus("请在左侧选择语料文件夹（需包含 wav 及同名 TextGrid）");
  }
}

function renderFileList() {
  const filter = els.filter.value.trim().toLowerCase();
  els.fileList.innerHTML = "";
  state.items
    .filter((item) => !filter || item.rel.toLowerCase().includes(filter))
    .forEach((item) => {
      const button = document.createElement("button");
      button.className = "file-item" + (item.id === state.activeId ? " active" : "");
      button.textContent = item.rel;
      button.title = item.rel;
      button.addEventListener("click", () => loadItemSafely(item.id));
      els.fileList.appendChild(button);
    });
}

async function savePendingChanges() {
  if (!state.activeId || !state.textgrid || (!state.dirty && !state.lipDirty)) return true;
  setStatus("正在保存当前修改…");
  try {
    if (state.dirty) {
      const textgridOk = await saveTextGrid({ suffix: "_webedit", silent: true });
      if (!textgridOk) {
        setStatus("TextGrid 保存失败，停留在当前文件");
        return false;
      }
    }
    if (state.lipDirty) {
      const lipOk = await saveLipAlignment({ silent: true });
      if (!lipOk) {
        setStatus("唇形偏移保存失败，停留在当前文件");
        return false;
      }
    }
    if (state.dirty || state.lipDirty) {
      setStatus("保存失败，停留在当前文件");
      return false;
    }
    setStatus("当前修改已保存");
    return true;
  } catch (error) {
    setStatus(`保存失败，停留在当前文件：${error}`);
    return false;
  }
}

async function loadItemSafely(id) {
  const navigation = ++state.navigationSequence;
  if (id === state.activeId) return;
  if (!(await savePendingChanges())) return;
  if (navigation !== state.navigationSequence) return;
  await loadItem(id);
}

function beginItemLoad() {
  state.loadController?.abort();
  state.loadController = new AbortController();
  return { sequence: ++state.loadSequence, signal: state.loadController.signal };
}

function isCurrentLoad(request) {
  return request.sequence === state.loadSequence && !request.signal.aborted;
}

async function loadItem(id) {
  const request = beginItemLoad();
  stopAudio();
  state.activeId = null;
  state.audioBuffer = null;
  state.textgrid = null;
  state.waveBaseImage = null;
  state.specBaseImage = null;
  state.lipData = null;
  state.lipOffset = 0;
  state.lipDirty = false;
  showLipControls(false);
  setStatus("正在加载…");
  try {
    const res = await fetch(`/api/item?id=${encodeURIComponent(id)}&_t=${Date.now()}`, { signal: request.signal });
    const data = await res.json();
    if (!isCurrentLoad(request)) return;
    if (!res.ok || data.error) throw new Error(data.error || "文件加载失败");
    const textgrid = parseTextGrid(data.textgrid);
    const buffer = await loadAudio(data.audioUrl, request.signal);
    if (!isCurrentLoad(request)) return;
    state.activeId = data.id;
    state.textgrid = textgrid;
    state.labSequence = Array.isArray(data.labWords) ? data.labWords.map((w) => String(w).trim()).filter(Boolean) : [];
    state.labWords = new Set(state.labSequence.map((w) => w.toLowerCase()));
    state.selected = null;
    state.selectedBoundary = null;
    state.selectedIndices = [];
    state.copiedLabIndex = null;
    state.undoStack = [];
    state.dirty = false;
    state.lipDirty = false;
    state.lastSavedOutput = "";
    state.visibleStart = 0;
    state.visibleDuration = Number(els.visible.value) || 3.2;
    state.audioBuffer = buffer;
    state.textgrid.xmax = buffer.duration;
    normalizeTextGrid(state.textgrid);
    loadLipData(data.id, request);
    renderFileList();
    setStatus(`${data.rel} / ${data.textgridName}`);
    drawAll();
  } catch (error) {
    if (isCurrentLoad(request)) setStatus(`加载失败：${error}`);
  }
}

async function loadAudio(url, signal) {
  const ctx = audioContext();
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error("音频加载失败");
  const data = await res.arrayBuffer();
  return await ctx.decodeAudioData(data);
}

async function loadLipData(id, request) {
  const isCurrent = () => isCurrentLoad(request) && state.activeId === id;
  try {
    const res = await fetch(`/api/lip?id=${encodeURIComponent(id)}&_t=${Date.now()}`, { signal: request.signal });
    const data = await res.json();
    if (!isCurrent()) return;
    if (!res.ok || data.error) throw new Error(data.error || "唇形加载失败");
    if (data.available) {
      state.lipData = data;
      state.lipOffset = data.offset || 0;
      state.lipDirty = false;
      showLipControls(true);
      updateLipInfo();
      drawSpectrogram();
    } else {
      state.lipData = null;
      state.lipOffset = 0;
      state.lipDirty = false;
      showLipControls(false);
    }
  } catch (err) {
    if (!isCurrent()) return;
    state.lipData = null;
    state.lipOffset = 0;
    state.lipDirty = false;
    showLipControls(false);
    setStatus(`唇形加载失败：${err}`);
  }
}

function showLipControls(visible) {
  const display = visible ? "" : "none";
  els.saveLip.style.display = display;
  els.lipOffsetInput.style.display = display;
  els.lipUnit.style.display = display;
  els.lipLeft.style.display = display;
  els.lipRight.style.display = display;
  els.toggleLipOpen.style.display = display;
  els.toggleLipWidth.style.display = display;
}

function updateLipInfo() {
  els.lipOffsetInput.value = (state.lipOffset * 1000).toFixed(0);
}

function nudgeLip(deltaMs) {
  if (!state.lipData) return;
  state.lipOffset += deltaMs / 1000;
  updateLipInfo();
  markLipDirty();
  drawSpectrogram();
}

function applyLipOffsetFromInput() {
  if (!state.lipData) return;
  const ms = parseFloat(els.lipOffsetInput.value);
  if (!Number.isFinite(ms)) { updateLipInfo(); return; }
  state.lipOffset = ms / 1000;
  markLipDirty();
  drawSpectrogram();
}

function startLipRepeat(deltaMs) {
  stopLipRepeat();
  nudgeLip(deltaMs);
  lipRepeatTimer = setTimeout(() => {
    lipRepeatTimer = setInterval(() => nudgeLip(deltaMs), LIP_REPEAT_RATE);
  }, LIP_REPEAT_DELAY);
}

function stopLipRepeat() {
  if (lipRepeatTimer) {
    clearTimeout(lipRepeatTimer);
    clearInterval(lipRepeatTimer);
    lipRepeatTimer = null;
  }
}

async function saveLipAlignment(options = {}) {
  if (!state.activeId || !state.lipData) return false;
  const savedId = state.activeId;
  const savedLipData = state.lipData;
  const savedOffset = state.lipOffset;
  try {
    const res = await fetch("/api/lip/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: savedId, offset: savedOffset }),
    });
    const data = await res.json();
    if (data.ok) {
      if (state.activeId !== savedId || state.lipData !== savedLipData) return true;
      state.lipData.offset = savedOffset;
      state.lipDirty = state.lipOffset !== savedOffset;
      if (!options.silent) {
        setStatus(`唇形偏移已保存：${(state.lipOffset * 1000).toFixed(0)}ms → ${data.path || 'timestamps.pkl'}`);
      }
      return true;
    } else {
      if (!options.silent) setStatus(data.error || "唇形偏移保存失败");
      return false;
    }
  } catch (err) {
    if (!options.silent) setStatus(`唇形偏移保存失败：${err}`);
    return false;
  }
}

function audioContext() {
  if (!state.audioContext) {
    state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
  }
  return state.audioContext;
}

function drawAll() {
  if (!state.textgrid) return;
  drawWaveform();
  drawSpectrogram();
  drawGrid();
  drawProgress();
}

function drawWaveform() {
  const canvas = els.wave;
  resizeCanvas(canvas);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!state.audioBuffer) return;
  const data = state.audioBuffer.getChannelData(0);
  const sr = state.audioBuffer.sampleRate;
  const startSample = Math.max(0, Math.floor(state.visibleStart * sr));
  const endSample = Math.min(data.length, Math.ceil(visibleEnd() * sr));
  const midY = canvas.height / 2;
  ctx.strokeStyle = "#444";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < canvas.width; x++) {
    const a = startSample + Math.floor((x / canvas.width) * (endSample - startSample));
    const b = startSample + Math.floor(((x + 1) / canvas.width) * (endSample - startSample));
    let min = 1;
    let max = -1;
    for (let i = a; i < Math.max(a + 1, b); i++) {
      const v = data[i] || 0;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    ctx.moveTo(x, midY - max * midY * 0.92);
    ctx.lineTo(x, midY - min * midY * 0.92);
  }
  ctx.stroke();
  state.waveBaseImage = ctx.getImageData(0, 0, canvas.width, canvas.height);
  drawBoundaryMarkers(ctx, canvas);
  drawTimeCursor(ctx, canvas);
}

function drawSpectrogram() {
  const canvas = els.spectrogram;
  resizeCanvas(canvas);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#f8f8f8";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!state.audioBuffer) return;
  const cols = Math.min(canvas.width, 700);
  const image = ctx.createImageData(cols, canvas.height);
  const data = state.audioBuffer.getChannelData(0);
  const sr = state.audioBuffer.sampleRate;
  const fftSize = 1024;
  const maxFreq = Math.min(5000, sr / 2);
  const maxBin = Math.floor((maxFreq / sr) * fftSize);
  const window = hann(fftSize);
  let globalMax = -Infinity;
  const spectra = [];
  for (let col = 0; col < cols; col++) {
    const t = state.visibleStart + (col / Math.max(1, cols - 1)) * state.visibleDuration;
    const center = Math.floor(t * sr);
    const re = new Float64Array(fftSize);
    const im = new Float64Array(fftSize);
    const start = center - Math.floor(fftSize / 2);
    for (let i = 0; i < fftSize; i++) {
      const sample = data[start + i] || 0;
      re[i] = sample * window[i];
    }
    fft(re, im);
    const mags = new Float64Array(maxBin + 1);
    for (let bin = 1; bin <= maxBin; bin++) {
      const mag = Math.log10(Math.hypot(re[bin], im[bin]) + 1e-8);
      mags[bin] = mag;
      if (mag > globalMax) globalMax = mag;
    }
    spectra.push(mags);
  }
  const floor = globalMax - 3.2;
  for (let col = 0; col < cols; col++) {
    const mags = spectra[col];
    for (let y = 0; y < canvas.height; y++) {
      const freqRatio = 1 - y / canvas.height;
      const bin = Math.max(1, Math.min(maxBin, Math.floor(freqRatio * maxBin)));
      const norm = clamp((mags[bin] - floor) / (globalMax - floor || 1), 0, 1);
      const shade = 255 - Math.floor(norm * 220);
      const idx = (y * cols + col) * 4;
      image.data[idx] = shade;
      image.data[idx + 1] = shade;
      image.data[idx + 2] = shade;
      image.data[idx + 3] = 255;
    }
  }
  const temp = document.createElement("canvas");
  temp.width = cols;
  temp.height = canvas.height;
  temp.getContext("2d").putImageData(image, 0, 0);
  ctx.drawImage(temp, 0, 0, canvas.width, canvas.height);
  drawIntensity(ctx, canvas);
  state.specBaseImage = ctx.getImageData(0, 0, canvas.width, canvas.height);
  drawBoundaryMarkers(ctx, canvas);
  drawTimeCursor(ctx, canvas);
}

function hann(size) {
  const out = new Float64Array(size);
  for (let i = 0; i < size; i++) out[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (size - 1));
  return out;
}

function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wlenR = Math.cos(ang);
    const wlenI = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let wr = 1;
      let wi = 0;
      for (let j = 0; j < len / 2; j++) {
        const uR = re[i + j];
        const uI = im[i + j];
        const vR = re[i + j + len / 2] * wr - im[i + j + len / 2] * wi;
        const vI = re[i + j + len / 2] * wi + im[i + j + len / 2] * wr;
        re[i + j] = uR + vR;
        im[i + j] = uI + vI;
        re[i + j + len / 2] = uR - vR;
        im[i + j + len / 2] = uI - vI;
        const nextR = wr * wlenR - wi * wlenI;
        wi = wr * wlenI + wi * wlenR;
        wr = nextR;
      }
    }
  }
}

function drawIntensity(ctx, canvas) {
  const data = state.audioBuffer.getChannelData(0);
  const sr = state.audioBuffer.sampleRate;
  const frame = Math.max(1, Math.floor(0.03 * sr));
  const points = Math.min(canvas.width, 900);
  let maxRms = 1e-9;
  const rms = new Float64Array(points);
  for (let x = 0; x < points; x++) {
    const t = state.visibleStart + (x / Math.max(1, points - 1)) * state.visibleDuration;
    const center = Math.floor(t * sr);
    const start = center - Math.floor(frame / 2);
    let sum = 0;
    for (let i = 0; i < frame; i++) {
      const v = data[start + i] || 0;
      sum += v * v;
    }
    rms[x] = Math.sqrt(sum / frame);
    if (rms[x] > maxRms) maxRms = rms[x];
  }
  ctx.strokeStyle = "#17a53a";
  ctx.lineWidth = Math.max(2, canvas.width / 900);
  ctx.beginPath();
  for (let x = 0; x < points; x++) {
    const db = 100 + 20 * Math.log10(Math.max(rms[x], 1e-10) / maxRms);
    const y = canvas.height - clamp((db - 40) / 60, 0, 1) * canvas.height;
    const px = (x / Math.max(1, points - 1)) * canvas.width;
    if (x === 0) ctx.moveTo(px, y);
    else ctx.lineTo(px, y);
  }
  ctx.stroke();
  ctx.fillStyle = "#16a036";
  ctx.fillText("强度", canvas.width - 52, 18);
  if (state.showLipOpen) drawLipOpenness(ctx, canvas);
  if (state.showLipWidth) drawLipWidth(ctx, canvas);
}

function drawLipOpenness(ctx, canvas) {
  if (!state.lipData || !state.lipData.times || !state.lipData.lipOpen) return;
  const times = state.lipData.times;
  const values = state.lipData.lipOpen;
  const offset = state.lipOffset;
  const n = times.length;
  if (n < 2) return;

  // Find the range of visible data indices
  const visStart = state.visibleStart - offset;
  const visEnd = visibleEnd() - offset;

  // Find first and last visible data points (with some padding)
  let firstIdx = 0;
  let lastIdx = n - 1;
  for (let i = 0; i < n; i++) {
    if (times[i] >= visStart) { firstIdx = Math.max(0, i - 1); break; }
  }
  for (let i = n - 1; i >= 0; i--) {
    if (times[i] <= visEnd) { lastIdx = Math.min(n - 1, i + 1); break; }
  }
  if (times[lastIdx] < visStart || times[firstIdx] > visEnd) return;

  // Compute value range for scaling
  let vMin = Infinity;
  let vMax = -Infinity;
  for (let i = firstIdx; i <= lastIdx; i++) {
    const v = values[i];
    if (Number.isFinite(v)) {
      if (v < vMin) vMin = v;
      if (v > vMax) vMax = v;
    }
  }
  if (!Number.isFinite(vMin)) return;
  const range = vMax - vMin || 1;

  ctx.strokeStyle = "#e63946";
  ctx.lineWidth = Math.max(1.5, canvas.width / 900);
  ctx.beginPath();
  let firstPoint = true;
  for (let i = firstIdx; i <= lastIdx; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) { firstPoint = true; continue; }
    const t = times[i] + offset;
    const x = timeToX(t, canvas);
    const y = canvas.height - ((v - vMin) / range) * canvas.height * 0.5 - 4;
    if (firstPoint) { ctx.moveTo(x, y); firstPoint = false; }
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.fillStyle = "#c0392b";
  ctx.font = "11px sans-serif";
  ctx.fillText("唇开度", 8, 32);
}

function drawLipWidth(ctx, canvas) {
  if (!state.lipData || !state.lipData.times || !state.lipData.lipWidth) return;
  const times = state.lipData.times;
  const values = state.lipData.lipWidth;
  const offset = state.lipOffset;
  const n = times.length;
  if (n < 2) return;

  const visStart = state.visibleStart - offset;
  const visEnd = visibleEnd() - offset;

  let firstIdx = 0;
  let lastIdx = n - 1;
  for (let i = 0; i < n; i++) {
    if (times[i] >= visStart) { firstIdx = Math.max(0, i - 1); break; }
  }
  for (let i = n - 1; i >= 0; i--) {
    if (times[i] <= visEnd) { lastIdx = Math.min(n - 1, i + 1); break; }
  }
  if (times[lastIdx] < visStart || times[firstIdx] > visEnd) return;

  let vMin = Infinity;
  let vMax = -Infinity;
  for (let i = firstIdx; i <= lastIdx; i++) {
    const v = values[i];
    if (Number.isFinite(v)) {
      if (v < vMin) vMin = v;
      if (v > vMax) vMax = v;
    }
  }
  if (!Number.isFinite(vMin)) return;
  const range = vMax - vMin || 1;

  ctx.strokeStyle = "#1a6fc4";
  ctx.lineWidth = Math.max(1.5, canvas.width / 900);
  ctx.beginPath();
  let firstPoint = true;
  for (let i = firstIdx; i <= lastIdx; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) { firstPoint = true; continue; }
    const t = times[i] + offset;
    const x = timeToX(t, canvas);
    const y = canvas.height - ((v - vMin) / range) * canvas.height * 0.5 - 4;
    if (firstPoint) { ctx.moveTo(x, y); firstPoint = false; }
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.fillStyle = "#1a5fa0";
  ctx.font = "11px sans-serif";
  ctx.fillText("唇宽", 64, 32);
}

function redrawDragOverlay() {
  for (const [canvas, baseImage] of [[els.wave, state.waveBaseImage], [els.spectrogram, state.specBaseImage]]) {
    if (!baseImage) continue;
    resizeCanvas(canvas);
    const ctx = canvas.getContext("2d");
    if (canvas.width === baseImage.width && canvas.height === baseImage.height) {
      ctx.putImageData(baseImage, 0, 0);
    } else {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = canvas === els.wave ? "#fff" : "#f8f8f8";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    drawBoundaryMarkers(ctx, canvas);
    drawTimeCursor(ctx, canvas);
  }
  drawGrid();
}

function drawTimeCursor(ctx, canvas) {
  if (!state.audioContext || !state.source) return;
  const t = state.playOffset + (state.audioContext.currentTime - state.playStartTime);
  if (t < state.visibleStart || t > visibleEnd()) return;
  const x = timeToX(t, canvas);
  ctx.strokeStyle = "#e63946";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(x, 0);
  ctx.lineTo(x, canvas.height);
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawBoundaryMarkers(ctx, canvas) {
  const words = wordTier();
  const phones = phoneTier();
  if (!words && !phones) return;
  const wordBounds = new Set();
  if (words) {
    for (const item of words.intervals) {
      if (item.xmin >= state.visibleStart && item.xmin <= visibleEnd()) wordBounds.add(item.xmin);
      if (item.xmax >= state.visibleStart && item.xmax <= visibleEnd()) wordBounds.add(item.xmax);
    }
  }
  const phoneBounds = new Set();
  if (phones) {
    for (const item of phones.intervals) {
      if (item.xmin >= state.visibleStart && item.xmin <= visibleEnd()) phoneBounds.add(item.xmin);
      if (item.xmax >= state.visibleStart && item.xmax <= visibleEnd()) phoneBounds.add(item.xmax);
    }
  }
  const tolerance = 0.01;
  ctx.lineWidth = Math.max(1.25, canvas.width / 1200);
  ctx.setLineDash([6, 5]);
  for (const t of wordBounds) {
    const x = timeToX(t, canvas);
    ctx.strokeStyle = "rgba(220, 24, 36, 0.72)";
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
  }
  ctx.setLineDash([3, 5]);
  for (const t of phoneBounds) {
    const nearWord = [...wordBounds].some((wt) => Math.abs(wt - t) < tolerance);
    if (nearWord) continue;
    const x = timeToX(t, canvas);
    ctx.strokeStyle = "rgba(220, 24, 36, 0.52)";
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
  }
  ctx.setLineDash([]);
}

function drawGrid() {
  const canvas = els.grid;
  resizeCanvas(canvas);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const words = wordTier();
  const phones = phoneTier();
  const wordY = 0;
  const phoneY = canvas.height * 0.48;
  ctx.strokeStyle = "#222";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, phoneY);
  ctx.lineTo(canvas.width, phoneY);
  ctx.stroke();
  drawTier(ctx, canvas, words, wordY, phoneY, wordTierName());
  drawTier(ctx, canvas, phones, phoneY, canvas.height, phoneTierName());
  if (state.drag && (state.drag.mode === "rangeSelect" || state.drag.mode === "pendingRangeSelect")) {
    const t1 = Math.min(state.drag.startTime, state.drag.endTime || state.drag.startTime);
    const t2 = Math.max(state.drag.startTime, state.drag.endTime || state.drag.startTime);
    if (t2 > t1) {
      const x1 = timeToX(t1, canvas);
      const x2 = timeToX(t2, canvas);
      ctx.fillStyle = "rgba(24, 169, 210, 0.18)";
      ctx.fillRect(x1, wordY, x2 - x1, phoneY - wordY);
      ctx.strokeStyle = "rgba(24, 169, 210, 0.6)";
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, wordY, x2 - x1, phoneY - wordY);
      ctx.lineWidth = 1;
    }
  }
  if (state.selectedBoundary && !state.drag) {
    const sb = state.selectedBoundary;
    const tier = tierByName(sb.tier);
    if (tier) {
      const x = timeToX(sb.time, canvas);
      const top = sb.tier === wordTierName() ? wordY : phoneY;
      const bottom = sb.tier === wordTierName() ? phoneY : canvas.height;
      ctx.strokeStyle = "#e63946";
      ctx.lineWidth = Math.max(1, canvas.width / 1500);
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.lineWidth = 1;
    }
  }
  ctx.fillStyle = "#244f5b";
  ctx.font = "bold 18px Georgia, serif";
  ctx.textBaseline = "bottom";
  ctx.fillText(`${state.visibleStart.toFixed(6)} s`, 28, canvas.height - 3);
  ctx.fillText(`${visibleEnd().toFixed(6)} s`, canvas.width - 110, canvas.height - 3);
}

function drawProgress() {
  const canvas = els.progress;
  resizeCanvas(canvas);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const dur = duration();
  if (!dur) return;
  ctx.fillStyle = "#d5d5cc";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#bbb";
  ctx.fillRect(2, 2, canvas.width - 4, canvas.height - 4);
  const winStart = (state.visibleStart / dur) * canvas.width;
  const winEnd = (visibleEnd() / dur) * canvas.width;
  ctx.fillStyle = "rgba(24, 169, 210, 0.5)";
  ctx.fillRect(winStart, 2, Math.max(2, winEnd - winStart), canvas.height - 4);
  ctx.strokeStyle = "#0e6070";
  ctx.lineWidth = 1;
  ctx.strokeRect(winStart, 2, Math.max(2, winEnd - winStart), canvas.height - 4);
  ctx.fillStyle = "#333";
  ctx.font = "bold 13px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(`${state.visibleStart.toFixed(1)}s – ${visibleEnd().toFixed(1)}s / ${dur.toFixed(1)}s`, canvas.width / 2, canvas.height / 2);
}

function drawTier(ctx, canvas, tier, top, bottom, name) {
  if (!tier) return;
  const selected = state.selected;
  for (let i = 0; i < tier.intervals.length; i++) {
    const item = tier.intervals[i];
    if (item.xmax < state.visibleStart || item.xmin > visibleEnd()) continue;
    const x1 = timeToX(item.xmin, canvas);
    const x2 = timeToX(item.xmax, canvas);
    const isMultiSelected = name === wordTierName() && state.selectedIndices.includes(i);
    if (selected && selected.tier === name && selected.index === i) {
      ctx.fillStyle = "rgba(255, 230, 0, 0.45)";
      ctx.fillRect(x1, top, x2 - x1, bottom - top);
    } else if (isMultiSelected) {
      ctx.fillStyle = "rgba(255, 220, 0, 0.32)";
      ctx.fillRect(x1, top, x2 - x1, bottom - top);
    } else if (name === wordTierName() && item.text && state.labWords.size > 0 && state.labWords.has(item.text.toLowerCase())) {
      ctx.fillStyle = "rgba(120, 200, 255, 0.35)";
      ctx.fillRect(x1, top, x2 - x1, bottom - top);
    }
    ctx.strokeStyle = item.text ? "#00657a" : "rgba(0, 101, 122, 0.42)";
    ctx.lineWidth = item.text ? Math.max(1, canvas.width / 1500) : Math.max(0.75, canvas.width / 2200);
    ctx.beginPath();
    ctx.moveTo(x1, top);
    ctx.lineTo(x1, bottom);
    ctx.moveTo(x2, top);
    ctx.lineTo(x2, bottom);
    ctx.stroke();
    if (item.text) {
      ctx.fillStyle = "#0e6070";
      ctx.font = name === wordTierName() ? `${Math.max(18, canvas.height * 0.09)}px Georgia, serif` : `${Math.max(16, canvas.height * 0.08)}px Georgia, serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(item.text, (x1 + x2) / 2, (top + bottom) / 2, Math.max(20, x2 - x1 - 8));
    }
  }
}

function hitTest(event) {
  const rect = els.grid.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const x = (event.clientX - rect.left) * dpr;
  const y = (event.clientY - rect.top) * dpr;
  const time = xToTime(x, els.grid);
  const tierName = y < els.grid.height * 0.48 ? wordTierName() : phoneTierName();
  const tier = tierByName(tierName);
  if (!tier) return null;
  const near = state.visibleDuration * 0.006;
  for (let i = 0; i < tier.intervals.length; i++) {
    const item = tier.intervals[i];
    if (Math.abs(item.xmin - time) < near) return { tier: tierName, index: i, edge: "start", time };
    if (Math.abs(item.xmax - time) < near) return { tier: tierName, index: i, edge: "end", time };
  }
  for (let i = 0; i < tier.intervals.length; i++) {
    const item = tier.intervals[i];
    if (item.xmin <= time && time <= item.xmax) return { tier: tierName, index: i, edge: null, time };
  }
  return { tier: tierName, index: -1, edge: null, time };
}

function onGridMouseDown(event) {
  if (!state.textgrid) return;
  const hit = hitTest(event);
  if (!hit) return;
  state.lastMouseTime = hit.time;

  if (event.ctrlKey && hit.tier === wordTierName() && hit.index >= 0) {
    const words = wordTier();
    const item = words.intervals[hit.index];
    if (item.text) {
      const idx = state.selectedIndices.indexOf(hit.index);
      if (idx >= 0) {
        state.selectedIndices.splice(idx, 1);
        if (state.selectedIndices.length) {
          state.selected = { tier: wordTierName(), index: state.selectedIndices[0] };
        } else {
          state.selected = null;
        }
      } else {
        state.selectedIndices.push(hit.index);
        state.selectedIndices.sort((a, b) => a - b);
        state.selected = { tier: wordTierName(), index: hit.index };
      }
      state.selectedBoundary = null;
      drawGrid();
      return;
    }
  }

  if (hit.edge) {
    const tier = tierByName(hit.tier);
    const item = tier.intervals[hit.index];
    const boundaryTime = hit.edge === "start" ? item.xmin : item.xmax;
    state.selectedBoundary = { tier: hit.tier, time: boundaryTime };
    state.drag = {
      mode: "pendingBoundary",
      hit,
      startClientX: event.clientX,
      startClientY: event.clientY,
      originalTime: boundaryTime,
    };
    drawGrid();
    return;
  }

  if (hit.tier === wordTierName() && hit.index >= 0) {
    const words = wordTier();
    const item = words.intervals[hit.index];
    if (item.text) {
      state.selectedBoundary = null;
      const selectedIndices = state.selectedIndices.includes(hit.index)
        ? [...state.selectedIndices]
        : [hit.index];
      state.selected = { tier: wordTierName(), index: hit.index };
      state.selectedIndices = selectedIndices.length > 1 ? [...selectedIndices] : [];
      saveUndoState();
      state.drag = {
        mode: "pendingWord",
        hit,
        startMouseTime: hit.time,
        startClientX: event.clientX,
        originalIntervals: structuredClone(words.intervals),
        originalPhones: structuredClone(phoneTier()?.intervals || []),
        selectedIndices,
      };
      drawGrid();
      return;
    }
  }

  if (hit.tier === phoneTierName() && hit.index >= 0) {
    state.selectedBoundary = null;
    state.selectedIndices = [];
    state.selected = { tier: hit.tier, index: hit.index };
    state.drag = null;
    drawGrid();
    return;
  }

  state.selectedBoundary = null;
  state.drag = {
    mode: "pendingRangeSelect",
    hit,
    startTime: hit.time,
    startClientX: event.clientX,
    startClientY: event.clientY,
  };
  drawGrid();
}

function onGridMouseMove(event) {
  if (!state.drag) return;
  const hit = hitTest(event);
  if (!hit) return;
  state.lastMouseTime = hit.time;
  if (state.drag.mode === "pendingBoundary") {
    if (Math.abs(event.clientX - state.drag.startClientX) > 3) {
      saveUndoState();
      state.drag.mode = "boundary";
      moveBoundary(state.drag.hit, hit.time);
      markDirty();
      drawGrid();
    }
  } else if (state.drag.mode === "boundary") {
    moveBoundary(state.drag.hit, hit.time);
    markDirty();
    redrawDragOverlay();
  } else if (state.drag.mode === "pendingRangeSelect") {
    if (Math.abs(event.clientX - state.drag.startClientX) > 3) {
      state.drag.mode = "rangeSelect";
      state.drag.endTime = hit.time;
      markDirty();
      drawGrid();
    }
  } else if (state.drag.mode === "rangeSelect") {
    state.drag.endTime = hit.time;
    drawGrid();
  } else if (state.drag.mode === "pendingWord") {
    if (Math.abs(event.clientX - state.drag.startClientX) > 3) {
      state.drag.mode = "word";
      dragWord(state.drag, hit.time);
      markDirty();
      redrawDragOverlay();
    }
  } else if (state.drag.mode === "word") {
    dragWord(state.drag, hit.time);
    markDirty();
    redrawDragOverlay();
  }
}

function onGridMouseUp() {
  if (state.drag) {
    if (state.drag.mode === "word") {
      const words = wordTier();
      const indices = state.drag.selectedIndices || (state.selected ? [state.selected.index] : []);
      for (const idx of indices) {
        const word = words.intervals[idx];
        if (word && word.text) ensurePhoneBoundariesAtWord(word);
      }
      if (indices.length > 1) {
        state.selectedIndices = updateSelectedIndices(words);
      }
    }
    if (state.drag.mode === "rangeSelect") {
      finalizeRangeSelect(state.drag);
    }
    if (state.drag.mode === "pendingRangeSelect") {
      const hit = state.drag.hit;
      if (hit.tier === wordTierName() && hit.index >= 0) {
        state.selectedIndices = [];
        state.selected = { tier: wordTierName(), index: hit.index };
      } else {
        finalizeRangeSelect(state.drag);
      }
    }
    if (state.drag.mode === "pendingBoundary") {
      drawGrid();
    }
  }
  state.drag = null;
  normalizeTextGrid(state.textgrid);
  drawAll();
}

function moveBoundary(hit, time) {
  const tier = tierByName(hit.tier);
  const item = tier.intervals[hit.index];
  const minGap = 0.02;
  const lower = hit.edge === "start" ? (tier.intervals[hit.index - 1]?.xmin ?? 0) + minGap : item.xmin + minGap;
  const upper = hit.edge === "start" ? item.xmax - minGap : (tier.intervals[hit.index + 1]?.xmax ?? duration()) - minGap;
  const newTime = clamp(time, lower, upper);
  const oldTime = hit.edge === "start" ? item.xmin : item.xmax;
  setBoundary(tier, hit.index, hit.edge, newTime);
  if (hit.tier === wordTierName()) moveMatchingPhoneBoundary(oldTime, newTime);
}

function setBoundary(tier, index, edge, time) {
  const item = tier.intervals[index];
  if (edge === "start") {
    if (tier.intervals[index - 1]) tier.intervals[index - 1].xmax = time;
    item.xmin = time;
  } else {
    item.xmax = time;
    if (tier.intervals[index + 1]) tier.intervals[index + 1].xmin = time;
  }
}

function deleteSelectedBoundary() {
  const sb = state.selectedBoundary;
  if (!sb) return;
  const tier = tierByName(sb.tier);
  if (!tier) return;
  const threshold = 0.005;
  let leftIndex = -1;
  let rightIndex = -1;
  for (let i = 0; i < tier.intervals.length; i++) {
    if (Math.abs(tier.intervals[i].xmax - sb.time) < threshold) leftIndex = i;
    if (Math.abs(tier.intervals[i].xmin - sb.time) < threshold) rightIndex = i;
  }
  if (leftIndex < 0 || rightIndex < 0 || leftIndex === rightIndex) return;
  const left = tier.intervals[leftIndex];
  const right = tier.intervals[rightIndex];
  const merged = {
    xmin: left.xmin,
    xmax: right.xmax,
    text: (left.text || right.text) ? [left.text, right.text].filter(Boolean).join(" ") : "",
  };
  tier.intervals.splice(Math.min(leftIndex, rightIndex), 2, merged);
  tier.intervals = fillGaps(tier.intervals, duration());
  const mergedIndex = tier.intervals.findIndex((item) => Math.abs(item.xmin - merged.xmin) < threshold && Math.abs(item.xmax - merged.xmax) < threshold);
  state.selected = mergedIndex >= 0 ? { tier: sb.tier, index: mergedIndex } : null;
  state.selectedBoundary = null;
  markDirty();
  drawAll();
}

function moveMatchingPhoneBoundary(oldTime, newTime) {
  const phones = phoneTier();
  if (!phones) return;
  let best = null;
  let bestDistance = 0.04;
  phones.intervals.forEach((item, index) => {
    [["start", item.xmin], ["end", item.xmax]].forEach(([edge, value]) => {
      const distance = Math.abs(value - oldTime);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = { index, edge };
      }
    });
  });
  if (best) setBoundary(phones, best.index, best.edge, newTime);
}

function dragWord(drag, mouseTime) {
  const words = wordTier();
  const phones = phoneTier();
  const originalWords = structuredClone(drag.originalIntervals);
  const originalPhones = structuredClone(drag.originalPhones);
  const indices = drag.selectedIndices && drag.selectedIndices.length > 1
    ? drag.selectedIndices
    : [drag.hit.index];

  const delta = mouseTime - drag.startMouseTime;
  const selectedItems = indices.map((i) => originalWords[i]).filter(Boolean);
  const minStart = Math.min(...selectedItems.map((w) => w.xmin));
  const maxEnd = Math.max(...selectedItems.map((w) => w.xmax));
  const totalLen = maxEnd - minStart;
  const clampedDelta = clamp(delta, -minStart, duration() - totalLen - minStart);

  const movedWords = selectedItems.map((item) => ({
    text: item.text,
    xmin: item.xmin + clampedDelta,
    xmax: item.xmax + clampedDelta,
    oldStart: item.xmin,
    oldEnd: item.xmax,
  }));

  const nonSelected = originalWords.filter((w, i) => w.text && w.xmax > w.xmin + 1e-7 && !indices.includes(i));
  for (const mw of movedWords) {
    if (!canPlaceWord(nonSelected, -1, mw.xmin, mw.xmax)) return;
  }

  const keptWords = originalWords.filter((w, i) => w.text && w.xmax > w.xmin + 1e-7 && !indices.includes(i));
  words.intervals = fillGaps([...keptWords, ...movedWords.map((mw) => ({ xmin: mw.xmin, xmax: mw.xmax, text: mw.text }))], duration());

  const newIndices = [];
  for (const mw of movedWords) {
    const idx = words.intervals.findIndex(
      (w) => w.text === mw.text && Math.abs(w.xmin - mw.xmin) < 1e-5 && Math.abs(w.xmax - mw.xmax) < 1e-5,
    );
    if (idx >= 0) newIndices.push(idx);
  }
  if (newIndices.length) {
    state.selected = { tier: wordTierName(), index: newIndices[0] };
    if (newIndices.length > 1) state.selectedIndices = newIndices.sort((a, b) => a - b);
    else state.selectedIndices = [];
  }

  if (phones) {
    const allMovingPhones = [];
    for (const { oldStart, oldEnd, xmin: newStart, xmax: newEnd } of movedWords) {
      const moving = nonEmpty(originalPhones)
        .filter((phone) => intervalBelongsToWindow(phone, oldStart, oldEnd))
        .map((phone) => ({
          xmin: newStart + (phone.xmin - oldStart),
          xmax: newStart + (phone.xmax - oldStart),
          text: phone.text,
        }));
      if (moving.length) {
        moving[0].xmin = newStart;
        moving[moving.length - 1].xmax = newEnd;
      }
      allMovingPhones.push(...moving);
    }
    const keptPhones = nonEmpty(originalPhones).filter(
      (phone) => !movedWords.some((mw) => intervalBelongsToWindow(phone, mw.oldStart, mw.oldEnd)),
    );
    phones.intervals = fillGaps([...keptPhones, ...allMovingPhones], duration());
  }
}

function canPlaceWord(intervals, index, start, end) {
  return intervals.every((item, i) => {
    if (i === index || !item.text) return true;
    return end <= item.xmin || start >= item.xmax;
  });
}

function finalizeRangeSelect(drag) {
  const t1 = Math.min(drag.startTime, drag.endTime || drag.startTime);
  const t2 = Math.max(drag.startTime, drag.endTime || drag.startTime);
  els.fitStart.value = t1.toFixed(3);
  els.fitEnd.value = t2.toFixed(3);
  const words = wordTier();
  if (!words) return;
  const sel = [];
  for (let i = 0; i < words.intervals.length; i++) {
    const w = words.intervals[i];
    if (w.text && w.xmax > t1 && w.xmin < t2) sel.push(i);
  }
  state.selectedIndices = sel;
  if (sel.length) {
    state.selected = { tier: wordTierName(), index: sel[0] };
  } else {
    state.selected = null;
  }
}

function updateSelectedIndices(words) {
  const indices = [];
  for (let i = 0; i < words.intervals.length; i++) {
    const w = words.intervals[i];
    if (w.text && state.selectedIndices.some((oldIdx) => oldIdx === i)) {
      indices.push(i);
    }
  }
  return indices;
}

function wordAtTime(time) {
  const words = wordTier();
  if (!words) return null;
  return words.intervals.find((word) => word.text && word.xmin <= time && time <= word.xmax) || null;
}

function phoneIntervalsInsideWord(word) {
  const phones = phoneTier();
  if (!phones) return [];
  return phones.intervals
    .map((phone, index) => ({ phone, index }))
    .filter(({ phone }) => intervalBelongsToWindow(phone, word.xmin, word.xmax))
    .sort((a, b) => a.phone.xmin - b.phone.xmin || a.phone.xmax - b.phone.xmax);
}

function labelsForIntervalCount(labels, count) {
  if (count <= 0) return [];
  if (labels.length === count) return labels;
  if (labels.length < count) return [...labels, ...Array(count - labels.length).fill("")];
  const out = labels.slice(0, count - 1);
  out.push(labels.slice(count - 1).join(" "));
  return out;
}

function relabelPhonesForWord(word) {
  const labels = pinyinToPhones(word.text);
  if (!labels.length) return;
  const phones = phoneTier();
  const inside = phoneIntervalsInsideWord(word);
  const fitted = labelsForIntervalCount(labels, inside.length);
  inside.forEach(({ index }, i) => {
    phones.intervals[index].text = fitted[i] || "";
  });
}

function splitPhoneAt(time) {
  let phones = phoneTier();
  if (!phones) {
    phones = { name: phoneTierName(), intervals: [{ xmin: 0, xmax: duration(), text: "" }] };
    state.textgrid.tiers.push(phones);
  }
  const index = phones.intervals.findIndex((phone) => phone.xmin < time && time < phone.xmax);
  if (index < 0) return;
  const phone = phones.intervals[index];
  if (time - phone.xmin < 0.015 || phone.xmax - time < 0.015) return;
  phones.intervals.splice(
    index,
    1,
    { xmin: phone.xmin, xmax: time, text: phone.text || "" },
    { xmin: time, xmax: phone.xmax, text: "" },
  );
  const word = wordAtTime(time);
  if (word) relabelPhonesForWord(word);
  phones.intervals = fillGaps(phones.intervals, duration());
  markDirty();
}

function autoPhonesForSelection() {
  if (!state.selected || state.selected.tier !== wordTierName()) return;
  const words = wordTier();
  const word = words.intervals[state.selected.index];
  if (!word || !word.text) return;
  saveUndoState();
  ensurePhonesForWord(word);
  ensurePhoneBoundariesAtWord(word);
  markDirty();
  drawGrid();
}

function ensurePhonesForWord(word) {
  let phonesTier = phoneTier();
  if (!phonesTier) {
    phonesTier = { name: phoneTierName(), intervals: [{ xmin: 0, xmax: duration(), text: "" }] };
    state.textgrid.tiers.push(phonesTier);
  }
  const labels = pinyinToPhones(word.text);
  if (!labels.length) return;
  const kept = phonesTier.intervals.filter((phone) => {
    const center = (phone.xmin + phone.xmax) / 2;
    return !(word.xmin <= center && center <= word.xmax);
  });
  const step = (word.xmax - word.xmin) / labels.length;
  labels.forEach((label, index) => {
    kept.push({
      xmin: word.xmin + index * step,
      xmax: index === labels.length - 1 ? word.xmax : word.xmin + (index + 1) * step,
      text: label,
    });
  });
  phonesTier.intervals = fillGaps(kept, duration());
  ensurePhoneBoundariesAtWord(word);
}

function ensurePhoneBoundariesAtWord(word) {
  const phones = phoneTier();
  if (!phones) return;
  const threshold = 0.01;
  const hasStart = phones.intervals.some((phone) => Math.abs(phone.xmin - word.xmin) < threshold || Math.abs(phone.xmax - word.xmin) < threshold);
  const hasEnd = phones.intervals.some((phone) => Math.abs(phone.xmin - word.xmax) < threshold || Math.abs(phone.xmax - word.xmax) < threshold);
  if (!hasStart) splitPhoneAt(word.xmin);
  if (!hasEnd) splitPhoneAt(word.xmax);
}

function percentile(values, p) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = clamp((sorted.length - 1) * p, 0, sorted.length - 1);
  const lo = Math.floor(index);
  const hi = Math.ceil(index);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (index - lo);
}

function smoothArray(values, radius) {
  if (!values.length || radius <= 0) return values;
  const out = new Array(values.length);
  for (let i = 0; i < values.length; i++) {
    let sum = 0;
    let count = 0;
    for (let j = Math.max(0, i - radius); j <= Math.min(values.length - 1, i + radius); j++) {
      sum += values[j];
      count += 1;
    }
    out[i] = sum / count;
  }
  return out;
}

function localIntensityEnvelope(start, end) {
  if (!state.audioBuffer) return null;
  const sr = state.audioBuffer.sampleRate;
  const data = state.audioBuffer.getChannelData(0);
  const frame = Math.max(1, Math.floor(0.025 * sr));
  const hop = Math.max(1, Math.floor(0.005 * sr));
  const first = clamp(Math.floor(start * sr), 0, data.length - 1);
  const last = clamp(Math.ceil(end * sr), first + frame, data.length);
  const times = [];
  const rms = [];
  for (let center = first; center < last; center += hop) {
    const a = Math.max(0, center - Math.floor(frame / 2));
    const b = Math.min(data.length, a + frame);
    let sum = 0;
    for (let i = a; i < b; i++) {
      const v = data[i] || 0;
      sum += v * v;
    }
    times.push(center / sr);
    rms.push(Math.sqrt(sum / Math.max(1, b - a)));
  }
  if (rms.length < 4) return null;
  const smoothed = smoothArray(rms, 2);
  const peak = Math.max(...smoothed);
  if (!Number.isFinite(peak) || peak <= 1e-10) return null;
  const db = smoothed.map((v) => 20 * Math.log10(Math.max(v, 1e-12) / peak));
  return { times, db, hopSec: hop / sr };
}

function mergeActiveRuns(times, active, bridgeGap, minDuration) {
  const runs = [];
  let start = null;
  let last = null;
  for (let i = 0; i < active.length; i++) {
    if (active[i]) {
      if (start === null) start = times[i];
      last = times[i];
    } else if (start !== null && last !== null) {
      if (times[i] - last > bridgeGap) {
        if (last - start >= minDuration) runs.push({ start, end: last });
        start = null;
        last = null;
      }
    }
  }
  if (start !== null && last !== null && last - start >= minDuration) {
    runs.push({ start, end: last });
  }
  return runs;
}

function overlapLength(aStart, aEnd, bStart, bEnd) {
  return Math.max(0, Math.min(aEnd, bEnd) - Math.max(aStart, bStart));
}

function detectIntensityBoundsForWord(word, rangeStart, rangeEnd, trimSec) {
  const searchPad = 0.22;
  const searchStart = Math.max(rangeStart, word.xmin - searchPad);
  const searchEnd = Math.min(rangeEnd, word.xmax + searchPad);
  if (searchEnd <= searchStart + 0.05) return null;
  const env = localIntensityEnvelope(searchStart, searchEnd);
  if (!env) return null;
  const finiteDb = env.db.filter((v) => Number.isFinite(v));
  if (finiteDb.length < 4) return null;
  const floorDb = percentile(finiteDb, 0.18);
  const high = Math.max(floorDb + 12, -32);
  const low = Math.max(floorDb + 8, -40);
  const activeHigh = env.db.map((v) => v >= high);
  const highRuns = mergeActiveRuns(env.times, activeHigh, 0.035, 0.045);
  if (!highRuns.length) return null;
  let best = null;
  let bestScore = -Infinity;
  const wordCenter = (word.xmin + word.xmax) / 2;
  for (const run of highRuns) {
    const score = overlapLength(word.xmin, word.xmax, run.start, run.end) * 20 - Math.abs((run.start + run.end) / 2 - wordCenter);
    if (score > bestScore) {
      bestScore = score;
      best = run;
    }
  }
  if (!best) return null;
  let startIndex = env.times.findIndex((t) => t >= best.start);
  let endIndex = env.times.findIndex((t) => t >= best.end);
  if (startIndex < 0) startIndex = 0;
  if (endIndex < 0) endIndex = env.times.length - 1;
  while (startIndex > 0 && env.db[startIndex - 1] >= low) startIndex -= 1;
  while (endIndex < env.db.length - 1 && env.db[endIndex + 1] >= low) endIndex += 1;
  const maxShift = 0.26;
  const inwardPad = trimSec;
  let newStart = Math.max(rangeStart, env.times[startIndex] + inwardPad);
  let newEnd = Math.min(rangeEnd, env.times[endIndex] + env.hopSec - inwardPad);
  if (Math.abs(newStart - word.xmin) > maxShift) newStart = word.xmin;
  if (Math.abs(newEnd - word.xmax) > maxShift) newEnd = word.xmax;
  if (newEnd <= newStart + 0.04) return null;
  return { start: newStart, end: newEnd, high, low, floorDb };
}

function nearestNonEmptyBounds(words, index) {
  let left = 0;
  let right = duration();
  for (let i = index - 1; i >= 0; i--) {
    if (words.intervals[i].text) {
      left = words.intervals[i].xmax;
      break;
    }
  }
  for (let i = index + 1; i < words.intervals.length; i++) {
    if (words.intervals[i].text) {
      right = words.intervals[i].xmin;
      break;
    }
  }
  return { left, right };
}

function setWordBounds(words, index, start, end) {
  const word = words.intervals[index];
  const oldStart = word.xmin;
  const oldEnd = word.xmax;
  word.xmin = start;
  word.xmax = end;
  if (words.intervals[index - 1] && !words.intervals[index - 1].text) words.intervals[index - 1].xmax = start;
  if (words.intervals[index + 1] && !words.intervals[index + 1].text) words.intervals[index + 1].xmin = end;
  alignPhoneOuterEdgesToWord(oldStart, oldEnd, start, end);
}

function alignPhoneOuterEdgesToWord(oldStart, oldEnd, newStart, newEnd) {
  const phones = phoneTier();
  if (!phones) return;
  const inside = [];
  for (let i = 0; i < phones.intervals.length; i++) {
    const phone = phones.intervals[i];
    if (!phone.text) continue;
    const center = (phone.xmin + phone.xmax) / 2;
    if (oldStart - 0.01 <= center && center <= oldEnd + 0.01) inside.push(i);
  }
  if (!inside.length) return;
  const first = Math.min(...inside);
  const last = Math.max(...inside);
  phones.intervals[first].xmin = newStart;
  phones.intervals[last].xmax = newEnd;
  if (phones.intervals[first - 1] && !phones.intervals[first - 1].text) phones.intervals[first - 1].xmax = newStart;
  if (phones.intervals[last + 1] && !phones.intervals[last + 1].text) phones.intervals[last + 1].xmin = newEnd;
}

function fitIntensityRange() {
  if (!state.textgrid || !state.audioBuffer) return;
  const words = wordTier();
  if (!words) return;
  const xmax = duration();
  let start = Number(els.fitStart.value);
  let end = Number(els.fitEnd.value);
  const trimMs = Number.isFinite(Number(els.fitTrimMs.value)) ? Number(els.fitTrimMs.value) : 10;
  const trimSec = clamp(trimMs, -50, 80) / 1000;
  if (!Number.isFinite(start)) start = state.visibleStart;
  if (!Number.isFinite(end)) end = visibleEnd();
  start = clamp(start, 0, xmax);
  end = clamp(end, 0, xmax);
  els.fitStart.value = start.toFixed(3);
  els.fitEnd.value = end.toFixed(3);
  els.fitTrimMs.value = String(Math.round(trimSec * 1000));
  if (end <= start + 0.05) {
    setStatus("请先给定有效的时间范围");
    return;
  }
  saveUndoState();
  let changed = 0;
  for (let i = 0; i < words.intervals.length; i++) {
    const word = words.intervals[i];
    if (!word.text || word.xmax <= start || word.xmin >= end) continue;
    const detected = detectIntensityBoundsForWord(word, start, end, trimSec);
    if (!detected) continue;
    const bounds = nearestNonEmptyBounds(words, i);
    const minGap = 0.015;
    const minDur = 0.06;
    const newStart = clamp(detected.start, bounds.left + minGap, bounds.right - minGap - minDur);
    const newEnd = clamp(detected.end, newStart + minDur, bounds.right - minGap);
    if (newEnd <= newStart + minDur) continue;
    if (Math.abs(newStart - word.xmin) < 0.003 && Math.abs(newEnd - word.xmax) < 0.003) continue;
    setWordBounds(words, i, newStart, newEnd);
    changed += 1;
  }
  normalizeTextGrid(state.textgrid);
  if (changed) {
    markDirty();
    setStatus(`强度贴合：已调整 ${changed} 个词，内收 ${Math.round(trimSec * 1000)} ms`);
  } else {
    setStatus("强度贴合：没有足够可信的边界调整");
  }
  drawAll();
}

async function loadDictionary(url) {
  try {
    const res = await fetch(url);
    const text = await res.text();
    const map = new Map();
    for (const line of text.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      const [pinyin, ...phones] = trimmed.split(/\s+/);
      if (pinyin && phones.length) map.set(pinyin.toLowerCase(), phones);
    }
    state.phoneDict = map;
    setStatus(`词典已加载：${map.size} 条`);
    return map;
  } catch {
    setStatus("内置词典加载失败（可用“词典”按钮自行上传）");
    return null;
  }
}

function parseDictText(text) {
  const map = new Map();
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const [pinyin, ...phones] = trimmed.split(/\s+/);
    if (pinyin && phones.length) map.set(pinyin.toLowerCase(), phones);
  }
  return map;
}

function pinyinToPhonesFallback(label) {
  const clean = label.trim().toLowerCase().replace("ü", "v");
  const match = clean.match(/^([a-zv]+)([1-5])$/);
  if (!match) return clean ? [clean] : [];
  let base = match[1];
  const tone = match[2];
  const initials = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s"];
  let initial = "";
  for (const candidate of initials) {
    if (base.startsWith(candidate) && base.length > candidate.length) {
      initial = candidate;
      base = base.slice(candidate.length);
      break;
    }
  }
  const phones = initial ? [initial] : [];
  if (base === "ong") phones.push(`o${tone}`, "ng");
  else if (base === "iong") phones.push(`io${tone}`, "ng");
  else if (base.endsWith("ng")) phones.push(`${base.slice(0, -2)}${tone}`, "ng");
  else if (base.endsWith("n")) phones.push(`${base.slice(0, -1)}${tone}`, "n");
  else phones.push(`${base}${tone}`);
  return phones.filter(Boolean);
}

function pinyinToPhones(label) {
  const clean = label.trim().toLowerCase();
  if (state.phoneDict) {
    const entry = state.phoneDict.get(clean);
    if (entry) return [...entry];
  }
  return pinyinToPhonesFallback(label);
}

function incrementTone(label) {
  return label.replace(/([1-5])$/, (_, tone) => String(Number(tone) >= 5 ? 1 : Number(tone) + 1));
}

function labIndexForWordSelection(wordIndex) {
  const words = wordTier();
  const item = words?.intervals[wordIndex];
  if (!item?.text || !state.labSequence.length) return null;
  const normalizedText = item.text.toLowerCase();
  let ordinal = -1;
  for (let i = 0; i <= wordIndex && i < words.intervals.length; i++) {
    if (words.intervals[i].text) ordinal += 1;
  }
  if (
    ordinal >= 0 &&
    ordinal < state.labSequence.length &&
    state.labSequence[ordinal].toLowerCase() === normalizedText
  ) {
    return ordinal;
  }
  let bestIndex = null;
  let bestDistance = Infinity;
  state.labSequence.forEach((word, index) => {
    if (word.toLowerCase() !== normalizedText) return;
    const distance = Math.abs(index - Math.max(0, ordinal));
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  });
  return bestIndex;
}

function nextCopiedWordText() {
  if (
    Number.isInteger(state.copiedLabIndex) &&
    state.copiedLabIndex + 1 >= 0 &&
    state.copiedLabIndex + 1 < state.labSequence.length
  ) {
    return state.labSequence[state.copiedLabIndex + 1];
  }
  return incrementTone(state.copiedWord);
}

function pasteCopiedWord() {
  if (!state.copiedWord || !state.selected || state.selected.tier !== wordTierName()) return;
  const words = wordTier();
  const blank = words.intervals[state.selected.index];
  if (!blank || blank.text) return;
  saveUndoState();
  const text = nextCopiedWordText();
  const phoneLabels = pinyinToPhones(text);
  const wordDur = Math.max(0.18, phoneLabels.length * 0.14);
  const gap = blank.xmin > 0.01 ? 0.03 : 0;
  const available = blank.xmax - blank.xmin - gap;
  const actualDur = Math.min(wordDur, Math.max(0.08, available));
  const oldXmax = blank.xmax;
  const oldXmin = blank.xmin;
  const newStart = oldXmin + gap;
  const newEnd = newStart + actualDur;
  const replacements = [];
  if (gap > 0) replacements.push({ xmin: oldXmin, xmax: newStart, text: "" });
  replacements.push({ xmin: newStart, xmax: newEnd, text });
  if (oldXmax - newEnd > 0.02) replacements.push({ xmin: newEnd, xmax: oldXmax, text: "" });
  words.intervals.splice(state.selected.index, 1, ...replacements);
  words.intervals = fillGaps(words.intervals, duration());
  const newIdx = words.intervals.findIndex((w) => w.text === text && Math.abs(w.xmin - newStart) < 1e-5);
  if (newIdx >= 0) {
    state.selected = { tier: wordTierName(), index: newIdx };
    state.selectedIndices = [];
    const newWord = words.intervals[newIdx];
    ensurePhonesForWord(newWord);
    ensurePhoneBoundariesAtWord(newWord);
  }
  markDirty();
  drawAll();
  setStatus(`已粘贴：${text}`);
}

function doSearch(query) {
  state.searchResults = [];
  state.searchIndex = -1;
  if (!query.trim()) {
    updateSearchInfo();
    return;
  }
  const words = wordTier();
  if (!words) return;
  const q = query.toLowerCase();
  for (let i = 0; i < words.intervals.length; i++) {
    const item = words.intervals[i];
    if (item.text && item.text.toLowerCase().includes(q)) {
      state.searchResults.push(i);
    }
  }
  if (state.searchResults.length) selectSearchResult(0);
  updateSearchInfo();
}

function selectSearchResult(index) {
  if (index < 0 || index >= state.searchResults.length) return;
  state.searchIndex = index;
  const words = wordTier();
  const wordIndex = state.searchResults[index];
  const word = words.intervals[wordIndex];
  state.selected = { tier: wordTierName(), index: wordIndex };
  state.selectedIndices = [];
  const center = word.xmin;
  state.visibleStart = clamp(center - state.visibleDuration / 2, 0, Math.max(0, duration() - state.visibleDuration));
  updateSearchInfo();
  drawAll();
}

function findNext() {
  if (!state.searchResults.length) return;
  selectSearchResult((state.searchIndex + 1) % state.searchResults.length);
}

function findPrev() {
  if (!state.searchResults.length) return;
  const next = state.searchIndex <= 0 ? state.searchResults.length - 1 : state.searchIndex - 1;
  selectSearchResult(next);
}

function replaceCurrent(text) {
  if (state.searchIndex < 0 || state.searchIndex >= state.searchResults.length) return;
  const words = wordTier();
  const wordIndex = state.searchResults[state.searchIndex];
  const word = words.intervals[wordIndex];
  saveUndoState();
  word.text = text;
  ensurePhonesForWord(word);
  ensurePhoneBoundariesAtWord(word);
  markDirty();
  drawAll();
  doSearch(document.getElementById("searchInput").value);
}

function replaceAll(text) {
  if (!state.searchResults.length) return;
  saveUndoState();
  const count = state.searchResults.length;
  const words = wordTier();
  for (const wordIndex of state.searchResults) {
    words.intervals[wordIndex].text = text;
    ensurePhonesForWord(words.intervals[wordIndex]);
    ensurePhoneBoundariesAtWord(words.intervals[wordIndex]);
  }
  markDirty();
  state.searchResults = [];
  state.searchIndex = -1;
  updateSearchInfo();
  drawAll();
  setStatus(`已替换 ${count} 处`);
}

function updateSearchInfo() {
  const el = document.getElementById("searchInfo");
  if (!state.searchResults.length) {
    el.textContent = "";
  } else {
    el.textContent = `${state.searchIndex + 1}/${state.searchResults.length}`;
  }
}

function clipIntervals(intervals, start, end) {
  return intervals
    .map((item) => ({
      xmin: Math.max(start, item.xmin),
      xmax: Math.min(end, item.xmax),
      text: item.text || "",
    }))
    .filter((item) => item.xmax > item.xmin + 1e-7);
}

function replacementWindows(mode, start, end, xmax) {
  if (mode === "before") return [[0, start]];
  if (mode === "after") return [[start, xmax]];
  if (mode === "inside") return [[start, end]];
  if (mode === "outside") return [[0, start], [end, xmax]];
  return [];
}

function complementWindows(windows, xmax) {
  const kept = [];
  let cursor = 0;
  [...windows]
    .sort((a, b) => a[0] - b[0])
    .forEach(([start, end]) => {
      if (start > cursor + 1e-7) kept.push([cursor, start]);
      cursor = Math.max(cursor, end);
    });
  if (cursor < xmax - 1e-7) kept.push([cursor, xmax]);
  return kept;
}

function spliceTier(currentTier, referenceTier, windows, xmax) {
  if (!referenceTier) return currentTier;
  const intervals = [];
  complementWindows(windows, xmax).forEach(([start, end]) => {
    intervals.push(...clipIntervals(currentTier.intervals, start, end));
  });
  windows.forEach(([start, end]) => {
    intervals.push(...clipIntervals(referenceTier.intervals, start, end));
  });
  return { name: currentTier.name, intervals: fillGaps(intervals, xmax) };
}

async function applyReferenceSplice() {
  if (!state.textgrid || !state.referenceTextGrid) return;
  saveUndoState();
  const mode = els.spliceMode.value;
  const xmax = duration();
  let start = Number(els.spliceStart.value);
  let end = Number(els.spliceEnd.value);
  if (!Number.isFinite(start)) start = state.visibleStart;
  if (!Number.isFinite(end)) end = visibleEnd();
  start = clamp(start, 0, xmax);
  end = clamp(end, 0, xmax);
  if ((mode === "inside" || mode === "outside") && end <= start) {
    setStatus("复用区间需满足 终点 > 起点");
    return;
  }
  els.spliceStart.value = start.toFixed(3);
  els.spliceEnd.value = end.toFixed(3);
  const refByName = new Map(state.referenceTextGrid.tiers.map((tier) => [tier.name, tier]));
  const windows = replacementWindows(mode, start, end, xmax);
  state.textgrid.tiers = state.textgrid.tiers.map((tier) => {
    return spliceTier(tier, refByName.get(tier.name), windows, xmax);
  });
  normalizeTextGrid(state.textgrid);
  markDirty();
  setStatus(`已复用参考标注（${mode}）`);
  drawAll();
}

async function saveTextGrid(options = {}) {
  if (!state.activeId || !state.textgrid) return false;
  const savedId = state.activeId;
  const savedTextgrid = state.textgrid;
  const textgrid = serializeTextGrid(savedTextgrid);
  const suffix = options.suffix ?? els.suffix.value;
  try {
    const res = await fetch("/api/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: savedId, suffix, textgrid }),
    });
    const data = await res.json();
    if (!data.ok) {
      setStatus(data.error || "保存失败");
      return false;
    }
    if (state.activeId !== savedId || state.textgrid !== savedTextgrid) return true;
    state.dirty = serializeTextGrid(state.textgrid) !== textgrid;
    state.lastSavedOutput = data.output;
    if (!options.silent) setStatus(`已保存：${data.output}`);

    return true;
  } catch (error) {
    if (!options.silent) setStatus(`保存失败：${error}`);
    return false;
  }
}

function togglePlay() {
  if (state.source) {
    stopAudio();
    return;
  }
  if (!state.audioBuffer) return;
  const ctx = audioContext();
  const source = ctx.createBufferSource();
  source.buffer = state.audioBuffer;
  source.connect(ctx.destination);
  state.playOffset = state.visibleStart;
  state.playStartTime = ctx.currentTime;
  source.start(0, state.playOffset);
  source.onended = () => {
    state.source = null;
    els.play.textContent = "播放";
  };
  state.source = source;
  els.play.textContent = "暂停";
  requestAnimationFrame(tick);
}

function tick() {
  if (state.source) {
    redrawDragOverlay();
    drawProgress();
    requestAnimationFrame(tick);
  }
}

function stopAudio() {
  if (state.source) {
    try {
      state.source.stop();
    } catch {
      // already stopped
    }
  }
  state.source = null;
  els.play.textContent = "播放";
}

function onWheel(event) {
  if (!state.textgrid) return;
  if (!event.shiftKey && !event.ctrlKey) return;
  event.preventDefault();
  if (event.shiftKey) {
    const delta = (event.deltaY || event.deltaX) * state.visibleDuration * 0.0015;
    state.visibleStart = clamp(state.visibleStart + delta, 0, Math.max(0, duration() - state.visibleDuration));
  } else if (event.ctrlKey) {
    const rect = event.currentTarget.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const x = (event.clientX - rect.left) * dpr;
    const anchor = xToTime(x, event.currentTarget);
    const factor = event.deltaY > 0 ? 1.18 : 0.85;
    const newDuration = clamp(state.visibleDuration * factor, 0.08, Math.max(0.1, duration()));
    const ratio = (anchor - state.visibleStart) / state.visibleDuration;
    state.visibleDuration = newDuration;
    els.visible.value = newDuration.toFixed(2);
    state.visibleStart = clamp(anchor - ratio * newDuration, 0, Math.max(0, duration() - newDuration));
  }
  drawAll();
}

function onDoubleClick(event) {
  const hit = hitTest(event);
  if (!hit || hit.index < 0) return;
  if (hit.tier === phoneTierName()) {
    saveUndoState();
    splitPhoneAt(hit.time);
    drawAll();
    return;
  }
  if (hit.tier === wordTierName()) {
    state.selected = { tier: wordTierName(), index: hit.index };
    autoPhonesForSelection();
  }
}

document.addEventListener("keydown", (event) => {
  const tag = event.target.tagName;
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || event.target.isContentEditable) return;
  if (event.altKey && (event.key === "ArrowLeft" || event.code === "ArrowLeft")) {
    nudgeLip(event.shiftKey ? -10 : -1);
    event.preventDefault();
    return;
  }
  if (event.altKey && (event.key === "ArrowRight" || event.code === "ArrowRight")) {
    nudgeLip(event.shiftKey ? 10 : 1);
    event.preventDefault();
    return;
  }
  if (event.altKey && (event.key === "Backspace" || event.code === "Backspace")) {
    if (state.selectedBoundary) {
      saveUndoState();
      deleteSelectedBoundary();
    }
    event.preventDefault();
    return;
  }
  if ((event.key === "Backspace" || event.code === "Backspace") && !event.ctrlKey && !event.altKey) {
    if (state.selectedIndices.length > 1) {
      const words = wordTier();
      if (words) {
        saveUndoState();
        for (const idx of state.selectedIndices) {
          if (idx < words.intervals.length) words.intervals[idx].text = "";
        }
        markDirty();
        drawAll();
        setStatus("已清空所选词的文本");
      }
    } else if (state.selected) {
      const tier = tierByName(state.selected.tier);
      if (tier && state.selected.index < tier.intervals.length) {
        const item = tier.intervals[state.selected.index];
        if (item.text) {
          saveUndoState();
          const oldText = item.text;
          item.text = "";
          if (state.selected.tier === wordTierName() && state.copiedWord === oldText) {
            state.copiedWord = "";
            state.copiedLabIndex = null;
          }
          markDirty();
          drawAll();
          setStatus("已清空文本");
        }
      }
    }
    event.preventDefault();
    return;
  }
  if (event.ctrlKey && event.key.toLowerCase() === "z") {
    if (state.undoStack.length === 0) return;
    state.textgrid.tiers = state.undoStack.pop();
    state.selected = null;
    state.selectedBoundary = null;
    state.selectedIndices = [];
    state.drag = null;
    markDirty();
    drawAll();
    setStatus("已撤销");
    event.preventDefault();
  } else if (event.ctrlKey && event.key.toLowerCase() === "c" && state.selected) {
    const tier = tierByName(state.selected.tier);
    const item = tier?.intervals[state.selected.index];
    if (item?.text) {
      state.copiedWord = item.text;
      state.copiedLabIndex = state.selected.tier === wordTierName() ? labIndexForWordSelection(state.selected.index) : null;
      const nextText = nextCopiedWordText();
      setStatus(`已复制：${state.copiedWord}；下次粘贴：${nextText}`);
      event.preventDefault();
    }
  } else if (event.ctrlKey && event.key.toLowerCase() === "v") {
    pasteCopiedWord();
    event.preventDefault();
  } else if (event.key === " " || event.key.toLowerCase() === "p") {
    togglePlay();
    event.preventDefault();
  } else if (event.key.length === 1 && !event.ctrlKey && !event.altKey && !event.metaKey) {
    if (state.selected && !state.drag) {
      const tier = tierByName(state.selected.tier);
      if (tier && state.selected.index < tier.intervals.length) {
        saveUndoState();
        tier.intervals[state.selected.index].text += event.key;
        if (state.selected.tier === wordTierName()) ensurePhonesForWord(tier.intervals[state.selected.index]);
        markDirty();
        drawAll();
        event.preventDefault();
      }
    }
  }
});

els.filter.addEventListener("input", renderFileList);
els.play.addEventListener("click", togglePlay);
els.save.addEventListener("click", async () => {
  const textgridOk = await saveTextGrid();
  if (textgridOk && state.lipDirty) await saveLipAlignment();
});
els.autoPhones.addEventListener("click", autoPhonesForSelection);
els.applySplice.addEventListener("click", applyReferenceSplice);
els.fitIntensity.addEventListener("click", fitIntensityRange);
document.getElementById("dictUploadBtn").addEventListener("click", () => {
  document.getElementById("dictFile").click();
});
document.getElementById("dictFile").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    const text = await file.text();
    state.phoneDict = parseDictText(text);
    setStatus(`词典已加载：${state.phoneDict.size} 条`);
  } catch {
    setStatus("词典解析失败");
  }
});
els.referenceFile.addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    const text = await file.text();
    state.referenceTextGrid = parseTextGrid(text);
    els.clearReference.style.display = "inline";
    setStatus(`参考 TextGrid：${file.name}`);
  } catch {
    setStatus("参考文件读取失败");
  }
});
els.clearReference.addEventListener("click", () => {
  state.referenceTextGrid = null;
  els.referenceFile.value = "";
  els.clearReference.style.display = "none";
  setStatus("已清除参考文件");
});
els.visible.addEventListener("change", () => {
  state.visibleDuration = clamp(Number(els.visible.value) || 3.2, 0.08, Math.max(0.1, duration()));
  state.visibleStart = clamp(state.visibleStart, 0, Math.max(0, duration() - state.visibleDuration));
  drawAll();
});
[els.wave, els.spectrogram, els.grid].forEach((canvas) => canvas.addEventListener("wheel", onWheel, { passive: false }));
els.grid.addEventListener("mousedown", onGridMouseDown);
els.grid.addEventListener("mousemove", onGridMouseMove);
window.addEventListener("mouseup", onGridMouseUp);
els.grid.addEventListener("dblclick", onDoubleClick);
window.addEventListener("resize", resizeAll);

els.progress.addEventListener("click", (event) => {
  if (!duration()) return;
  const rect = els.progress.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const x = (event.clientX - rect.left) * dpr;
  const ratio = x / els.progress.width;
  state.visibleStart = clamp(ratio * duration() - state.visibleDuration / 2, 0, Math.max(0, duration() - state.visibleDuration));
  drawAll();
});

els.toggleSidebar.addEventListener("click", () => {
  document.body.classList.toggle("sidebar-hidden");
  setTimeout(resizeAll, 100);
});

async function chooseFolder() {
  if (!(await savePendingChanges())) return "";
  setStatus("正在打开文件夹选择框…");
  try {
    const initial = els.folderPath.value.trim();
    const res = await fetch(`/api/choose-folder?initial=${encodeURIComponent(initial)}&_t=${Date.now()}`);
    const data = await res.json();
    if (data.error) {
      setStatus(data.error);
      return "";
    }
    if (data.path) {
      els.folderPath.value = data.path;
      return data.path;
    }
    setStatus("已取消选择");
    return "";
  } catch (err) {
    setStatus(`文件夹选择失败：${err}`);
    return "";
  }
}

async function scanFolder(path) {
  if (!path) return;
  const navigation = ++state.navigationSequence;
  if (!(await savePendingChanges())) return;
  if (navigation !== state.navigationSequence) return;
  const request = beginItemLoad();
  setStatus("正在扫描…");
  try {
    const res = await fetch(`/api/scan?path=${encodeURIComponent(path)}&_t=${Date.now()}`, { signal: request.signal });
    const data = await res.json();
    if (!isCurrentLoad(request)) return;
    if (data.error) { setStatus(data.error); return; }
    state.items = data.items;
    state.activeId = null;
    state.audioBuffer = null;
    state.textgrid = null;
    state.lipData = null;
    state.lipOffset = 0;
    showLipControls(false);
    stopAudio();
    renderFileList();
    setStatus(`已扫描：${data.root}（${data.items.length} 条）`);
    if (state.items.length) await loadItem(data.items[0].id);
  } catch (err) {
    if (isCurrentLoad(request)) setStatus(`扫描失败：${err}`);
  }
}

els.scan.addEventListener("click", async () => {
  let path = els.folderPath.value.trim();
  if (!path) path = await chooseFolder();
  await scanFolder(path);
});

els.folderPath.addEventListener("click", async () => {
  if (!els.folderPath.value.trim()) {
    const path = await chooseFolder();
    if (path) await scanFolder(path);
  }
});

els.folderPath.addEventListener("dblclick", async () => {
  const path = await chooseFolder();
  if (path) await scanFolder(path);
});

els.folderPath.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    els.scan.click();
  }
});

els.lipLeft.addEventListener("mousedown", (event) => {
  event.preventDefault();
  startLipRepeat(event.shiftKey ? -10 : -1);
});
els.lipLeft.addEventListener("mouseup", stopLipRepeat);
els.lipLeft.addEventListener("mouseleave", stopLipRepeat);
els.lipRight.addEventListener("mousedown", (event) => {
  event.preventDefault();
  startLipRepeat(event.shiftKey ? 10 : 1);
});
els.lipRight.addEventListener("mouseup", stopLipRepeat);
els.lipRight.addEventListener("mouseleave", stopLipRepeat);
els.lipOffsetInput.addEventListener("change", applyLipOffsetFromInput);
els.lipOffsetInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    applyLipOffsetFromInput();
    saveLipAlignment();
  }
});
els.saveLip.addEventListener("click", saveLipAlignment);

els.toggleLipOpen.addEventListener("click", () => {
  state.showLipOpen = !state.showLipOpen;
  els.toggleLipOpen.textContent = state.showLipOpen ? "唇开✓" : "唇开✗";
  drawSpectrogram();
});
els.toggleLipWidth.addEventListener("click", () => {
  state.showLipWidth = !state.showLipWidth;
  els.toggleLipWidth.textContent = state.showLipWidth ? "唇宽✓" : "唇宽✗";
  drawSpectrogram();
});

const searchInput = document.getElementById("searchInput");
const searchNextBtn = document.getElementById("searchNextBtn");
const searchPrevBtn = document.getElementById("searchPrevBtn");
const replaceInput = document.getElementById("replaceInput");
const replaceBtn = document.getElementById("replaceBtn");
const replaceAllBtn = document.getElementById("replaceAllBtn");

searchInput.addEventListener("input", () => doSearch(searchInput.value));
searchInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.shiftKey ? findPrev() : findNext();
    event.preventDefault();
  }
});
searchNextBtn.addEventListener("click", findNext);
searchPrevBtn.addEventListener("click", findPrev);
replaceBtn.addEventListener("click", () => {
  if (replaceInput.value) replaceCurrent(replaceInput.value);
});
replaceAllBtn.addEventListener("click", () => {
  if (replaceInput.value) replaceAll(replaceInput.value);
});

document.getElementById("labUploadBtn").addEventListener("click", () => {
  document.getElementById("labFile").click();
});
document.getElementById("labFile").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    const text = await file.text();
    const words = text.split(/\s+/).map((w) => w.trim()).filter(Boolean);
    state.labSequence = words;
    state.labWords = new Set(words.map((w) => w.toLowerCase()));
    document.getElementById("labClearBtn").style.display = "inline";
    setStatus(`词表已加载：${state.labSequence.length} 个词`);
    drawAll();
  } catch {
    setStatus("词表文件解析失败");
  }
});
document.getElementById("labClearBtn").addEventListener("click", () => {
  state.labSequence = [];
  state.labWords = new Set();
  state.copiedLabIndex = null;
  document.getElementById("labFile").value = "";
  document.getElementById("labClearBtn").style.display = "none";
  setStatus("已清除词表高亮");
  drawAll();
});

setInterval(() => {
  if (state.activeId && state.textgrid && (state.dirty || state.lipDirty)) {
    savePendingChanges().then((ok) => {
      if (!ok) setStatus("自动保存失败，修改仍保留在当前页面");
    }).catch((error) => setStatus(`自动保存失败：${error}`));
  }
}, 60000);

function applyTierNamesFromInputs() {
  const w = (els.wordTierInput.value || "").trim() || "words";
  const p = (els.phoneTierInput.value || "").trim() || "phones";
  state.wordTierName = w;
  state.phoneTierName = p;
  localStorage.setItem("pt_wordTier", w);
  localStorage.setItem("pt_phoneTier", p);
  state.selected = null;
  state.selectedBoundary = null;
  state.selectedIndices = [];
  drawAll();
  setStatus(`词层: ${w} / 音素层: ${p}`);
}

els.wordTierInput.value = state.wordTierName;
els.phoneTierInput.value = state.phoneTierName;
els.wordTierInput.addEventListener("change", applyTierNamesFromInputs);
els.phoneTierInput.addEventListener("change", applyTierNamesFromInputs);

loadDictionary("default.dict");
loadList().catch((error) => setStatus(String(error)));
