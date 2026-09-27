// M12 source_ids: ORIGIN-WEBEDITOR, PENDING-DICTIONARY, SRC-PRAAT.
// V2 editing functions migrated by scripts/migrate_m12_editor.py.
// DOM, network and process-wide state replaced by per-editor injected ports.
import {translateWords} from './movement.mjs';
import {captureAnnotation,eraseWindows,pasteIntervals} from './clipboard.mjs';
export function createEditor(options = {}) {
  let annotationClipboard=null;
  const state = {
    audioBuffer: null, textgrid: null, visibleStart: 0, visibleDuration: 3.2,
    selected: null, selectedBoundary: null, selectedIndices: [], drag: null,
    lastMouseTime: 0, dirty: false, undoStack: [], referenceTextGrid: null,
    phoneDict: null, searchResults: [], searchIndex: -1,
    labSequence: [], labWords: new Set(), copiedWord: '', copiedLabIndex: null,
    wordTierName: 'words', phoneTierName: 'phones',
    sequenceIndex: 0, sequenceStart: null,
  };
  const controls = Object.fromEntries(['fitStart','fitEnd','fitTrimMs','spliceMode','spliceStart','spliceEnd','searchInput'].map(k=>[k,{value:''}]));
  controls.fitTrimMs.value='10'; controls.spliceMode.value='outside';
  const els = {...controls, grid: null};
  const setStatus = message => options.message?.(message);
  const markDirty = () => {state.dirty=true;};
  const drawAll = () => options.changed?.();
  const drawGrid = drawAll, redrawDragOverlay = drawAll;
  const updateSearchInfo = () => {};
  function saveUndoState() {
    if(!state.textgrid)return;
    state.undoStack.push({tiers:structuredClone(state.textgrid.tiers),sequenceIndex:state.sequenceIndex,sequenceStart:state.sequenceStart,labSequence:state.labSequence});
    if(state.undoStack.length>50)state.undoStack.shift();
  }
  function normalizeTextGrid(tg) {
    if(!tg)return;
    const xmax=duration()||tg.xmax||0;
    for(const tier of tg.tiers)if(tier.intervals && [wordTierName(),phoneTierName()].includes(tier.name) && (tier.xmin??tg.xmin)===0 && Math.abs((tier.xmax??tg.xmax)-xmax)<=1e-6)tier.intervals=fillGaps(tier.intervals,tier.xmax??tg.xmax);
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

function escapeTextGrid(text) {
  return String(text || "").replace(/"/g, '""');
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

function tierByName(name) {
  return state.textgrid?.tiers.find((tier) => tier.name === name && tier.intervals && (tier.xmin??state.textgrid.xmin)===0 && Math.abs((tier.xmax??state.textgrid.xmax)-duration())<=1e-6) || null;
}

function wordTierName() {
  return state.wordTierName;
}

function phoneTierName() {
  return state.phoneTierName;
}

function wordTier() {
  return tierByName(state.wordTierName);
}

function phoneTier() {
  return tierByName(state.phoneTierName);
}

function hitTest(event) {
  const rect = els.grid.getBoundingClientRect();
  const x = (event.clientX - rect.left) * els.grid.width / (rect.width || els.grid.width);
  const y = (event.clientY - rect.top) * els.grid.height / (rect.height || els.grid.height);
  const time = xToTime(x, els.grid);
  const tierName = y < els.grid.height * 0.48 ? wordTierName() : phoneTierName();
  const tier = tierByName(tierName);
  if (!tier) return null;
  const near = state.visibleDuration * 0.006;
  let nearest=null,nearestDistance=near;
  for (let i = 0; i < tier.intervals.length; i++) {
    const item = tier.intervals[i];
    for(const [edge,at] of [['start',item.xmin],['end',item.xmax]]){const distance=Math.abs(at-time);if(distance<nearestDistance){nearestDistance=distance;nearest={tier:tierName,index:i,edge,time};}}
  }
  if(nearest)return nearest;
  for (let i = 0; i < tier.intervals.length; i++) {
    const item = tier.intervals[i];
    if (item.xmin <= time && time <= item.xmax) return { tier: tierName, index: i, edge: null, time };
  }
  return { tier: tierName, index: -1, edge: null, time };
}

function onGridMouseDown(event,forceRange=false) {
  if (!state.textgrid) return;
  const hit = hitTest(event);
  if (!hit) return;
  state.lastMouseTime = hit.time;

  if(forceRange||event.shiftKey){
    state.selectedBoundary=null;state.drag={mode:'pendingRangeSelect',hit,startTime:hit.time,startClientX:event.clientX,startClientY:event.clientY};drawGrid();return;
  }

  if (event.ctrlKey && !hit.edge && hit.tier === wordTierName() && hit.index >= 0) {
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
    beginBoundaryDrag(hit,event);
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
  if (["pendingBoundary","boundary"].includes(state.drag.mode)) {
    dragBoundaryTo(hit.time,event.clientX);
  } else if (state.drag.mode === "pendingRangeSelect") {
    if (Math.abs(event.clientX - state.drag.startClientX) > 3) {
      state.drag.mode = "rangeSelect";
      state.drag.endTime = hit.time;
      drawGrid();
    }
  } else if (state.drag.mode === "rangeSelect") {
    state.drag.endTime = hit.time;
    drawGrid();
  } else if (state.drag.mode === "pendingWord") {
    if (Math.abs(event.clientX - state.drag.startClientX) > 3) {
      saveUndoState();
      state.drag.mode = "word";
      if(dragWord(state.drag, hit.time)!==false)markDirty();
      redrawDragOverlay();
    }
  } else if (state.drag.mode === "word") {
    if(dragWord(state.drag, hit.time)!==false)markDirty();
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

// Reusable by all time plots. Words win ties at shared word/phone boundaries.
function boundaryAt(time,tolerance) {
  let result=null,distance=tolerance;
  for(const tier of [wordTier(),phoneTier()])for(const [index,item] of (tier?.intervals||[]).entries()){
    const delta=Math.abs(item.xmax-time);
    if(index<tier.intervals.length-1&&delta<distance){distance=delta;result={tier:tier.name,index,edge:'end',time};}
  }
  return result;
}
function beginBoundaryDrag(hit,event) {
  const tier=tierByName(hit.tier),item=tier?.intervals[hit.index];
  if(!item||!hit.edge)return;
  const originalTime=hit.edge==='start'?item.xmin:item.xmax;
  state.selected={tier:hit.tier,index:hit.index};state.selectedIndices=[];
  state.selectedBoundary={tier:hit.tier,time:originalTime};
  state.drag={mode:'pendingBoundary',hit,startClientX:event.clientX,originalTime,detach:!!event.ctrlKey,
    originalIntervals:structuredClone(tier.intervals),originalPhones:structuredClone(phoneTier()?.intervals||[])};
  drawGrid();
}
function dragBoundaryTo(time,clientX) {
  const drag=state.drag;if(!drag||!['pendingBoundary','boundary'].includes(drag.mode))return;
  if(drag.mode==='pendingBoundary'){
    if(Math.abs(clientX-drag.startClientX)<=3)return;
    saveUndoState();drag.mode='boundary';
  }
  if(drag.detach)detachBoundary(drag,time);else moveBoundary(drag.hit,time);
  markDirty();drawGrid();
}
function detachBoundary(drag,time) {
  const tier=tierByName(drag.hit.tier);if(!tier)return;
  tier.intervals=structuredClone(drag.originalIntervals);
  if(drag.hit.tier===wordTierName()&&phoneTier())phoneTier().intervals=structuredClone(drag.originalPhones);
  const at=drag.originalTime,rightIndex=tier.intervals.findIndex((v,i)=>i>0&&v.xmin===at);
  if(rightIndex<1)return;
  const left=tier.intervals[rightIndex-1],right=tier.intervals[rightIndex];
  const phones=drag.hit.tier===wordTierName()?phoneTier():null;
  const phoneRight=phones?.intervals.findIndex((v,i)=>i>0&&v.xmin===at)??-1;
  let lower=left.xmin+.000001,upper=right.xmax-.000001;
  if(phones&&phoneRight>0){lower=Math.max(lower,phones.intervals[phoneRight-1].xmin+.000001);upper=Math.min(upper,phones.intervals[phoneRight].xmax-.000001);}
  const moved=Number(clamp(time,lower,upper).toFixed(6));
  const separate=(target,index)=>{
    if(moved<at){target.intervals[index-1].xmax=moved;target.intervals.splice(index,0,{xmin:moved,xmax:at,text:''});}
    else if(moved>at){target.intervals[index].xmin=moved;target.intervals.splice(index,0,{xmin:at,xmax:moved,text:''});}
  };
  separate(tier,rightIndex);if(phones&&phoneRight>0)separate(phones,phoneRight);
  state.selected={tier:tier.name,index:moved>at?rightIndex+1:rightIndex-1};state.selectedIndices=[];
  state.selectedBoundary={tier:tier.name,time:moved};
}
function moveBoundary(hit, time) {
  const tier = tierByName(hit.tier);
  if (!tier || (hit.edge === 'start' && hit.index === 0) || (hit.edge === 'end' && hit.index === tier.intervals.length - 1)) return;
  const item = tier.intervals[hit.index];
  const minGap = 0.000001; // File precision only, no millisecond editing restriction.
  let lower = hit.edge === "start" ? (tier.intervals[hit.index - 1]?.xmin ?? 0) + minGap : item.xmin + minGap;
  let upper = hit.edge === "start" ? item.xmax - minGap : (tier.intervals[hit.index + 1]?.xmax ?? duration()) - minGap;
  const oldTime = hit.edge === "start" ? item.xmin : item.xmax;
  if(hit.tier===wordTierName()){
    const phones=phoneTier()?.intervals,idx=phones?.findIndex((v,i)=>i>0&&Math.abs(v.xmin-oldTime)<1e-9)??-1;
    if(phones&&idx>0){lower=Math.max(lower,phones[idx-1].xmin+minGap);upper=Math.min(upper,phones[idx].xmax-minGap);}
  }
  if (upper < lower) return;
  const newTime = Number(clamp(time, lower, upper).toFixed(6));
  setBoundary(tier, hit.index, hit.edge, newTime);
  if ((hit.edge === 'start' ? item.xmin : item.xmax) !== newTime) return;
  if (state.selectedBoundary?.tier === hit.tier && state.selectedBoundary.time === oldTime) state.selectedBoundary.time = newTime;
  if (hit.tier === wordTierName()) moveMatchingPhoneBoundary(oldTime, newTime);
}

function setBoundary(tier, index, edge, time) {
  const item = tier.intervals[index];
  if (!item || !Number.isFinite(time)) return;
  if (edge === 'start' && (index === 0 || time <= tier.intervals[index-1].xmin || time >= item.xmax)) return;
  if (edge === 'end' && (index === tier.intervals.length-1 || time <= item.xmin || time >= tier.intervals[index+1].xmax)) return;
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
  const rightIndex=tier.intervals.findIndex((item,i)=>i>0&&item.xmin===sb.time&&tier.intervals[i-1].xmax===sb.time);
  const leftIndex=rightIndex-1;
  if(rightIndex<1)return;
  const left = tier.intervals[leftIndex];
  const right = tier.intervals[rightIndex];
  const merged = {
    xmin: left.xmin,
    xmax: right.xmax,
    text: (left.text || right.text) ? [left.text, right.text].filter(Boolean).join(" ") : "",
  };
  tier.intervals.splice(Math.min(leftIndex, rightIndex), 2, merged);
  tier.intervals = fillGaps(tier.intervals, duration());
  const mergedIndex = tier.intervals.findIndex((item) => item.xmin===merged.xmin&&item.xmax===merged.xmax);
  state.selected = mergedIndex >= 0 ? { tier: sb.tier, index: mergedIndex } : null;
  state.selectedIndices = [];
  state.selectedBoundary = null;
  markDirty();
  drawAll();
}

function moveMatchingPhoneBoundary(oldTime, newTime) {
  const phones = phoneTier();
  if (!phones) return;
  let best = null;
  let bestDistance = 1e-9;
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

function applyMoved(result) {
  wordTier().intervals=result.words;
  if(phoneTier())phoneTier().intervals=result.phones;
  state.selected={tier:wordTierName(),index:result.indices[0]};
  state.selectedIndices=result.indices.length>1?result.indices:[];
  state.selectedBoundary=null;
}
function moveSelected(delta) {
  if(!state.selected||state.selected.tier!==wordTierName())throw Error('请先在音节层选中需要移动的标注。');
  const indices=state.selectedIndices.length?state.selectedIndices:[state.selected.index];
  const result=translateWords(wordTier().intervals,phoneTier()?.intervals||[],indices,delta,state.textgrid.xmax,fillGaps);
  saveUndoState();applyMoved(result);markDirty();drawAll();
}
function dragWord(drag, mouseTime) {
  const indices=drag.selectedIndices?.length?drag.selectedIndices:[drag.hit.index];
  const selected=indices.map(i=>drag.originalIntervals[i]);
  const delta=clamp(mouseTime-drag.startMouseTime,-Math.min(...selected.map(i=>i.xmin)),state.textgrid.xmax-Math.max(...selected.map(i=>i.xmax)));
  try{applyMoved(translateWords(drag.originalIntervals,drag.originalPhones,indices,delta,state.textgrid.xmax,fillGaps));return true;}
  catch(error){setStatus(error.message);return false;}
}

function canPlaceWord(intervals, index, start, end) {
  return intervals.every((item, i) => {
    if (i === index || !item.text) return true;
    return end <= item.xmin || start >= item.xmax;
  });
}

function finalizeRangeSelect(drag) {
  const t1 = Math.min(drag.startTime, drag.endTime ?? drag.startTime);
  const t2 = Math.max(drag.startTime, drag.endTime ?? drag.startTime);
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
  if(!phoneTierName())return;
  time=Number(time.toFixed(6));
  let phones = phoneTier();
  if (!phones) {
    phones = { name: phoneTierName(), intervals: [{ xmin: 0, xmax: duration(), text: "" }] };
    state.textgrid.tiers.push(phones);
  }
  const index = phones.intervals.findIndex((phone) => phone.xmin < time && time < phone.xmax);
  if (index < 0) return;
  const phone = phones.intervals[index];
  if(time<=phone.xmin||time>=phone.xmax)return;
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
  if(!phoneTierName())return;
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
  const threshold = 1e-9;
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
  if (available < 0.08) { setStatus('空白区间不足 80 ms，未粘贴。'); return; }
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
  doSearch(controls.searchInput.value);
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
  if (!!currentTier.intervals !== !!referenceTier.intervals) throw Error('参考文件同名层的类型不一致。');
  const low=currentTier.xmin??0,high=currentTier.xmax??xmax;
  if((referenceTier.xmin??0)!==low || (referenceTier.xmax??xmax)!==high)throw Error('参考文件同名层的时间范围不一致。');
  if (currentTier.points) {
    const inside = time => windows.some(([start,end]) => start <= time && time <= end);
    return {...currentTier, points:[...currentTier.points.filter(p=>!inside(p.number)),...referenceTier.points.filter(p=>inside(p.number))].map(p=>({...p})).sort((a,b)=>a.number-b.number)};
  }
  const intervals = [];
  complementWindows(windows, xmax).forEach(([start, end]) => {
    intervals.push(...clipIntervals(currentTier.intervals, start, end));
  });
  windows.forEach(([start, end]) => {
    intervals.push(...clipIntervals(referenceTier.intervals, start, end));
  });
  const filled=fillGaps(intervals,xmax).flatMap(i=>{const a=Math.max(low,i.xmin),b=Math.min(high,i.xmax);return b>a?[{...i,xmin:a,xmax:b}]:[];});
  return { ...currentTier, intervals: filled };
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
  if (!state.textgrid.tiers.some(t=>refByName.has(t.name))) throw Error('参考文件中没有同名层。');
  const windows = replacementWindows(mode, start, end, xmax);
  state.textgrid.tiers = state.textgrid.tiers.map((tier) => {
    return spliceTier(tier, refByName.get(tier.name), windows, xmax);
  });
  markDirty();
  setStatus(`已复用参考标注（${mode}）`);
  drawAll();
}
  function undo(){
    if(!state.textgrid||!state.undoStack.length)return;
    const previous=state.undoStack.pop();state.textgrid.tiers=previous.tiers;
    if(previous.labSequence===state.labSequence){state.sequenceIndex=previous.sequenceIndex;state.sequenceStart=previous.sequenceStart;}
    else state.sequenceStart=null;
    state.selected=null;state.selectedBoundary=null;state.selectedIndices=[];state.drag=null;
    markDirty();drawAll();setStatus('已撤销');
  }
  function editText(text,manualLabels){
    const selected=state.selected, tier=selected&&tierByName(selected.tier);
    if(!tier||!tier.intervals[selected.index]||tier.intervals[selected.index].text===text)return;
    saveUndoState();tier.intervals[selected.index].text=text;
    if(selected.tier===wordTierName()&&text){
      const word=tier.intervals[selected.index];
      if(manualLabels){const inside=phoneIntervalsInsideWord(word),labels=inside.length===1?[text]:labelsForIntervalCount(manualLabels,inside.length);inside.forEach(({phone},i)=>{phone.text=labels[i]||'';});}
      else ensurePhonesForWord(word);
    }
    markDirty();drawAll();
  }
  function clearText(){
    if(state.selectedIndices.length>1){saveUndoState();for(const i of state.selectedIndices)if(wordTier()?.intervals[i])wordTier().intervals[i].text='';markDirty();drawAll();}
    else editText('');
  }
  function copy(){
    const sel=state.selected,item=sel&&tierByName(sel.tier)?.intervals[sel.index];
    if(!item?.text)return;
    state.copiedWord=item.text;state.copiedLabIndex=sel.tier===wordTierName()?labIndexForWordSelection(sel.index):null;
    setStatus(`已复制：${item.text}；下次粘贴：${nextCopiedWordText()}`);
  }
  function captureSelected(){
    const sel=state.selected,tier=sel&&tierByName(sel.tier);if(!tier)throw Error('请先选中一个标注。');
    const role=sel.tier===wordTierName()?'word':'phone';
    return captureAnnotation(tier.intervals,role==='word'?phoneTier()?.intervals:null,role==='word'&&state.selectedIndices.length?state.selectedIndices:[sel.index],role);
  }
  function deleteAnnotation(){
    const clip=captureSelected(),tier=tierByName(state.selected.tier);
    const next=eraseWindows(tier.intervals,clip.windows),phones=clip.phones?eraseWindows(phoneTier().intervals,clip.windows):null;
    saveUndoState();tier.intervals=next;if(phones)phoneTier().intervals=phones;
    state.selected=null;state.selectedIndices=[];state.selectedBoundary=null;state.searchResults=[];state.searchIndex=-1;
    markDirty();drawAll();setStatus('已删除所选标注，原位置留空；Ctrl＋Z 可撤销。');
  }
  function copyAnnotation(cut=false){
    const clip=captureSelected();if(cut)deleteAnnotation();annotationClipboard=clip;
    setStatus(`已${cut?'剪切':'复制'}标注，保留原时长与内部边界。点击空白位置后 Ctrl＋V 粘贴。`);
  }
  function pasteAnnotation(time=state.lastMouseTime){
    const clip=annotationClipboard;if(!clip)throw Error('请先使用 Ctrl＋C 或 Ctrl＋X 复制或剪切标注。');
    const name=clip.role==='word'?wordTierName():phoneTierName(),tier=tierByName(name);
    if(!tier||state.selected&&state.selected.tier!==name)throw Error('请在与剪贴内容相同的标注层选择粘贴位置。');
    const next=pasteIntervals(tier.intervals,clip.intervals,time,clip.span,duration());
    if(clip.phones&&!phoneTier())throw Error('剪贴内容含音素边界，请先选择音素层。');
    const phones=clip.phones?pasteIntervals(phoneTier().intervals,clip.phones,time,clip.span,duration()):null;
    saveUndoState();tier.intervals=next;if(phones)phoneTier().intervals=phones;
    const start=Number(time.toFixed(6)),indices=clip.intervals.map(i=>next.findIndex(n=>n.xmin===Number((start+i.xmin).toFixed(6))&&n.xmax===Number((start+i.xmax).toFixed(6))));
    state.selected={tier:name,index:indices[0]};state.selectedIndices=clip.role==='word'&&indices.length>1?indices:[];state.selectedBoundary=null;state.searchResults=[];state.searchIndex=-1;
    markDirty();drawAll();setStatus('已粘贴标注，原文字、时长和内部边界保持。');
  }
  return {state,controls,setCanvas:canvas=>{els.grid=canvas;},normalizeTextGrid,saveUndoState,undo,editText,clearText,copy,copyAnnotation,deleteAnnotation,pasteAnnotation,moveSelected,boundaryAt,beginBoundaryDrag,dragBoundaryTo,
clamp,visibleEnd,duration,timeToX,xToTime,escapeTextGrid,fmt,fillGaps,nonEmpty,intervalCenter,intervalOverlap,intervalBelongsToWindow,tierByName,wordTierName,phoneTierName,wordTier,phoneTier,hitTest,onGridMouseDown,onGridMouseMove,onGridMouseUp,moveBoundary,setBoundary,deleteSelectedBoundary,moveMatchingPhoneBoundary,dragWord,canPlaceWord,finalizeRangeSelect,updateSelectedIndices,wordAtTime,phoneIntervalsInsideWord,labelsForIntervalCount,relabelPhonesForWord,splitPhoneAt,autoPhonesForSelection,ensurePhonesForWord,ensurePhoneBoundariesAtWord,percentile,smoothArray,localIntensityEnvelope,mergeActiveRuns,overlapLength,detectIntensityBoundsForWord,nearestNonEmptyBounds,setWordBounds,alignPhoneOuterEdgesToWord,fitIntensityRange,parseDictText,pinyinToPhonesFallback,pinyinToPhones,incrementTone,labIndexForWordSelection,nextCopiedWordText,pasteCopiedWord,doSearch,selectSearchResult,findNext,findPrev,replaceCurrent,replaceAll,clipIntervals,replacementWindows,complementWindows,spliceTier,applyReferenceSplice};
}
