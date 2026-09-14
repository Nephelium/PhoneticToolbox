"""One-time, bounded V2 function extraction. Never modifies the V2 source."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT.parent / 'PhoneticToolbox_v2'
REL = Path('phonetic_toolbox/gui/resources/web_praat_editor/app.js')
text = (V2 / REL).read_text(encoding='utf-8')
matches = list(re.finditer(r'^(?:async )?function (\w+)\(', text, re.M))
functions = {}
for index, match in enumerate(matches):
    end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
    block = text[match.start():end]
    # All selected functions end before the top-level event registration block.
    functions[match[1]] = block.rstrip()

names = '''clamp visibleEnd duration timeToX xToTime escapeTextGrid fmt fillGaps nonEmpty intervalCenter intervalOverlap intervalBelongsToWindow tierByName wordTierName phoneTierName wordTier phoneTier hitTest onGridMouseDown onGridMouseMove onGridMouseUp moveBoundary setBoundary deleteSelectedBoundary moveMatchingPhoneBoundary dragWord canPlaceWord finalizeRangeSelect updateSelectedIndices wordAtTime phoneIntervalsInsideWord labelsForIntervalCount relabelPhonesForWord splitPhoneAt autoPhonesForSelection ensurePhonesForWord ensurePhoneBoundariesAtWord percentile smoothArray localIntensityEnvelope mergeActiveRuns overlapLength detectIntensityBoundsForWord nearestNonEmptyBounds setWordBounds alignPhoneOuterEdgesToWord fitIntensityRange parseDictText pinyinToPhonesFallback pinyinToPhones incrementTone labIndexForWordSelection nextCopiedWordText pasteCopiedWord doSearch selectSearchResult findNext findPrev replaceCurrent replaceAll clipIntervals replacementWindows complementWindows spliceTier applyReferenceSplice'''.split()
header = '''// M12 source_ids: ORIGIN-WEBEDITOR, PENDING-DICTIONARY, SRC-PRAAT.
// V2 editing functions migrated by scripts/migrate_m12_editor.py.
// DOM, network and process-wide state replaced by per-editor injected ports.
export function createEditor(options = {}) {
  const state = {
    audioBuffer: null, textgrid: null, visibleStart: 0, visibleDuration: 3.2,
    selected: null, selectedBoundary: null, selectedIndices: [], drag: null,
    lastMouseTime: 0, dirty: false, undoStack: [], referenceTextGrid: null,
    phoneDict: null, searchResults: [], searchIndex: -1,
    labSequence: [], labWords: new Set(), copiedWord: '', copiedLabIndex: null,
    wordTierName: 'words', phoneTierName: 'phones',
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
    state.undoStack.push(structuredClone(state.textgrid.tiers));
    if(state.undoStack.length>50)state.undoStack.shift();
  }
  function normalizeTextGrid(tg) {
    if(!tg)return;
    const xmax=duration()||tg.xmax||0;
    for(const tier of tg.tiers)if(tier.intervals)tier.intervals=fillGaps(tier.intervals,xmax);
  }
'''
blocks = []
for name in names:
    block = functions[name]
    block = block.replace('document.getElementById("searchInput")', 'controls.searchInput')
    blocks.append(block)
footer = '''
  function undo(){
    if(!state.textgrid||!state.undoStack.length)return;
    state.textgrid.tiers=state.undoStack.pop();
    state.selected=null;state.selectedBoundary=null;state.selectedIndices=[];state.drag=null;
    markDirty();drawAll();setStatus('已撤销');
  }
  function editText(text){
    const selected=state.selected, tier=selected&&tierByName(selected.tier);
    if(!tier||!tier.intervals[selected.index])return;
    saveUndoState();tier.intervals[selected.index].text=text;
    if(selected.tier===wordTierName()&&text)ensurePhonesForWord(tier.intervals[selected.index]);
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
  return {state,controls,setCanvas:canvas=>{els.grid=canvas;},normalizeTextGrid,saveUndoState,undo,editText,clearText,copy,
'''
dest=ROOT/'frontend/src/modules/annotation'
dest.mkdir(parents=True,exist_ok=True)
(dest/'editor.mjs').write_text(header+'\n\n'.join(blocks)+footer+','.join(names)+'};\n}\n',encoding='utf-8')
(dest/'default.dict').write_bytes((V2/REL.parent/'default.dict').read_bytes())
sources=[REL,REL.parent/'index.html',REL.parent/'default.dict',Path('phonetic_toolbox/services/web_praat_server.py'),Path('phonetic_toolbox/services/io/lip.py'),Path('Phonetic_Export/index.html')]
manifest={'module':'M12','date':'2026-09-14','source_ids':['ORIGIN-WEBEDITOR','PENDING-DICTIONARY','SRC-PRAAT'],
          'files':[{'path':p.as_posix(),'sha256':hashlib.sha256((V2/p).read_bytes()).hexdigest()} for p in sources],
          'functions':names,'changes':['Per-editor state and injected UI ports','DOM-free model except injected canvas hit testing','Point tiers retained outside interval operations'],
          'provenance':'V2 local migration confirmed; earlier origin and dictionary redistribution license remain unresolved'}
(ROOT/'third_party/m12-migration.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Extracted',len(names),'functions; V2 unchanged.')
