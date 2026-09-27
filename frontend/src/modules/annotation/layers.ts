import type {Grid,AnnotationTier,EditorState} from './editor.mjs';

export function editableTier(grid:Grid,tier:AnnotationTier):boolean {
 return !!tier.intervals&&(tier.xmin??grid.xmin)===0&&Math.abs((tier.xmax??grid.xmax)-grid.xmax)<=1e-6;
}
// Preferences only select names that actually exist in this document.
export function editingNames(grid:Grid,preferred:{word?:string;phone?:string}={}) {
 const tiers=grid.tiers.filter(t=>editableTier(grid,t)),names=tiers.map(t=>t.name);
 const match=(list:string[])=>names.find(n=>list.includes(n.toLowerCase()));
 const phone=names.includes(preferred.phone??'')?preferred.phone!:match(['phones','phonemes','音素']);
 const word=names.includes(preferred.word??'')&&preferred.word!==phone?preferred.word!:match(['syllables','words','音节','词'])??names.find(n=>n!==phone)??'';
 return {word,phone:phone&&phone!==word?phone:names.find(n=>n!==word)??''};
}
export function newIntervalTiers(grid:Grid,names:string[]):AnnotationTier[] {
 const clean=names.map(n=>n.trim());
 if(!clean.length||clean.some(n=>!n||n.length>200||/[\r\n\u0000-\u001f]/.test(n)))throw Error('请输入 1–200 字符的层名，不含换行或控制字符。');
 if(new Set(clean).size!==clean.length||clean.some(n=>grid.tiers.some(t=>t.name===n)))throw Error('层名不能重复，请为每层输入不同名称。');
 if(grid.tiers.length+clean.length>64)throw Error('TextGrid 最多支持 64 层。');
 return clean.map(name=>({name,xmin:grid.xmin,xmax:grid.xmax,intervals:[{xmin:grid.xmin,xmax:grid.xmax,text:''}]}));
}
export function editorSelection(state:EditorState):[number,number]|undefined {
 if(state.drag?.mode==='rangeSelect')return [Math.min(state.drag.startTime,state.drag.endTime),Math.max(state.drag.startTime,state.drag.endTime)];
 const tier=state.textgrid?.tiers.find(t=>t.name===state.selected?.tier);
 const intervals=state.selected?.tier===state.wordTierName&&state.selectedIndices.length>1
  ?state.selectedIndices.map(i=>tier?.intervals?.[i]).filter(i=>!!i)
  :[tier?.intervals?.[state.selected?.index??-1]].filter(i=>!!i);
 if(intervals.length)return [Math.min(...intervals.map(i=>i.xmin)),Math.max(...intervals.map(i=>i.xmax))];
}
