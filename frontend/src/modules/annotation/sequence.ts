import {uiError as displayError,uiMessage as displayMessage} from '../module-localization.ts';
import {tr} from '../../i18n/index.ts';
import type {Editor,Interval} from './editor.mjs';

// User-requested onset/rime editing. V2 dictionary-based equal subdivision stays
// available separately; these boundaries always come from the user's clicks.
export function sequenceParts(editor:Editor,label:string):string[]{
 const pinyin=label.trim().toLowerCase().match(/^(zh|ch|sh|[bpmfdtnlgkhjqxrzcs])?([a-züv]+)([0-5])$/);
 if(pinyin)return [pinyin[1],pinyin[2]+pinyin[3]].filter(Boolean);
 return editor.pinyinToPhones(label);
}
export function resetSequence(editor:Editor){
 const s=editor.state,words=editor.wordTier()?.intervals.filter(i=>i.text.trim())??[];
 let index=0;while(index<words.length&&index<s.labSequence.length&&words[index].text.toLowerCase()===s.labSequence[index].toLowerCase())index++;
 s.sequenceIndex=index;s.sequenceStart=null;
}
export type PhoneSplitMode='cursor'|'equal';
export function hasUnsplitPhones(editor:Editor,word:Interval):boolean {
 const inside=editor.phoneTier()?.intervals.filter(i=>i.xmin<word.xmax&&i.xmax>word.xmin)??[];
 return inside.length===1&&((inside[0].xmin===word.xmin&&inside[0].xmax===word.xmax)||!inside[0].text.trim());
}
export function splitInitialPhones(editor:Editor,word:Interval,clickTime:number,mode:PhoneSplitMode,labels:string[]):string {
 const phones=editor.phoneTier();
 if(!phones||!hasUnsplitPhones(editor,word))throw displayError(displayMessage('modules.annotation.text.8d731ca754de'));
 if(labels.length<2)throw displayError(displayMessage('modules.annotation.text.8ce2a5bef422'));
 const first=mode==='cursor'?Number(clickTime.toFixed(6)):word.xmin+(word.xmax-word.xmin)/labels.length;
 const times=[word.xmin,...labels.slice(1).map((_,i)=>Number((first+i*(word.xmax-first)/(labels.length-1)).toFixed(6))),word.xmax];
 if(times.some((t,i)=>!Number.isFinite(t)||(i>0&&t<=times[i-1])))throw displayError(displayMessage('modules.annotation.text.32159c4836fe'));
 const pieces=labels.map((text,i)=>({xmin:times[i],xmax:times[i+1],text}));
 const index=phones.intervals.findIndex(i=>i.xmin<word.xmax&&i.xmax>word.xmin),original=phones.intervals[index];
 if(original.xmin<word.xmin)pieces.unshift({...original,xmax:word.xmin});
 if(original.xmax>word.xmax)pieces.push({...original,xmin:word.xmax});
 editor.saveUndoState();phones.intervals.splice(index,1,...pieces);
 const s=editor.state;s.selected={tier:s.wordTierName,index:editor.wordTier()!.intervals.indexOf(word)};s.selectedIndices=[];s.selectedBoundary={tier:s.phoneTierName,time:times[1]};s.dirty=true;
 return tr('modules.annotation.text.1ba17a1e6e0c',{v0:mode==='cursor'?tr('modules.annotation.text.0a5117379b42'):tr('modules.annotation.text.efb5d4a912aa'),v1:labels.join(' / '),v2:times[1].toFixed(6)});
}
function insertEmpty(intervals:Interval[],start:number,end:number,label:string):Interval[]{
 if(intervals.some(i=>i.text.trim()&&i.xmin<end&&i.xmax>start))throw displayError(displayMessage('modules.annotation.text.e6501d83d058'));
 const kept:Interval[]=[];
 for(const item of intervals){
  if(item.xmax<=start||item.xmin>=end)kept.push({...item});
  else{if(item.xmin<start)kept.push({...item,xmax:start});if(item.xmax>end)kept.push({...item,xmin:end});}
 }
 return [...kept,{xmin:start,xmax:end,text:label}].sort((a,b)=>a.xmin-b.xmin);
}
// Ordinary editing also works after loading an existing grid, without a word list.
function editingTime(grid:{xmin:number;xmax:number},rawTime:number,audioDuration?:number):number {
 if(!Number.isFinite(rawTime)||rawTime<grid.xmin-1e-6||rawTime>grid.xmax+1e-6)throw displayError(displayMessage('modules.annotation.text.d04fbf94573d'));
 if(rawTime===audioDuration&&Math.abs(rawTime-grid.xmax)<=1e-6)return grid.xmax;
 if(rawTime<=grid.xmin)return grid.xmin;
 if(rawTime>=grid.xmax)return grid.xmax;
 return Math.max(grid.xmin,Math.min(grid.xmax,Number(rawTime.toFixed(6))));
}
export function manualDoubleClick(editor:Editor,rawTime:number):string {
 const s=editor.state,grid=s.textgrid,words=editor.wordTier(),phones=editor.tiersLinked(s.wordTierName,s.phoneTierName)?editor.phoneTier():null;
 if(!grid||!words)throw displayError(displayMessage('modules.annotation.text.b9f671b3eeef'));
 const time=editingTime(grid,rawTime,s.audioBuffer?.duration);
 if(s.sequenceStart===null){
  if(time>=grid.xmax||words.intervals.some(w=>w.text.trim()&&w.xmin<=time&&time<w.xmax))throw displayError(displayMessage('modules.annotation.text.31720e8ffee6'));
  s.sequenceStart=time;s.selected=null;s.selectedIndices=[];s.selectedBoundary=null;
  return tr('modules.annotation.text.296117306dd2',{v0:time.toFixed(6)});
 }
 const start=s.sequenceStart;if(time<=start)throw displayError(displayMessage('modules.annotation.text.f3ca6454e6e9'));
 const nextWords=insertEmpty(words.intervals,start,time,''),nextPhones=phones?insertEmpty(phones.intervals,start,time,''):undefined;
 editor.saveUndoState();words.intervals=nextWords;if(phones&&nextPhones)phones.intervals=nextPhones;
 s.sequenceStart=null;s.selected={tier:s.wordTierName,index:nextWords.findIndex(w=>w.xmin===start&&w.xmax===time)};s.selectedIndices=[];s.selectedBoundary=null;s.dirty=true;
 return tr('modules.annotation.text.edb8e400a4c0');
}
export function sequenceDoubleClick(editor:Editor,rawTime:number,continuePrevious=false,splitMode:PhoneSplitMode='cursor'):string{
 const s=editor.state,grid=s.textgrid,words=editor.wordTier(),phones=editor.tiersLinked(s.wordTierName,s.phoneTierName)?editor.phoneTier():null;
 if(!grid||!words)throw displayError(displayMessage('modules.annotation.text.7612b28c36a0'));
 const time=editingTime(grid,rawTime,s.audioBuffer?.duration);
 // An internal click edits an existing syllable and never consumes the list.
 const word=words.intervals.find(i=>i.text.trim()&&time>i.xmin&&time<i.xmax);
 if(!continuePrevious&&s.sequenceStart===null&&word){
  if(!phones)throw displayError(displayMessage('modules.annotation.text.ff36a089c24b'));
  const labels=sequenceParts(editor,word.text);
  if(labels.length<2)throw displayError(displayMessage('modules.annotation.text.0c239d3fd6e6'));
  return splitInitialPhones(editor,word,time,splitMode,labels);
 }
 if(s.sequenceIndex>=s.labSequence.length)throw displayError(s.labSequence.length?displayMessage('modules.annotation.text.90be1e7c9845'):displayMessage('modules.annotation.text.dfdffcf46f09'));
 let start=s.sequenceStart;
 if(continuePrevious){
  const previous=words.intervals.filter(i=>i.text.trim()&&i.xmax<=time).at(-1);
  if(!previous)throw displayError(displayMessage('modules.annotation.text.32e158efca36'));
  start=previous.xmax;
 }
 if(start===null){
  if(time>=grid.xmax)throw displayError(displayMessage('modules.annotation.text.7f2091ab233f'));
  s.sequenceStart=time;s.selected=null;s.selectedIndices=[];s.selectedBoundary=null;
  return tr('modules.annotation.text.9da16eed1674',{v0:time.toFixed(6),v1:s.labSequence[s.sequenceIndex]});
 }
 if(time-start<1e-6)throw displayError(displayMessage('modules.annotation.text.96cfa55e44c1'));
 const label=s.labSequence[s.sequenceIndex];
 // Prepare both tiers before mutation, including overlap validation.
 const nextWords=insertEmpty(words.intervals,start,time,label),nextPhones=phones?insertEmpty(phones.intervals,start,time,label):null;
 editor.saveUndoState();words.intervals=nextWords;if(phones&&nextPhones)phones.intervals=nextPhones;
 s.sequenceIndex++;s.sequenceStart=null;s.selected={tier:s.wordTierName,index:nextWords.findIndex(i=>i.xmin===start&&i.xmax===time)};
 s.selectedIndices=[];s.selectedBoundary=null;s.dirty=true;
 return tr('modules.annotation.text.ddd31ae6d8d6',{v0:label,v1:start.toFixed(6),v2:time.toFixed(6),v3:phones?tr('modules.annotation.text.3d92b83c1fa5'):'。'});
}
