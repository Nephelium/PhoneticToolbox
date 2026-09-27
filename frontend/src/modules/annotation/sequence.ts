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
 if(!phones||!hasUnsplitPhones(editor,word))throw Error('该音节已切分或音素外边界不同，请直接拖动边界，或在音素层双击添加边界。');
 if(labels.length<2)throw Error('该音节没有可自动切分的多个音素。');
 const first=mode==='cursor'?Number(clickTime.toFixed(6)):word.xmin+(word.xmax-word.xmin)/labels.length;
 const times=[word.xmin,...labels.slice(1).map((_,i)=>Number((first+i*(word.xmax-first)/(labels.length-1)).toFixed(6))),word.xmax];
 if(times.some((t,i)=>!Number.isFinite(t)||(i>0&&t<=times[i-1])))throw Error('切分点需在音节内部，并为每个音素保留正时长。');
 const pieces=labels.map((text,i)=>({xmin:times[i],xmax:times[i+1],text}));
 const index=phones.intervals.findIndex(i=>i.xmin<word.xmax&&i.xmax>word.xmin),original=phones.intervals[index];
 if(original.xmin<word.xmin)pieces.unshift({...original,xmax:word.xmin});
 if(original.xmax>word.xmax)pieces.push({...original,xmin:word.xmax});
 editor.saveUndoState();phones.intervals.splice(index,1,...pieces);
 const s=editor.state;s.selected={tier:s.wordTierName,index:editor.wordTier()!.intervals.indexOf(word)};s.selectedIndices=[];s.selectedBoundary={tier:s.phoneTierName,time:times[1]};s.dirty=true;
 return `已按${mode==='cursor'?'双击位置':'等分'}切分 ${labels.join(' / ')}，首个边界 ${times[1].toFixed(6)} s。`;
}
function insertEmpty(intervals:Interval[],start:number,end:number,label:string):Interval[]{
 if(intervals.some(i=>i.text.trim()&&i.xmin<end&&i.xmax>start))throw Error('新音节与已有标注重叠，请重新选择空白区间。');
 const kept:Interval[]=[];
 for(const item of intervals){
  if(item.xmax<=start||item.xmin>=end)kept.push({...item});
  else{if(item.xmin<start)kept.push({...item,xmax:start});if(item.xmax>end)kept.push({...item,xmin:end});}
 }
 return [...kept,{xmin:start,xmax:end,text:label}].sort((a,b)=>a.xmin-b.xmin);
}
// Ordinary editing also works after loading an existing grid, without a word list.
export function manualDoubleClick(editor:Editor,rawTime:number):string {
 const s=editor.state,grid=s.textgrid,words=editor.wordTier(),phones=editor.phoneTier();
 if(!grid||!words)throw Error('请先选择或创建音节层。');
 const time=Number(rawTime.toFixed(6));if(!Number.isFinite(time)||time<grid.xmin||time>grid.xmax)throw Error('点击时间超出录音范围。');
 if(s.sequenceStart===null){
  if(time>=grid.xmax||words.intervals.some(w=>w.text.trim()&&w.xmin<=time&&time<w.xmax))throw Error('请在空白音节区双击确认新标注起点。');
  s.sequenceStart=time;s.selected=null;s.selectedIndices=[];s.selectedBoundary=null;
  return `起点 ${time.toFixed(6)} s 已确定，再双击终点建立空白标注。`;
 }
 const start=s.sequenceStart;if(time<=start)throw Error('终点必须晚于起点；Esc 可取消后重选。');
 const nextWords=insertEmpty(words.intervals,start,time,''),nextPhones=phones?insertEmpty(phones.intervals,start,time,''):undefined;
 editor.saveUndoState();words.intervals=nextWords;if(phones&&nextPhones)phones.intervals=nextPhones;
 s.sequenceStart=null;s.selected={tier:s.wordTierName,index:nextWords.findIndex(w=>w.xmin===start&&w.xmax===time)};s.selectedIndices=[];s.selectedBoundary=null;s.dirty=true;
 return '已新增空白标注，可直接输入文字，或在下方文本框编辑。';
}
export function sequenceDoubleClick(editor:Editor,rawTime:number,continuePrevious=false,splitMode:PhoneSplitMode='cursor'):string{
 const s=editor.state,grid=s.textgrid,words=editor.wordTier(),phones=editor.phoneTier();
 if(!grid||!words||!phones)throw Error('需要覆盖录音全时域的音节层和音素层，请检查层名。');
 if(!Number.isFinite(rawTime)||rawTime<grid.xmin||rawTime>grid.xmax+1e-6)throw Error('点击时间超出录音范围。');
 const time=Math.min(grid.xmax,Number(rawTime.toFixed(6)));
 // An internal click edits an existing syllable and never consumes the list.
 const word=words.intervals.find(i=>i.text.trim()&&time>i.xmin&&time<i.xmax);
 if(!continuePrevious&&s.sequenceStart===null&&word){
  const labels=sequenceParts(editor,word.text);
  if(labels.length<2)throw Error('该音节没有可拆分的声母，音素层保留一段。');
  return splitInitialPhones(editor,word,time,splitMode,labels);
 }
 if(s.sequenceIndex>=s.labSequence.length)throw Error(s.labSequence.length?'词表已标完，可在“下一音节”选择继续位置。':'请先上传或粘贴拼音词表。');
 let start=s.sequenceStart;
 if(continuePrevious){
  const previous=words.intervals.filter(i=>i.text.trim()&&i.xmax<=time).at(-1);
  if(!previous)throw Error('前面还没有音节，请先普通双击确认起点。');
  start=previous.xmax;
 }
 if(start===null){
  if(time>=grid.xmax)throw Error('起点必须早于录音终点。');
  s.sequenceStart=time;s.selected=null;s.selectedIndices=[];s.selectedBoundary=null;
  return `起点 ${time.toFixed(6)} s 已确定，再双击终点生成 ${s.labSequence[s.sequenceIndex]}。Esc 取消起点。`;
 }
 if(time-start<1e-6)throw Error('终点必须晚于起点；按 Esc 可取消起点后重选。');
 const label=s.labSequence[s.sequenceIndex];
 // Prepare both tiers before mutation, including overlap validation.
 const nextWords=insertEmpty(words.intervals,start,time,label),nextPhones=insertEmpty(phones.intervals,start,time,label);
 editor.saveUndoState();words.intervals=nextWords;phones.intervals=nextPhones;
 s.sequenceIndex++;s.sequenceStart=null;s.selected={tier:s.wordTierName,index:nextWords.findIndex(i=>i.xmin===start&&i.xmax===time)};
 s.selectedIndices=[];s.selectedBoundary=null;s.dirty=true;
 return `已新增 ${label}：${start.toFixed(6)}–${time.toFixed(6)} s，音素层边界已同步。`;
}
