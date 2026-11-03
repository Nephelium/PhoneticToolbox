import {uiError as displayError,uiMessage as displayMessage} from '../module-localization.ts';
import {tr} from '../../i18n/index.ts';
// R1: translate selected syllables and their phones without scaling durations.
export function translateWords(words, phones, indices, delta, end, fillGaps) {
  if(!Number.isFinite(delta))throw displayError(displayMessage('modules.annotation.text.a70bcde1163a'));
  const selected=new Set(indices),moving=words.filter((w,i)=>selected.has(i)&&w.text.trim());
  if(!moving.length)throw displayError(displayMessage('modules.annotation.text.781b0c99100f'));
  const moved=moving.map(w=>({...w,xmin:w.xmin+delta,xmax:w.xmax+delta}));
  if(moved.some(w=>w.xmin<0||w.xmax>end))throw displayError(displayMessage('modules.annotation.text.2a446f3d0b40'));
  const kept=words.filter((w,i)=>w.text.trim()&&!selected.has(i));
  if(moved.some(w=>kept.some(k=>w.xmin<k.xmax&&w.xmax>k.xmin)))throw displayError(displayMessage('modules.annotation.text.301b30298c7c'));
  const windows=[];
  for(const w of [...moving].sort((a,b)=>a.xmin-b.xmin)){
    const last=windows.at(-1);
    if(last&&w.xmin<=last[1]+1e-6)last[1]=Math.max(last[1],w.xmax);else windows.push([w.xmin,w.xmax]);
  }
  const movingPhones=[],keptPhones=[];
  for(const phone of phones){
    const intersect=windows.filter(([a,b])=>phone.xmin<b&&phone.xmax>a);
    if(!intersect.length){if(phone.text.trim())keptPhones.push({...phone});continue;}
    if(phone.text.trim()){
      if(!intersect.some(([a,b])=>phone.xmin>=a-1e-6&&phone.xmax<=b+1e-6))throw displayError(displayMessage('modules.annotation.text.1de0bdb936c5'));
      movingPhones.push({...phone,xmin:phone.xmin+delta,xmax:phone.xmax+delta});
    }else for(const [a,b] of intersect)movingPhones.push({xmin:Math.max(a,phone.xmin)+delta,xmax:Math.min(b,phone.xmax)+delta,text:''});
  }
  if(movingPhones.some(p=>p.xmin<0||p.xmax>end||keptPhones.some(k=>p.xmin<k.xmax&&p.xmax>k.xmin)))throw displayError(displayMessage('modules.annotation.text.b9c89c1e5462'));
  const nextWords=fillGaps([...kept,...moved],end),nextPhones=fillGaps([...keptPhones,...movingPhones],end);
  return {words:nextWords,phones:nextPhones,indices:moved.map(w=>nextWords.findIndex(v=>v.text===w.text&&v.xmin===w.xmin&&v.xmax===w.xmax))};
}
