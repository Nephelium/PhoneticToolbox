import {uiError as displayError,uiMessage as displayMessage} from '../module-localization.ts';
import {tr} from '../../i18n/index.ts';
import type {Grid,AnnotationTier,Interval} from './editor.mjs';
import type {AudioAsset} from '../../platform/types.ts';

// SRC-PRAAT: labelled and short text formats, doubled quotes and multiline labels.
// Point tiers remain intact even though the two active editing rows are intervals.
export function parseGrid(text:string,allowNoTiers=false):Grid {
  if(text.length>2_000_000)throw displayError(displayMessage('modules.annotation.text.ba88fa9fbcc0'));
  const tokens:(string|number)[]=[];let i=0;
  while(i<text.length){
    const c=text[i];
    if(c==='"'){
      i++;let value='',closed=false;
      while(i<text.length){if(text[i]==='"'){if(text[i+1]==='"'){value+='"';i+=2;continue;}i++;closed=true;break;}value+=text[i++];}
      if(!closed)throw displayError(displayMessage('modules.annotation.text.f95b633d5ace'));tokens.push(value);
    }else if(text.startsWith('<exists>',i)){tokens.push('<exists>');i+=8;}
    else if(/[+\-.0-9]/.test(c)){
      const match=text.slice(i).match(/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?/);
      if(!match)throw displayError(displayMessage('modules.annotation.text.34e125849185'));
      if(i===0||text[i-1]!=='[')tokens.push(Number(match[0]));i+=match[0].length;
    }else if(/[a-zA-Z_]/.test(c)){
      const match=text.slice(i).match(/^[a-zA-Z_][a-zA-Z0-9_?]*/)!;
      if(/^(nan|inf|infinity)$/i.test(match[0]))throw displayError(displayMessage('modules.annotation.text.47fd9eeb5f0a'));i+=match[0].length;
    }else i++;
    if(tokens.length>501_000)throw displayError(displayMessage('modules.annotation.text.0d36769542d5'));
  }
  let cursor=0;
  const str=()=>{const value=tokens[cursor++];if(typeof value!=='string')throw displayError(displayMessage('modules.annotation.text.9a5bc9089b98'));return value;};
  const num=()=>{const value=tokens[cursor++];if(typeof value!=='number'||!Number.isFinite(value))throw displayError(displayMessage('modules.annotation.text.0aa281c79d13'));return value;};
  const count=(limit:number)=>{const value=num();if(!Number.isInteger(value)||value<0||value>limit)throw displayError(displayMessage('modules.annotation.text.06c43437ecf0'));return value;};
  if(!['ooTextFile','ooTextFile short'].includes(str())||str()!=='TextGrid')throw displayError(displayMessage('modules.annotation.text.c2b5425045c2'));
  const grid:Grid={xmin:num(),xmax:num(),tiers:[]};
  if(grid.xmin<0||grid.xmax<=grid.xmin||str()!=='<exists>')throw displayError(displayMessage('modules.annotation.text.ae12146ab2d3'));
  const n=count(64);let total=0;const names=new Set<string>();
  for(let t=0;t<n;t++){
    const kind=str(),name=str(),xmin=num(),xmax=num(),size=count(100_000-total);total+=size;
    if(names.has(name))throw displayError(displayMessage('modules.annotation.text.f6d086e105d4'));names.add(name);
    if(xmin<grid.xmin||xmax>grid.xmax||xmax<xmin)throw displayError(displayMessage('modules.annotation.text.8a621824c6ae'));
    const tier:AnnotationTier={name,xmin,xmax};let last=xmin;
    if(kind==='IntervalTier'){
      tier.intervals=[];
      for(let k=0;k<size;k++){const a=num(),b=num(),label=str();if(a<last||b<=a||b>xmax)throw displayError(displayMessage('modules.annotation.text.ffd5b211cd42'));tier.intervals.push({xmin:a,xmax:b,text:label});last=b;}
    }else if(kind==='TextTier'){
      tier.points=[];
      for(let k=0;k<size;k++){const number=num(),mark=str();if(number<last||number>xmax)throw displayError(displayMessage('modules.annotation.text.10a72138c850'));tier.points.push({number,mark});last=number;}
    }else throw displayError(displayMessage('modules.annotation.text.8025da45c88d'));
    grid.tiers.push(tier);
  }
  if(cursor!==tokens.length||(!grid.tiers.length&&!allowNoTiers))throw displayError(displayMessage('modules.annotation.text.de46b7cda9fe'));
  return grid;
}
export function serializeGrid(grid:Grid):string {
  const quote=(s:string)=>'"'+s.replaceAll('"','""')+'"';
  const num=(n:number)=>{if(!Number.isFinite(n))throw displayError(displayMessage('modules.annotation.text.36d8bd805795'));return n.toFixed(6).replace(/0+$/,'').replace(/\.$/,'.0');};
  const lines=['File type = "ooTextFile"','Object class = "TextGrid"','',`xmin = ${num(grid.xmin)}`,`xmax = ${num(grid.xmax)}`,'tiers? <exists>',`size = ${grid.tiers.length}`,'item []:'];
  grid.tiers.forEach((tier,t)=>{
    lines.push(`    item [${t+1}]:`,`        class = "${tier.intervals?'IntervalTier':'TextTier'}"`,`        name = ${quote(tier.name)}`,`        xmin = ${num(tier.xmin??grid.xmin)}`,`        xmax = ${num(tier.xmax??grid.xmax)}`);
    if(tier.intervals){lines.push(`        intervals: size = ${tier.intervals.length}`);tier.intervals.forEach((item,i)=>lines.push(`        intervals [${i+1}]:`,`            xmin = ${num(item.xmin)}`,`            xmax = ${num(item.xmax)}`,`            text = ${quote(item.text)}`));}
    else{lines.push(`        points: size = ${tier.points!.length}`);tier.points!.forEach((item,i)=>lines.push(`        points [${i+1}]:`,`            number = ${num(item.number)}`,`            mark = ${quote(item.mark)}`));}
  });
  const text=lines.join('\n')+'\n';parseGrid(text,true);return text;
}
export function decodeText(buffer:ArrayBuffer):string {
  const bytes=new Uint8Array(buffer);
  if(bytes[0]===0xff&&bytes[1]===0xfe)return new TextDecoder('utf-16le',{fatal:true}).decode(bytes);
  if(bytes[0]===0xfe&&bytes[1]===0xff)return new TextDecoder('utf-16be',{fatal:true}).decode(bytes);
  try{return new TextDecoder('utf-8',{fatal:true}).decode(bytes);}catch{return new TextDecoder('gb18030',{fatal:true}).decode(bytes);}
}
export function validateEditingGrid(grid:Grid,duration:number){
  if(grid.xmin!==0||Math.abs(grid.xmax-duration)>1e-6)throw displayError(displayMessage('modules.annotation.text.00058b9ac10c'));
  // Preserve the original domain and all unrelated tiers. Do not stretch labels.
  return grid;
}
export function loadEditingGrid(text:string|undefined,duration:number):{grid:Grid;created:boolean}{
  if(!Number.isFinite(duration)||duration<=0)throw displayError(displayMessage('modules.annotation.text.d0eefe11ae2b'));
  const parsed=text?.trim()?validateEditingGrid(parseGrid(text,true),duration):undefined;
  if(parsed?.tiers.length)return {grid:parsed,created:false};
  return {grid:parsed??{xmin:0,xmax:duration,tiers:[]},created:true};
}
export function validateEditingAudio(asset:AudioAsset){
  if(asset.sampleRate<8000||asset.sampleRate>96000||asset.frames*asset.channels.length>32_000_000||asset.channels.length>8)throw displayError(displayMessage('modules.annotation.text.cbc5e7031667'));
  return asset;
}
export function preferredGrid<T extends {name:string}>(audio:T,files:T[]):T|undefined {
  const stem=audio.name.replace(/\.wav$/i,'');
  for(const suffix of ['','_自动保存','_webedit','_post','_auto']){const matches=files.filter(f=>f.name.toLowerCase()===(stem+suffix+'.textgrid').toLowerCase());if(matches.length>1)throw displayError(displayMessage('modules.annotation.text.ddfa99de6015'));if(matches[0])return matches[0];}
  const candidates=files.filter(f=>f.name.toLowerCase().startsWith(stem.toLowerCase()+'_')&&f.name.toLowerCase().endsWith('.textgrid'));
  if(candidates.length===1)return candidates[0];
  if(candidates.length>1)throw displayError(displayMessage('modules.annotation.text.ce3003aed705'));
}
export function selectedInterval(grid:Grid|null,selection:{tier:string;index:number}|null):Interval|undefined{return selection?grid?.tiers.find(t=>t.name===selection.tier)?.intervals?.[selection.index]:undefined;}
