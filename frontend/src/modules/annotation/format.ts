import type {Grid,AnnotationTier,Interval} from './editor.mjs';
import type {AudioAsset} from '../../platform/types.ts';

// SRC-PRAAT: labelled and short text formats, doubled quotes and multiline labels.
// Point tiers remain intact even though the two active editing rows are intervals.
export function parseGrid(text:string):Grid {
  if(text.length>2_000_000)throw Error('TextGrid 超过 2 MB。');
  const tokens:(string|number)[]=[];let i=0;
  while(i<text.length){
    const c=text[i];
    if(c==='"'){
      i++;let value='',closed=false;
      while(i<text.length){if(text[i]==='"'){if(text[i+1]==='"'){value+='"';i+=2;continue;}i++;closed=true;break;}value+=text[i++];}
      if(!closed)throw Error('TextGrid 文本引号未闭合。');tokens.push(value);
    }else if(text.startsWith('<exists>',i)){tokens.push('<exists>');i+=8;}
    else if(/[+\-.0-9]/.test(c)){
      const match=text.slice(i).match(/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?/);
      if(!match)throw Error('TextGrid 数字格式错误。');
      if(i===0||text[i-1]!=='[')tokens.push(Number(match[0]));i+=match[0].length;
    }else if(/[a-zA-Z_]/.test(c)){
      const match=text.slice(i).match(/^[a-zA-Z_][a-zA-Z0-9_?]*/)!;
      if(/^(nan|inf|infinity)$/i.test(match[0]))throw Error('TextGrid 含非有限时间。');i+=match[0].length;
    }else i++;
    if(tokens.length>501_000)throw Error('TextGrid 层级或区间数量过多。');
  }
  let cursor=0;
  const str=()=>{const value=tokens[cursor++];if(typeof value!=='string')throw Error('TextGrid 文本结构错误。');return value;};
  const num=()=>{const value=tokens[cursor++];if(typeof value!=='number'||!Number.isFinite(value))throw Error('TextGrid 时间结构错误。');return value;};
  const count=(limit:number)=>{const value=num();if(!Number.isInteger(value)||value<0||value>limit)throw Error('TextGrid 数量越界。');return value;};
  if(!['ooTextFile','ooTextFile short'].includes(str())||str()!=='TextGrid')throw Error('请选择 Praat 文本 TextGrid 文件。');
  const grid:Grid={xmin:num(),xmax:num(),tiers:[]};
  if(grid.xmin<0||grid.xmax<=grid.xmin||str()!=='<exists>')throw Error('TextGrid 总时间范围无效。');
  const n=count(64);let total=0;const names=new Set<string>();
  for(let t=0;t<n;t++){
    const kind=str(),name=str(),xmin=num(),xmax=num(),size=count(100_000-total);total+=size;
    if(names.has(name))throw Error('TextGrid 层名重复，无法安全编辑。');names.add(name);
    if(xmin<grid.xmin||xmax>grid.xmax||xmax<xmin)throw Error('TextGrid 层时间超出文件。');
    const tier:AnnotationTier={name,xmin,xmax};let last=xmin;
    if(kind==='IntervalTier'){
      tier.intervals=[];
      for(let k=0;k<size;k++){const a=num(),b=num(),label=str();if(a<last||b<=a||b>xmax)throw Error('TextGrid 区间重叠或时间无效。');tier.intervals.push({xmin:a,xmax:b,text:label});last=b;}
    }else if(kind==='TextTier'){
      tier.points=[];
      for(let k=0;k<size;k++){const number=num(),mark=str();if(number<last||number>xmax)throw Error('TextGrid 点时间无效。');tier.points.push({number,mark});last=number;}
    }else throw Error('不支持的 TextGrid 层类型。');
    grid.tiers.push(tier);
  }
  if(cursor!==tokens.length||!grid.tiers.length)throw Error('TextGrid 不完整或含多余内容。');
  return grid;
}
export function serializeGrid(grid:Grid):string {
  const quote=(s:string)=>'"'+s.replaceAll('"','""')+'"';
  const num=(n:number)=>{if(!Number.isFinite(n))throw Error('标注时间无效。');return n.toFixed(6).replace(/0+$/,'').replace(/\.$/,'.0');};
  const lines=['File type = "ooTextFile"','Object class = "TextGrid"','',`xmin = ${num(grid.xmin)}`,`xmax = ${num(grid.xmax)}`,'tiers? <exists>',`size = ${grid.tiers.length}`,'item []:'];
  grid.tiers.forEach((tier,t)=>{
    lines.push(`    item [${t+1}]:`,`        class = "${tier.intervals?'IntervalTier':'TextTier'}"`,`        name = ${quote(tier.name)}`,`        xmin = ${num(tier.xmin??grid.xmin)}`,`        xmax = ${num(tier.xmax??grid.xmax)}`);
    if(tier.intervals){lines.push(`        intervals: size = ${tier.intervals.length}`);tier.intervals.forEach((item,i)=>lines.push(`        intervals [${i+1}]:`,`            xmin = ${num(item.xmin)}`,`            xmax = ${num(item.xmax)}`,`            text = ${quote(item.text)}`));}
    else{lines.push(`        points: size = ${tier.points!.length}`);tier.points!.forEach((item,i)=>lines.push(`        points [${i+1}]:`,`            number = ${num(item.number)}`,`            mark = ${quote(item.mark)}`));}
  });
  const text=lines.join('\n')+'\n';parseGrid(text);return text;
}
export function decodeText(buffer:ArrayBuffer):string {
  const bytes=new Uint8Array(buffer);
  if(bytes[0]===0xff&&bytes[1]===0xfe)return new TextDecoder('utf-16le',{fatal:true}).decode(bytes);
  if(bytes[0]===0xfe&&bytes[1]===0xff)return new TextDecoder('utf-16be',{fatal:true}).decode(bytes);
  try{return new TextDecoder('utf-8',{fatal:true}).decode(bytes);}catch{return new TextDecoder('gb18030',{fatal:true}).decode(bytes);}
}
export function validateEditingGrid(grid:Grid,duration:number){
  if(grid.xmin!==0||Math.abs(grid.xmax-duration)>1e-6)throw Error('TextGrid 与音频时间范围不一致，请先在 Praat 核对。');
  // Preserve the original domain and all unrelated tiers. Do not stretch labels.
  return grid;
}
export function validateEditingAudio(asset:AudioAsset){
  if(asset.sampleRate<8000||asset.sampleRate>96000||asset.frames>8_000_000||asset.channels.length>8)throw Error('标注工作台支持 8–96 kHz、最多 8 声道及 800 万帧，请先转换或切分录音。');
  return asset;
}
export function preferredGrid<T extends {name:string}>(audio:T,files:T[]):T|undefined {
  const stem=audio.name.replace(/\.wav$/i,'');
  for(const suffix of ['_webedit','_post','_auto','']){const matches=files.filter(f=>f.name.toLowerCase()===(stem+suffix+'.textgrid').toLowerCase());if(matches.length>1)throw Error('存在同名标注资源，请明确选择。');if(matches[0])return matches[0];}
}
export function selectedInterval(grid:Grid|null,selection:{tier:string;index:number}|null):Interval|undefined{return selection?grid?.tiers.find(t=>t.name===selection.tier)?.intervals?.[selection.index]:undefined;}
