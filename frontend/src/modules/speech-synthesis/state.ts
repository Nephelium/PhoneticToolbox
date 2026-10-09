import catalog from './catalog.json' with {type:'json'};
export interface Curve {points:[number,number][];override:number|null}
export const f0Methods={praat_cc:'Praat 互相关（CC）',praat_ac:'Praat 自相关（AC）',reaper:'REAPER'} as const;
export interface Config {schema_version:'m06/2';duration:number;sample_rate:number;sequence:string;fade_in:number;fade_out:number;smooth:number;f0_range:[number,number];f0_method:keyof typeof f0Methods;curves:Record<string,Curve>;silence:[number,number][];boundaries:number[];f0_transform:{preset:'假声'|'嘎裂'|null;offset_hz:number}}
export const parameters:Record<string,[number,number,number,string]>=Object.fromEntries(Object.entries(catalog.parameters).map(([n,v])=>[n,[Number(v[0]),Number(v[1]),Number(v[2]),String(v[3])]]));
export const vowels=catalog.vowels;export const presets=catalog.presets as Record<string,Record<string,number>>;
export const defaults=():Config=>JSON.parse(JSON.stringify(catalog.defaults));
export const clone=(c:Config)=>JSON.parse(JSON.stringify(c)) as Config;
export function number(value:unknown,label:string){if(value===''||value===null||!Number.isFinite(Number(value)))throw Error(label+'需要有限数值');return Number(value);}
export function valid(c:Config):Config {
 if((c.schema_version as string)==='m06/1')throw Error('此文件使用旧版 AV 标尺，无法直接换算。原文件保留，请使用新版参数。');
 if(c.schema_version!=='m06/2'||Object.keys(c.curves).sort().join()!=Object.keys(parameters).sort().join())throw Error('参数字段不完整');
 if(!Object.hasOwn(c,'f0_method'))c.f0_method='praat_cc';
 const legacy=c as Config & {render?:{method?:string}};
 if(Object.hasOwn(c,'render')){if(legacy.render?.method!=='klatt')throw Error('此文件使用已移除的重合成方法。原文件保留，请使用 Klatt 参数文件。');delete legacy.render;}
 if((c.f0_method as string)==='harvest')throw Error('Harvest 已移除，请使用 Praat 或 REAPER 参数文件。');
 if(!Object.hasOwn(f0Methods,c.f0_method))throw Error('F0 提取算法无效');
 if(!Number.isFinite(c.duration)||c.duration<.1||c.duration>100)throw Error('总时长应为 0.1–100 秒');
 if(!Number.isInteger(c.sample_rate)||c.sample_rate<8000||c.sample_rate>192000)throw Error('采样率无效');
 if(!Array.isArray(c.f0_range)||c.f0_range.length!==2||!c.f0_range.every(Number.isFinite)||c.f0_range[0]<1||c.f0_range[1]>3000||c.f0_range[0]>=c.f0_range[1])throw Error('F0 范围无效');
 const transform=c.f0_transform;
 if(!transform||![null,'假声','嘎裂'].includes(transform.preset)||!Number.isFinite(transform.offset_hz)||Math.abs(transform.offset_hz)>2999||transform.preset===null&&transform.offset_hz!==0||transform.preset==='假声'&&transform.offset_hz<0||transform.preset==='嘎裂'&&transform.offset_hz>0)throw Error('F0 平移状态无效');
 for(const [k,lo,hi] of [['fade_in',0,1000],['fade_out',0,1000],['smooth',1,50]] as const)if(!Number.isInteger(c[k])||c[k]<lo||c[k]>hi)throw Error(k+' 超出范围');
 for(const [n,curve] of Object.entries(c.curves)){
  if(!Array.isArray(curve.points)||!curve.points.length||curve.points.length>10001||curve.points.some((p,i)=>p.length!==2||!p.every(Number.isFinite)||p[0]<0||p[0]>c.duration||i>0&&p[0]<curve.points[i-1][0]))throw Error(n+' 曲线无效');
  const [lo,hi]=n==='F0'?c.f0_range:parameters[n].slice(1,3) as number[];
  if(curve.points.some(([,v])=>v<lo||v>hi))throw Error(n+' 曲线数值超出范围');
  if(n==='F0'&&curve.points.some(([,v])=>v-transform.offset_hz<1||v-transform.offset_hz>3000))throw Error('F0 基础曲线超出范围');
  if(curve.override!==null&&(!Number.isFinite(curve.override)||curve.override<lo||curve.override>hi))throw Error(n+' 覆盖值超出范围');
  if(n==='F0'&&curve.override!==null&&(curve.override-transform.offset_hz<1||curve.override-transform.offset_hz>3000))throw Error('F0 基础曲线超出范围');
 }
 if(typeof c.sequence!=='string'||c.sequence.length>2048)throw Error('IPA 序列过长');
 if(!Array.isArray(c.silence)||c.silence.some(p=>p.length!==2||!p.every(Number.isFinite)||p[0]<0||p[1]>c.duration||p[0]>=p[1]))throw Error('静音区间无效');
 if(!Array.isArray(c.boundaries)||c.boundaries.some(t=>!Number.isFinite(t)||t<0||t>c.duration))throw Error('元音边界无效');
 return c;
}
export function ipa(text:string){if(!text.trim())throw Error('请输入元音序列');let prev=false;for(const [i,ch] of [...text].entries()){if(ch.toLowerCase() in vowels||ch===' ')prev=true;else if('+-*/'.includes(ch)&&prev)continue;else throw Error(`IPA 第 ${i+1} 个字符无效：${ch}`);}}
export function override(c:Config,name:string,text:string){
 const curve=c.curves[name];if(!text.trim()){curve.override=null;return;}
 const factor=name==='Shimmer'?100:1,scalar=Number(text);
 if(Number.isFinite(scalar)){const next=clone(c);next.curves[name].override=scalar/factor;valid(next);curve.override=scalar/factor;return;}
 const parts=text.replaceAll('，',',').replaceAll('；',';').split(';'),points:[number,number][]=[];
 for(const [i,part] of parts.entries()){
  const values=part.trim().split(/[,\s]+/).map(x=>number(x,name)/factor),a=i*c.duration/parts.length,b=(i+1)*c.duration/parts.length;
  if(values.length===1)points.push([a,values[0]],[b,values[0]]);
  else values.forEach((v,j)=>points.push([a+j/(values.length-1)*(b-a),v]));
 }
 const next=clone(c);next.curves[name]={points,override:null};valid(next);curve.points=points;curve.override=null;
}
export function resize(c:Config,duration:number){number(duration,'总时长');if(duration<.1||duration>100)throw Error('总时长应为 0.1–100 秒');const ratio=duration/c.duration;for(const curve of Object.values(c.curves))curve.points=curve.points.map(([t,v])=>[t*ratio,v]);c.silence=c.silence.map(([a,b])=>[a*ratio,b*ratio]);c.boundaries=c.boundaries.map(t=>t*ratio);c.duration=duration;}
export function f0Range(c:Config,low:number,high:number){const next=clone(c);next.f0_range=[low,high];for(const curve of [next.curves.F0]){curve.points=curve.points.map(([t,v])=>[t,Math.min(high,Math.max(low,v))]);if(curve.override!==null)curve.override=Math.min(high,Math.max(low,curve.override));}valid(next);Object.assign(c,next);}
export function preset(c:Config,name:string){
 if(!presets[name])throw Error('未知预设');
 const next=clone(c),previous=next.f0_transform,curve=next.curves.F0;
 const target=name==='假声'||name==='嘎裂'?name:null;
 if(target!==previous.preset){
  const effective: [number,number][]=curve.override===null?curve.points:[[0,curve.override],[c.duration,curve.override]];
  const base=effective.map(([t,v])=>[t,v-previous.offset_hz] as [number,number]);
  const values=base.map(([,v])=>v),minimum=Math.min(...values),maximum=Math.max(...values);
  // Time-weighted mean, including endpoint extension and zero-length steps.
  let integral=base[0][0]*base[0][1]+(c.duration-base.at(-1)![0])*base.at(-1)![1];
  for(let i=1;i<base.length;i++)integral+=(base[i][0]-base[i-1][0])*(base[i][1]+base[i-1][1])/2;
  const mean=integral/c.duration;
  let offset=target==='假声'?Math.max(0,300-mean):target==='嘎裂'?Math.min(0,70-mean):0;
  offset=Math.max(Math.min(0,20-minimum),Math.min(3000-maximum,offset));
  curve.points=base.map(([t,v])=>[t,v+offset]);curve.override=null;
  // Keep the existing axis whenever possible so the shift is visible on screen.
  // Expand only for points that would otherwise fall outside the plot.
  next.f0_range=[Math.max(1,Math.min(minimum+offset,next.f0_range[0])),Math.min(3000,Math.max(maximum+offset,next.f0_range[1]))];
  next.f0_transform={preset:target,offset_hz:offset};
 }
 for(const [key,value] of Object.entries(presets[name]))next.curves[key].override=value;
 valid(next);Object.assign(c,next);
}
export function draw(c:Config,name:string,a:number,b:number,value:number,reset=false){const curve=c.curves[name];if(curve.override!==null)throw Error('请先清除覆盖，再编辑曲线');const lo=Math.max(0,Math.min(c.duration,a,b)),hi=Math.max(0,Math.min(c.duration,Math.max(a,b))),range=name==='F0'?c.f0_range:parameters[name].slice(1,3) as number[];const lower=name==='F0'?Math.max(range[0],1+c.f0_transform.offset_hz):range[0],upper=name==='F0'?Math.min(range[1],3000+c.f0_transform.offset_hz):range[1];const v=Math.min(upper,Math.max(lower,reset?parameters[name][0]+(name==='F0'?c.f0_transform.offset_hz:0):value));curve.points=[...curve.points.filter(([t])=>t<lo||t>hi),[lo,v] as [number,number],[hi,v] as [number,number]].sort((a,b)=>a[0]-b[0]);}
function csvCell(v:unknown){return '"'+String(v).replaceAll('"','""')+'"';}
export function exportParams(c:Config){valid(c);const rows:unknown[][]=[['Parameter','Time','Value','Global'],['__DURATION__',c.duration,0,false],['__VOWEL_INPUT__',0,c.sequence,false],['__PTB_CONFIG__',0,JSON.stringify(c),false]];for(const [n,curve] of Object.entries(c.curves)){if(curve.override!==null)rows.push([n,0,curve.override,true]);else for(const [t,v] of curve.points)rows.push([n,t,v,false]);}return rows.map(r=>r.map(csvCell).join(',')).join('\r\n');}
function csv(text:string){const rows:string[][]=[];let row:string[]=[],cell='',quoted=false;for(let i=0;i<text.length;i++){const ch=text[i];if(ch==='"'){if(quoted&&text[i+1]==='"'){cell+='"';i++;}else quoted=!quoted;}else if(!quoted&&ch===','){row.push(cell);cell='';}else if(!quoted&&(ch==='\n'||ch==='\r')){if(ch==='\r'&&text[i+1]==='\n')i++;row.push(cell);rows.push(row);row=[];cell='';}else cell+=ch;}if(quoted)throw Error('CSV 引号未闭合');if(cell||row.length){row.push(cell);rows.push(row);}return rows;}
export function importParams(text:string){if(new TextEncoder().encode(text).length>8_000_000)throw Error('参数文件超过 8 MB');text=text.replace(/^\ufeff/,'');if(text.trim().startsWith('{')){const value=JSON.parse(text);return valid(['m06/1','m06/2'].includes(value?.schema_version)&&value.config?value.config:value);}const rows=csv(text);if(rows.shift()?.join(',')!=='Parameter,Time,Value,Global')throw Error('CSV 表头无效');const full=rows.filter(r=>r[0]==='__PTB_CONFIG__');if(full.length>1)throw Error('重复完整快照');if(full.length)return valid(JSON.parse(full[0][2]));throw Error('旧版 CSV 缺少新版标尺信息，原文件保留，请使用新版参数。');
}

export const isCurrent=(submittedRevision:number,currentRevision:number,ticket:number,currentTicket:number)=>submittedRevision===currentRevision&&ticket===currentTicket;
