import {localized} from '../module-localization.ts';
import {tr} from '../../i18n/index.ts';
import type {components} from '../../../../contracts/generated/api';

export type InverseLayout='separate'|'audio-if'|'egg-if'|'combined';
export type SignalKey='audio'|'egg'|'inverse';
const layoutGroups:Record<InverseLayout,SignalKey[][]>={separate:[['audio'],['egg'],['inverse']],'audio-if':[['audio','inverse'],['egg']],'egg-if':[['egg','inverse'],['audio']],combined:[['audio','egg','inverse']]};
export const inverseLayouts:{id:InverseLayout;label:string;groups:SignalKey[][]}[]=localized(()=>([
  {id:'separate',label:tr("modules.egg-analysis.all_separate_6_plots"),groups:layoutGroups.separate},
  {id:'audio-if',label:tr("modules.egg-analysis.audio_if_4_plots"),groups:layoutGroups['audio-if']},
  {id:'egg-if',label:tr("modules.egg-analysis.egg_if_4_plots"),groups:layoutGroups['egg-if']},
  {id:'combined',label:tr("modules.egg-analysis.audio_egg_if_2_plots"),groups:layoutGroups.combined},
]));

const styles={
 audio:{color:'var(--wave)',exportColor:'#174b82',lineWidth:1.8,dash:undefined},
 inverse:{color:'var(--warning)',exportColor:'#a63f10',lineWidth:2.2,dash:'8 4'},
 egg:{color:'var(--violet)',exportColor:'#633d91',lineWidth:2.2,dash:'1 5'},
};
const signalLabel=(key:SignalKey)=>key==='audio'?tr('modules.egg-analysis.audio'):key==='egg'?'EGG':'IF';
const signalShape=(key:SignalKey)=>tr('modules.egg-analysis.'+(key==='audio'?'solid':key==='egg'?'dotted':'dashed'));
const styledTrace=<T extends object>(key:SignalKey,values:T)=>({...styles[key],...values,get label(){return signalLabel(key);},get shape(){return signalShape(key);}});
export function extent(values:number[]):[number,number]{
  let lo=Infinity,hi=-Infinity;
  for(const value of values)if(Number.isFinite(value)){lo=Math.min(lo,value);hi=Math.max(hi,value);}
  if(!Number.isFinite(lo))return [-1,1];
  const pad=Math.max(.001,(hi-lo)*.05);return [lo-pad,hi+pad];
}
export function peakNormalized(values:number[]){
  const peak=values.reduce((m,v)=>Math.max(m,Math.abs(v)),0);
  return values.map(v=>peak>0?v/peak:0);
}
export function inversePanels(data:components['schemas']['EggInverseData'],layout:InverseLayout){
  const groups=layoutGroups[layout];
  const milliseconds=data.relative_times_s.map(t=>t*1000);
  return groups.flatMap(keys=>{
    const title=()=>keys.map(key=>keys.length>1?tr('modules.egg-analysis.signalStyle',{signal:signalLabel(key),style:signalShape(key)}):signalLabel(key)).join(' / ');
    const spectra=keys.map(key=>styledTrace(key,{times:data.frequencies_hz,values:data[`${key}_db`]}));
    const normalized=keys.length===3;
    const waves=keys.map((key,index)=>styledTrace(key,{...(key==='audio'?{color:'var(--waveform-color,var(--wave))',exportColor:'var(--waveform-color)'}:{}),times:milliseconds,
      values:normalized?peakNormalized(data[`${key}_values`]):data[`${key}_values`],right:keys.length===2&&index===1}));
    return [
      {id:keys.join('-')+'-spectrum',get title(){return tr('modules.egg-analysis.spectrumTitle',{signals:title()});},x:[0,Math.min(5000,data.frequencies_hz.at(-1)??5000)] as [number,number],
        y:extent(spectra.flatMap(trace=>trace.values)),unit:'dB',xUnit:'Hz',traces:spectra,right:undefined,rightUnit:undefined},
      {id:keys.join('-')+'-wave',get title(){return tr(normalized?'modules.egg-analysis.normalizedWaveTitle':'modules.egg-analysis.waveTitle',{signals:title()});},
        x:[milliseconds[0]??-25,milliseconds.at(-1)??25] as [number,number],y:normalized?[-1.05,1.05] as [number,number]:extent(waves[0].values),
        right:keys.length===2?extent(waves[1].values):undefined,get rightUnit(){return keys.length===2?signalLabel(keys[1]):undefined;},
        get unit(){return normalized?tr('modules.egg-analysis.normalized_amplitude'):keys.length===2?signalLabel(keys[0]):tr('modules.egg-analysis.amplitude');},xUnit:'ms',traces:waves},
    ];
  });
}
