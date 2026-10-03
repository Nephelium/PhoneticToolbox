import type {components} from '../../../../contracts/generated/api';

export type InverseLayout='separate'|'audio-if'|'egg-if'|'combined';
export type SignalKey='audio'|'egg'|'inverse';
export const inverseLayouts:{id:InverseLayout;label:string;groups:SignalKey[][]}[]=[
  {id:'separate',label:'全部分开 · 6 图',groups:[['audio'],['egg'],['inverse']]},
  {id:'audio-if',label:'音频 + IF · 4 图',groups:[['audio','inverse'],['egg']]},
  {id:'egg-if',label:'EGG + IF · 4 图',groups:[['egg','inverse'],['audio']]},
  {id:'combined',label:'音频 + EGG + IF · 2 图',groups:[['audio','egg','inverse']]},
];
const styles={
  audio:{label:'音频',color:'var(--wave)',exportColor:'#174b82',lineWidth:1.8,dash:undefined,shape:'实线'},
  inverse:{label:'IF',color:'var(--warning)',exportColor:'#a63f10',lineWidth:2.2,dash:'8 4',shape:'虚线'},
  egg:{label:'EGG',color:'var(--violet)',exportColor:'#633d91',lineWidth:2.2,dash:'1 5',shape:'点线'},
};
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
  const groups=inverseLayouts.find(item=>item.id===layout)!.groups;
  const milliseconds=data.relative_times_s.map(t=>t*1000);
  return groups.flatMap(keys=>{
    const title=keys.map(key=>styles[key].label+(keys.length>1?`（${styles[key].shape}）`:'')).join(' / ');
    const spectra=keys.map(key=>({...styles[key],times:data.frequencies_hz,values:data[`${key}_db`]}));
    const normalized=keys.length===3;
    const waves=keys.map((key,index)=>({...styles[key],...(key==='audio'?{color:'var(--waveform-color,var(--wave))',exportColor:'var(--waveform-color)'}:{}),times:milliseconds,
      values:normalized?peakNormalized(data[`${key}_values`]):data[`${key}_values`],right:keys.length===2&&index===1}));
    return [
      {id:keys.join('-')+'-spectrum',title:title+'频谱',x:[0,Math.min(5000,data.frequencies_hz.at(-1)??5000)] as [number,number],
        y:extent(spectra.flatMap(trace=>trace.values)),unit:'dB',xUnit:'Hz',traces:spectra,right:undefined,rightUnit:undefined},
      {id:keys.join('-')+'-wave',title:title+(normalized?'波形 · 各自峰值归一化':'波形 · 中心 50 ms'),
        x:[milliseconds[0]??-25,milliseconds.at(-1)??25] as [number,number],y:normalized?[-1.05,1.05] as [number,number]:extent(waves[0].values),
        right:keys.length===2?extent(waves[1].values):undefined,rightUnit:keys.length===2?styles[keys[1]].label:undefined,
        unit:normalized?'归一化振幅':keys.length===2?styles[keys[0]].label:'振幅',xUnit:'ms',traces:waves},
    ];
  });
}
