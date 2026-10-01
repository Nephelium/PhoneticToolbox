import type {EggTaskConfig} from '../../platform/research.ts';
import {defaults,taskConfig,validate} from './state.ts';
// Display coordinates only. The normalized analysis samples stay unchanged.
export function visibleAmplitude(times:readonly number[],values:readonly (number|null)[],start:number,end:number):[number,number] {
  let peak=0;
  for(let i=0;i<Math.min(times.length,values.length);i++) {
    const value=values[i];
    if(times[i]>=start&&times[i]<=end&&value!==null&&Number.isFinite(value))peak=Math.max(peak,Math.abs(value));
  }
  // A zero signal needs a finite axis; nonzero quiet signals keep their scale.
  const limit=peak>0?peak*1.08:1;
  return [-limit,limit];
}

// A draft can be temporarily empty or invalid while typing. Keep the last
// valid scientific axes until the edited request can actually be computed.
export function displayConfig(draft:EggTaskConfig,snapshot:EggTaskConfig|undefined,duration:number,rate:number):EggTaskConfig {
  try {validate(taskConfig(draft,'preview'),duration,rate);return {...snapshot,...draft};}
  catch {return snapshot??defaults();}
}
