import {uiError,uiErrorMessage,uiMessage,joinMessages} from '../module-localization.ts';
import {tr} from '../../i18n/index.ts';
import type {LipRow} from './port.ts';

export const parameterKeys=['area','height','outer_width','inner_width','total_width','open','circularity','face_width','face_height'];
export function checkedOffsetMs(value:number){
 if(!Number.isFinite(value)||Math.abs(value)>2000)throw uiError(uiMessage("modules.lip-extraction.offset_must_be_within_2000_to_2000_ms"));
 return Math.round(value)/1000;
}
/** Shared time axis; normalization affects only this comparison display. */
export function normalizedParameter(rows:LipRow[],key:string,offset:number){
 let low=Infinity,high=-Infinity;
 for(const row of rows){const v=row.metrics?.[key];if(typeof v==='number'&&Number.isFinite(v)){low=Math.min(low,v);high=Math.max(high,v);}}
 return {times:rows.map(r=>r.time_s+offset),values:rows.map(r=>{const v=r.metrics?.[key];return typeof v==='number'&&Number.isFinite(v)?(high>low?(v-low)/(high-low):.5):null;})};
}
export function waveformEnvelope(buffer:AudioBuffer,limit=12000){
 const step=Math.max(1,Math.ceil(buffer.length/limit)),times:number[]=[],values:number[]=[];
 // Channel 1 is explicit: phase-opposed stereo must not cancel the waveform.
 const data=buffer.getChannelData(0);
 for(let start=0;start<data.length;start+=step){let low=Infinity,high=-Infinity,li=start,hi=start;
  for(let i=start;i<Math.min(start+step,data.length);i++){if(data[i]<low){low=data[i];li=i;}if(data[i]>high){high=data[i];hi=i;}}
  for(const i of li<hi?[li,hi]:[hi,li]){times.push(i/buffer.sampleRate);values.push(data[i]);}
 }
 return {times,values,duration:buffer.duration,sampleRate:buffer.sampleRate,channels:buffer.numberOfChannels};
}
export type AlignmentWaveform=ReturnType<typeof waveformEnvelope>;
