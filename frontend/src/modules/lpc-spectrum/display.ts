import {uiError,uiErrorMessage,uiMessage,joinMessages} from '../module-localization.ts';
import {tr} from '../../i18n/index.ts';
import type {LpcResult} from './state.ts';

/** Explicit relative display only; neither saved FFT nor LPC values are edited. */
export function referenceTrace(result:LpcResult){
  const fft=result.reference_spectrum;if(!fft)return;
  const peak=(frequencies:number[],values:number[])=>Math.max(...values.filter((_,i)=>frequencies[i]<=result.config.freq_max_hz));
  const offset=peak(result.spectrum.frequencies_hz,result.spectrum.magnitude_db)-peak(fft.frequencies_hz,fft.magnitude_db);
  return {frequencies_hz:fft.frequencies_hz,magnitude_db:fft.magnitude_db.map(v=>v+offset),offset_db:offset};
}

/** Display bounds use the existing spectrum. The LPC samples never change. */
export function axisLimits(result:LpcResult,dynamic:boolean,minimum:unknown,maximum:unknown):[number,number]{
  if(dynamic){
    const values=result.spectrum.magnitude_db.filter((_,i)=>result.spectrum.frequencies_hz[i]<=result.config.freq_max_hz);
    if(!values.length)throw uiError(uiMessage("modules.lpc-spectrum.no_spectral_values_to_display_in_this_frequency_range"));
    return [Math.min(...values)-5,Math.max(...values)+5];
  }
  const a=Number(minimum),b=Number(maximum);
  if(String(minimum).trim()===''||String(maximum).trim()===''||!Number.isFinite(a)||!Number.isFinite(b)||a<-200||b>100||b<=a)
    throw uiError(uiMessage("modules.lpc-spectrum.fixed_vertical_axis_requires_200_minimum_db_maximum_db"));
  return [a,b];
}
