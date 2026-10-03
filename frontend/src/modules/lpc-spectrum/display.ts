import type {LpcResult} from './state.ts';

/** Display bounds use the existing spectrum. The LPC samples never change. */
export function axisLimits(result:LpcResult,dynamic:boolean,minimum:unknown,maximum:unknown):[number,number]{
  if(dynamic){
    const values=result.spectrum.magnitude_db.filter((_,i)=>result.spectrum.frequencies_hz[i]<=result.config.freq_max_hz);
    if(!values.length)throw Error('当前频率范围没有可显示的谱值。');
    return [Math.min(...values)-5,Math.max(...values)+5];
  }
  const a=Number(minimum),b=Number(maximum);
  if(String(minimum).trim()===''||String(maximum).trim()===''||!Number.isFinite(a)||!Number.isFinite(b)||a<-200||b>100||b<=a)
    throw Error('固定纵轴须满足 −200 ≤ dB 下限 < 上限 ≤ 100。');
  return [a,b];
}
