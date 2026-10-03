import type {LpcResult} from './state.ts';
import {quoteFamily} from '../../design/fonts.ts';
import {plotTickLabel} from '../../platform/plotTicks.ts';

const escape=(text:string)=>text.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]!));

/** Fixed paper size and full task frequency range, independent of UI zoom/theme. */
export function spectrumSvg(result:LpcResult,y:[number,number]){
  const width=768,height=432,left=64,right=22,top=40,bottom=52;
  const pw=width-left-right,ph=height-top-bottom,maximum=result.config.freq_max_hz;
  const X=(f:number)=>left+f/maximum*pw,Y=(v:number)=>top+(y[1]-v)/(y[1]-y[0])*ph;
  const font=result.config.font??{latin:'Times New Roman',zh:'SimSun',size_px:12};
  const family=escape([quoteFamily(font.latin),quoteFamily(font.zh),'serif'].join(','));
  const ipa=escape(['"PTB-Doulos"',quoteFamily(font.zh),'serif'].join(','));
  const path=result.spectrum.frequencies_hz.map((f,i)=>(i?'L':'M')+X(f).toFixed(3)+','+Y(result.spectrum.magnitude_db[i]).toFixed(3)).join('');
  const text=(x:number,y:number,value:string,anchor='middle',extra='')=>`<text x="${x}" y="${y}" text-anchor="${anchor}" ${extra}>${escape(value)}</text>`;
  const ticks=Array.from({length:5},(_,i)=>{
    const v=y[0]+i/4*(y[1]-y[0]),f=i/4*maximum;
    return `<line x1="${left}" x2="${width-right}" y1="${Y(v)}" y2="${Y(v)}" stroke="#ddd" stroke-dasharray="2 4"/>`+
      text(left-8,Y(v)+4,plotTickLabel(v,(y[1]-y[0])/4),'end')+text(X(f),height-bottom+21,plotTickLabel(f,maximum/4));
  }).join('');
  const label=result.label?text(left+pw/2,23,result.label,'middle',`style="font-family:${ipa}"`):'';
  const source=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" style="font-family:${family};font-size:${font.size_px}px;fill:black">
    <rect width="100%" height="100%" fill="white"/>
    <desc>LPC spectrum; y-axis ${y[0]} to ${y[1]} dB; task ${escape(result.input_sha256)}; time ${result.selection.start_s} to ${result.selection.end_s} s</desc>
    <defs><clipPath id="lpc-export-clip"><rect x="${left}" y="${top}" width="${pw}" height="${ph}"/></clipPath></defs>
    ${ticks}<path d="${path}" fill="none" stroke="black" stroke-width="1" clip-path="url(#lpc-export-clip)"/>
    <rect x="${left}" y="${top}" width="${pw}" height="${ph}" fill="none" stroke="black"/>
    ${label}${text(left+pw/2,height-9,'Frequency (Hz)')}${text(0,0,'Amplitude (dB)','middle',`transform="translate(17 ${top+ph/2}) rotate(-90)"`)}
  </svg>`;
  return {text:source,width,height};
}

