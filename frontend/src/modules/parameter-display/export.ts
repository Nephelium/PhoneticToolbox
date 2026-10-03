import {exportSize,png300dpi} from './png.ts';
import {paintSvg} from '../../design/svg-fonts.ts';
import {withFigureTitle} from '../../design/figure-title.ts';
import {waveformExportColor} from '../../design/waveform-color.ts';

const ns='http://www.w3.org/2000/svg';
// Detached print palette: never switch the live page's theme during export.
const palette:Record<string,string>={'--text':'#182332','--muted':'#475569','--border':'#cbd5e1','--panel':'#ffffff','--selection':'#e9eff6','--accent':'#245ab5','--warning':'#946000','--teal':'#007a79','--danger':'#c93445','--violet':'#7649a8','--success':'#26713f'};
function node<K extends keyof SVGElementTagNameMap>(tag:K,attributes:Record<string,string|number>={},text?:string){
  const element=document.createElementNS(ns,tag);
  for(const [key,value] of Object.entries(attributes))element.setAttribute(key,String(value));
  if(text!==undefined)element.textContent=text;return element;
}
function printed(source:SVGSVGElement){
  const clone=source.cloneNode(true) as SVGSVGElement;
  const originals=[source,...source.querySelectorAll<SVGElement>('*')],copies=[clone,...clone.querySelectorAll<SVGElement>('*')];
  originals.forEach((element,i)=>{
    const target=copies[i],computed=getComputedStyle(element);
    for(const property of ['fill','stroke','stroke-width','stroke-dasharray','font-family','font-size','font-weight','text-anchor','opacity']){
      let value=property.startsWith('font-')?computed.getPropertyValue(property):(element.getAttribute(property)??computed.getPropertyValue(property));
      value=value.replace(/var\((--[\w-]+)\)/g,(_,key)=>palette[key]??computed.getPropertyValue(key));
      if(value)target.style.setProperty(property,value);
    }
    if(element.tagName==='text')target.style.fill=palette['--text'];
    for(const key of Object.keys(palette))target.style.setProperty(key,palette[key]);
  });
  clone.querySelectorAll('.wave-line').forEach(e=>{(e as SVGElement).style.stroke=waveformExportColor(palette['--accent']);});
  clone.querySelectorAll('.wave-baseline').forEach(e=>{(e as SVGElement).style.stroke=palette['--border'];});
  clone.querySelectorAll('.wave-selection').forEach(e=>{(e as SVGElement).style.fill=palette['--selection'];});
  clone.querySelectorAll('.playback-cursor').forEach(e=>e.remove());
  clone.style.background='white';clone.style.width='';clone.style.height='';return clone;
}

export interface WholeFigure {
  chart:SVGSVGElement; waveform:HTMLElement; title:string;
  start:number; end:number; plotLeft:number; plotRight:number;
}
// Snapshot synchronously before awaiting rasterization so tab/selection changes
// cannot mix audio and parameter data from different moments.
export function wholeFigureSvg(input:WholeFigure){
  const {chart,waveform,title,start,end,plotLeft,plotRight}=input;
  const width=chart.viewBox.baseVal.width,chartHeight=chart.viewBox.baseVal.height;
  const fontScale=(parseFloat(getComputedStyle(chart).getPropertyValue("--figure-size"))||12)/12;
  const waves=[...waveform.querySelectorAll<SVGSVGElement>('.wave-track > svg')];
  if(!waves.length||!Number.isFinite(start)||end<=start)throw Error('波形尚未就绪，请读取音频后重试。');
  const spec=waveform.querySelector<HTMLElement>('.spectrogram-view');
  const canvas=spec?.querySelector<HTMLCanvasElement>('canvas');
  if(spec&&(!canvas||spec.querySelector('[role=status],[role=alert]')))throw Error('语谱图尚未就绪，请等待完成或关闭语谱图后导出。');
  const trackHeight=150+60*fontScale,specHeight=canvas?210+70*fontScale:0,chartY=48*fontScale+waves.length*trackHeight+specHeight,height=chartY+chartHeight+12;
  exportSize(width,height);
  const root=node('svg',{xmlns:ns,width,height,viewBox:`0 0 ${width} ${height}`});
  root.style.fontFamily=getComputedStyle(chart).fontFamily;
  root.append(node('rect',{width,height,fill:'white'}));
  root.append(node('text',{x:width/2,y:24*fontScale,'text-anchor':'middle',fill:palette['--text'],'font-size':14*fontScale},title));
  function ticks(y:number){for(let i=0;i<5;i++)root.append(node('text',{x:plotLeft+i*(plotRight-plotLeft)/4,y,fill:palette['--muted'],'font-size':12*fontScale,'text-anchor':'middle'},(start+(end-start)*i/4).toFixed(3)));}
  waves.forEach((wave,index)=>{
    const y=48*fontScale+index*trackHeight,copy=printed(wave);
    root.append(node('text',{x:width/2,y:y+14*fontScale,'text-anchor':'middle',fill:palette['--text'],'font-size':12*fontScale},wave.parentElement?.querySelector('.track-label span')?.textContent??`声道 ${index+1}`));
    // Accept both legacy normalized geometry and the pixel-coordinate annotation
    // surface. Keep print glyphs undistorted while scaling each track's geometry.
    const trackWidth=plotRight-plotLeft,labels=[...copy.querySelectorAll('text')];
    const sourceWidth=wave.viewBox.baseVal.width,sourceHeight=wave.viewBox.baseVal.height;
    const geometry=node('g',{transform:`scale(${trackWidth/sourceWidth} ${150/sourceHeight})`});
    while(copy.firstChild)geometry.append(copy.firstChild);
    copy.append(geometry);
    labels.forEach(label=>{const x=Number(label.getAttribute('x')),ly=Number(label.getAttribute('y'));label.setAttribute('x',String(x*trackWidth/sourceWidth));label.setAttribute('y',String(ly*150/sourceHeight));label.style.fontSize=12*fontScale+'px';copy.append(label);});
    copy.setAttribute('viewBox',`0 0 ${trackWidth} 150`);
    copy.setAttribute('x',String(plotLeft));copy.setAttribute('y',String(y+22*fontScale));copy.setAttribute('width',String(trackWidth));copy.setAttribute('height','150');
    root.append(copy);ticks(y+150+42*fontScale);
  });
  if(canvas){
    const y=48*fontScale+waves.length*trackHeight;
    root.append(node('text',{x:width/2,y:y+14*fontScale,'text-anchor':'middle',fill:palette['--text'],'font-size':12*fontScale},'语谱图'));
    root.append(node('image',{x:plotLeft,y:y+25*fontScale,width:plotRight-plotLeft,height:210,preserveAspectRatio:'none',href:canvas.toDataURL('image/png')}));
    const labels=[...spec!.querySelectorAll('.frequency-axis span')].map(e=>e.textContent??'');
    labels.forEach((label,i)=>root.append(node('text',{x:plotLeft-6,y:y+33*fontScale+i*100,'text-anchor':'end','font-size':11*fontScale,fill:palette['--muted']},label)));
    ticks(y+210+48*fontScale);
  }
  const parameter=printed(chart);parameter.setAttribute('x','0');parameter.setAttribute('y',String(chartY));parameter.setAttribute('width',String(width));parameter.setAttribute('height',String(chartHeight));
  root.append(parameter);
  return {text:new XMLSerializer().serializeToString(root),width,height};
}

export async function wholeFigurePng(input:WholeFigure){
  const snapshot=wholeFigureSvg(input),size=exportSize(snapshot.width,snapshot.height);
  const canvas=document.createElement('canvas');canvas.width=size.width;canvas.height=size.height;
  try{
    await paintSvg(canvas,snapshot.text,snapshot.width,snapshot.height);
    const blob=await new Promise<Blob>((resolve,reject)=>canvas.toBlob(b=>b?resolve(b):reject(Error('PNG 编码失败。')),'image/png'));
    return new Blob([png300dpi(new Uint8Array(await blob.arrayBuffer()))],{type:'image/png'});
  }finally{canvas.width=canvas.height=1;}
}

export async function currentFigurePng(chart:SVGSVGElement){
  const title=chart.closest('.parameter-figure')?.querySelector('h3')?.textContent??'参数图';
  const {root,width,height}=withFigureTitle(printed(chart),title,parseFloat(getComputedStyle(chart).fontSize)||12);
  const text=new XMLSerializer().serializeToString(root),size=exportSize(width,height);
  const canvas=document.createElement('canvas');canvas.width=size.width;canvas.height=size.height;
  try{
    await paintSvg(canvas,text,width,height);
    const blob=await new Promise<Blob>((resolve,reject)=>canvas.toBlob(b=>b?resolve(b):reject(Error('PNG 编码失败。')),'image/png'));
    return new Blob([png300dpi(new Uint8Array(await blob.arrayBuffer()))],{type:'image/png'});
  }finally{canvas.width=canvas.height=1;}
}

export function downloadImage(blob:Blob,name:string){
  const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),5000);
}
