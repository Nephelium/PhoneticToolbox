import {paintSvg,styledSvg} from './svg-fonts.ts';
import {exportSize,png300dpi} from './png.ts';
import {waveformExportColor} from './waveform-color.ts';

/** Snapshot only panel titles, axes and data; UI legends never enter PNGs. */
export function scientificPlotsSvg(panels:HTMLElement[]){
  if(!panels.length)throw Error('图形尚未就绪。');
  const charts=panels.map(panel=>{const svg=panel.querySelector<SVGSVGElement>('.scientific-plot>svg');if(!svg)throw Error('图形尚未就绪。');return {panel,svg,copy:styledSvg(svg)};});
  const ns='http://www.w3.org/2000/svg',root=document.createElementNS(ns,'svg');
  const columns=charts.length>1?2:1,gap=16,padding=36;
  const w=Math.max(...charts.map(c=>c.svg.viewBox.baseVal.width)),h=Math.max(...charts.map(c=>c.svg.viewBox.baseVal.height));
  const width=columns*w+(columns-1)*gap,height=Math.ceil(charts.length/columns)*(h+padding+gap);
  root.setAttribute('xmlns',ns);root.setAttribute('viewBox',`0 0 ${width} ${height}`);
  root.setAttribute('width',String(width));root.setAttribute('height',String(height));
  const bg=document.createElementNS(ns,'rect');bg.setAttribute('width','100%');bg.setAttribute('height','100%');bg.setAttribute('fill','white');root.append(bg);
  charts.forEach(({panel,svg,copy},i)=>{
    const x=i%columns*(w+gap),y=Math.floor(i/columns)*(h+padding+gap),font=getComputedStyle(svg);
    const title=document.createElementNS(ns,'text');title.textContent=panel.querySelector('h3,h2')?.textContent??svg.getAttribute('aria-label')??'';
    title.setAttribute('x',String(x+w/2));title.setAttribute('y',String(y+22));title.setAttribute('text-anchor','middle');
    title.style.fontFamily=font.fontFamily;title.style.fontSize=font.fontSize;title.style.fill='#182332';root.append(title);
    copy.setAttribute('x',String(x));copy.setAttribute('y',String(y+padding));copy.setAttribute('width',String(w));copy.setAttribute('height',String(h));copy.style.width='';copy.style.height='';
    copy.querySelectorAll('text').forEach(t=>{t.style.fill='#475569';});
    copy.querySelectorAll<SVGElement>('.scientific-trace[data-export-color]').forEach(trace=>{
      const color=trace.dataset.exportColor==='var(--waveform-color)'?waveformExportColor('#174b82'):trace.dataset.exportColor!;
      for(const part of [trace,...trace.querySelectorAll<SVGElement>('*')]){
        if(part.style.stroke!=='none')part.style.stroke=color;
        if(part.style.fill!=='none')part.style.fill=color;
      }
      trace.style.opacity='1';
    });
    // Dark UI grids are too faint on white paper. Only grid/frame strokes.
    copy.querySelectorAll<SVGElement>('line,rect').forEach(part=>{
      if(part.closest('.scientific-trace')||part.closest('clipPath')||part.querySelector('title'))return;
      part.style.stroke=part.tagName==='rect'?'#64748b':'#cbd5e1';
    });
    root.append(copy);
  });
  return {text:new XMLSerializer().serializeToString(root),width,height};
}

export async function scientificPlotsPng(panels:HTMLElement[]){
  const {text,width,height}=scientificPlotsSvg(panels),size=exportSize(width,height);
  const canvas=document.createElement('canvas');canvas.width=size.width;canvas.height=size.height;
  try{await paintSvg(canvas,text,width,height);const blob=await new Promise<Blob>((resolve,reject)=>canvas.toBlob(b=>b?resolve(b):reject(Error('PNG 编码失败。')),'image/png'));return new Blob([png300dpi(new Uint8Array(await blob.arrayBuffer()))],{type:'image/png'});}
  finally{canvas.width=canvas.height=1;}
}

export function downloadPng(blob:Blob,name:string){const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),5000);}
