import {paintSvg,styledSvg} from './svg-fonts.ts';
import {exportSize,png300dpi} from './png.ts';

/** Snapshot titles, legends and scientific SVGs before any asynchronous work. */
export async function scientificPlotsPng(panels:HTMLElement[]){
  if(!panels.length)throw Error('图形尚未就绪。');
  const charts=panels.map(panel=>{const svg=panel.querySelector<SVGSVGElement>('svg');if(!svg)throw Error('图形尚未就绪。');return {panel,svg,copy:styledSvg(svg)};});
  const ns='http://www.w3.org/2000/svg',root=document.createElementNS(ns,'svg');
  const columns=charts.length>1?2:1,gap=16,padding=60;
  const w=Math.max(...charts.map(c=>c.svg.viewBox.baseVal.width)),h=Math.max(...charts.map(c=>c.svg.viewBox.baseVal.height));
  const width=columns*w+(columns-1)*gap,height=Math.ceil(charts.length/columns)*(h+padding+gap);
  const size=exportSize(width,height);
  root.setAttribute('xmlns',ns);root.setAttribute('viewBox',`0 0 ${width} ${height}`);
  root.setAttribute('width',String(width));root.setAttribute('height',String(height));
  const bg=document.createElementNS(ns,'rect');bg.setAttribute('width','100%');bg.setAttribute('height','100%');bg.setAttribute('fill','white');root.append(bg);
  charts.forEach(({panel,svg,copy},i)=>{
    const x=i%columns*(w+gap),y=Math.floor(i/columns)*(h+padding+gap),font=getComputedStyle(svg);
    const label=(value:string,dx:number,dy:number,color:string,size:string)=>{const t=document.createElementNS(ns,'text');t.textContent=value;t.setAttribute('x',String(dx));t.setAttribute('y',String(dy));t.style.fontFamily=font.fontFamily;t.style.fontSize=size;t.style.fill=color;root.append(t);};
    label(panel.querySelector('h3')?.textContent??'',x+12,y+21,'#182332','16px');
    let legendX=x+12;
    for(const legend of panel.querySelectorAll<HTMLElement>('.plot-legend>span')){label(legend.textContent??'',legendX,y+44,getComputedStyle(legend).color,font.fontSize);legendX+=legend.getBoundingClientRect().width+20;}
    copy.setAttribute('x',String(x));copy.setAttribute('y',String(y+padding));copy.setAttribute('width',String(w));copy.setAttribute('height',String(h));copy.style.width='';copy.style.height='';
    copy.querySelectorAll('text').forEach(t=>{t.style.fill='#475569';});
    root.append(copy);
  });
  const text=new XMLSerializer().serializeToString(root),canvas=document.createElement('canvas');canvas.width=size.width;canvas.height=size.height;
  try{await paintSvg(canvas,text,width,height);const blob=await new Promise<Blob>((resolve,reject)=>canvas.toBlob(b=>b?resolve(b):reject(Error('PNG 编码失败。')),'image/png'));return new Blob([png300dpi(new Uint8Array(await blob.arrayBuffer()))],{type:'image/png'});}
  finally{canvas.width=canvas.height=1;}
}

export function downloadPng(blob:Blob,name:string){const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),5000);}
