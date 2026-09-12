import {fontsReady,fontPayload,doulosUrl} from '../state/fonts.ts';
import doulosLicense from '../assets/Doulos-OFL.txt?raw';
const ns='http://www.w3.org/2000/svg';
let bundled:Promise<string>|undefined;
async function doulosData(){
 if(!bundled)bundled=(async()=>{const response=await fetch(doulosUrl);if(!response.ok)throw Error('Doulos SIL 字体资源无法读取。');const bytes=new Uint8Array(await response.arrayBuffer());let binary='';for(let i=0;i<bytes.length;i+=8192)binary+=String.fromCharCode(...bytes.subarray(i,i+8192));return 'data:font/ttf;base64,'+btoa(binary);})();
 return bundled;
}
export function styledSvg(source:SVGSVGElement){
 const clone=source.cloneNode(true) as SVGSVGElement;
 const originals=[source,...source.querySelectorAll<SVGElement>('*')],copies=[clone,...clone.querySelectorAll<SVGElement>('*')];
 originals.forEach((item,i)=>{const style=getComputedStyle(item);for(const key of ['fill','stroke','stroke-width','stroke-dasharray','font-family','font-size','font-weight','font-style','text-anchor','opacity'])copies[i].style.setProperty(key,style.getPropertyValue(key));});
 return clone;
}
export async function editableSvg(clone:SVGSVGElement){
 const css=fontPayload.value?.css||`@font-face{font-family:PTB-Doulos;src:url("${new URL(doulosUrl,location.href).href}")}`;
 await fontsReady();const data=await doulosData();const style=document.createElementNS(ns,'style');style.textContent=css.replaceAll(new URL(doulosUrl,location.href).href,data);clone.prepend(style);
 const description=document.createElementNS(ns,'desc');description.textContent='IPA: Doulos SIL. Other text remains editable and requires the selected fonts on the viewing device.\n'+doulosLicense;clone.prepend(description);
 clone.setAttribute('xmlns',ns);return new XMLSerializer().serializeToString(clone);
}
// SVG-as-image cannot reliably access document-local font faces. Snapshot the
// text geometry, rasterize the non-text scene, and paint text with the same
// browser Canvas font environment as the visible page. No system font copying.
export async function paintSvg(canvas:HTMLCanvasElement,source:string,width:number,height:number){
 await fontsReady();
 const root=new DOMParser().parseFromString(source,'image/svg+xml').documentElement as unknown as SVGSVGElement;
 const host=document.createElement('div');host.style.cssText='position:fixed;left:-100000px;top:0;visibility:hidden;pointer-events:none';
 root.style.width=width+'px';root.style.height=height+'px';root.style.maxWidth='none';root.style.display='block';host.append(root);document.body.append(host);
 type Text={text:string;matrix:DOMMatrix;x:number;y:number;font:string;fill:string;anchor:CanvasTextAlign;opacity:number};const labels:Text[]=[];
 try{
  for(const element of root.querySelectorAll<SVGTextElement>('text')){
   const style=getComputedStyle(element),matrix=element.getCTM();if(!matrix)throw Error('图中文字定位失败。');
   const text=[...element.childNodes].filter(n=>n.nodeType===Node.TEXT_NODE||n.nodeName==='tspan').map(n=>n.textContent).join('');
   const opacity=Number(style.opacity);labels.push({text,matrix:new DOMMatrix([matrix.a,matrix.b,matrix.c,matrix.d,matrix.e,matrix.f]),x:element.x.baseVal[0]?.value??0,y:element.y.baseVal[0]?.value??0,font:`${style.fontStyle} ${style.fontWeight} ${style.fontSize} ${style.fontFamily}`,fill:style.fill,anchor:style.textAnchor==='middle'?'center':style.textAnchor==='end'?'right':'left',opacity});
   element.remove();
  }
 }finally{host.remove();}
 const raw=new XMLSerializer().serializeToString(root),url=URL.createObjectURL(new Blob([raw],{type:'image/svg+xml;charset=utf-8'}));
 try{
  const picture=new Image();await new Promise<void>((resolve,reject)=>{const timer=setTimeout(()=>{picture.src='';reject(Error('图像导出超时，请重试。'));},15000);picture.onload=()=>{clearTimeout(timer);resolve();};picture.onerror=()=>{clearTimeout(timer);reject(Error('图像转换失败。'));};picture.src=url;});
  const ctx=canvas.getContext('2d');if(!ctx)throw Error('图像绘制环境不可用。');ctx.drawImage(picture,0,0,canvas.width,canvas.height);
  const sx=canvas.width/width,sy=canvas.height/height;
  for(const label of labels){const m=label.matrix;ctx.save();ctx.setTransform(sx*m.a,sy*m.b,sx*m.c,sy*m.d,sx*m.e,sy*m.f);ctx.font=label.font;ctx.fillStyle=label.fill;ctx.textAlign=label.anchor;ctx.textBaseline='alphabetic';ctx.globalAlpha=label.opacity;ctx.fillText(label.text,label.x,label.y);ctx.restore();}
 }finally{URL.revokeObjectURL(url);}
}
