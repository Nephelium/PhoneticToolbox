/** Rasterize the wallpaper once per appearance/viewport change. Hundreds of live
 * backdrop filters cause intermittent missing tiles in Windows Qt WebEngine.
 * Buttons share this bitmap, so scrolling never samples another compositor layer. */
export interface GlassWallpaper {url:string;blur:number;opacity:number;glassBlur:number}
let revision=0,timer:ReturnType<typeof setTimeout>|undefined,key='',bitmap='';
let current:GlassWallpaper={url:'',blur:0,opacity:0,glassBlur:0};
let installed=false;
const root=()=>document.documentElement;
export function refreshGlassWallpaper(value?:GlassWallpaper){
 if(value)current=value;
 if(!installed){
  installed=true;
  window.addEventListener('resize',()=>refreshGlassWallpaper());
  new MutationObserver(()=>refreshGlassWallpaper()).observe(root(),{attributes:true,attributeFilter:['style','data-theme','data-palette','data-custom-palette']});
 }
 const style=getComputedStyle(root()),zoom=parseFloat(style.zoom)||1;
 const width=Math.ceil(innerWidth/zoom),height=Math.ceil(innerHeight/zoom),base=style.getPropertyValue('--app').trim();
 const next=JSON.stringify([current,width,height,base]);
 if(next===key)return;
 key=next;const ticket=++revision;
 clearTimeout(timer);
 if(!current.url){publish('',width,height);return;}
 const options={...current};
 timer=setTimeout(()=>{void render(options,width,height,base,ticket);},80);
}
function publish(url:string,width:number,height:number){
 const old=bitmap;bitmap=url;
 root().style.setProperty('--button-glass-image',url?`url("${url}")`:'none');
 root().style.setProperty('--button-glass-image-size',`${width}px ${height}px`);
 if(old)URL.revokeObjectURL(old);
}
async function render(options:GlassWallpaper,width:number,height:number,base:string,ticket:number){
 try{
  const image=new Image();image.src=options.url;await image.decode();
  if(ticket!==revision)return;
  const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;
  const context=canvas.getContext('2d');if(!context)return;
  context.fillStyle=base;context.fillRect(0,0,width,height);
  // Match the outer wallpaper's cover size and overscan, including page zoom.
  const inset=Math.ceil(options.blur*3),w=width+2*inset,h=height+2*inset;
  const scale=Math.max(w/image.naturalWidth,h/image.naturalHeight);
  const iw=image.naturalWidth*scale,ih=image.naturalHeight*scale;
  context.globalAlpha=options.opacity/100;
  context.filter=`blur(${Math.hypot(options.blur,options.glassBlur)}px) saturate(1.15)`;
  context.drawImage(image,(width-iw)/2,(height-ih)/2,iw,ih);
  const blob=await new Promise<Blob|null>(resolve=>canvas.toBlob(resolve,'image/png'));
  if(!blob||ticket!==revision)return;
  publish(URL.createObjectURL(blob),width,height);
 }catch{
  // Keep a plain translucent material if an image has been replaced mid-decode.
  if(ticket===revision)publish('',width,height);
 }
}
export function syncGlassWallpaperFrame(frame:HTMLIFrameElement){
 const doc=frame.contentDocument;if(!doc)return;
 const style=getComputedStyle(root()),zoom=parseFloat(style.zoom)||1,bounds=frame.getBoundingClientRect();
 for(const name of ['--button-glass-image','--button-glass-image-size'])doc.documentElement.style.setProperty(name,style.getPropertyValue(name));
 doc.documentElement.style.setProperty('--button-glass-image-position',`${-bounds.left/zoom}px ${-bounds.top/zoom}px`);
}
