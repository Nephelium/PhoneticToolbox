import {fontsReady} from '../../state/fonts.ts';
import {quoteFamily} from '../../design/fonts.ts';
import {effectiveIpaSize,type ConversionToken,type MandarinIpaDraft} from './state.ts';
import {resultMetrics} from './layout.ts';
import {backgroundColor,blurPixels,coverImage,type PngBackground} from './export-background.ts';
import {contrast} from '../../design/themes.ts';

export interface MandarinIpaExportSnapshot {
  tokens:readonly ConversionToken[];draft:MandarinIpaDraft;width:number;uiFont:string;ipaColor?:string;cellWidth?:number;standardName?:string;background?:PngBackground;
}
export interface ExportHooks {
  ready?:()=>Promise<void>;createCanvas?:()=>HTMLCanvasElement;fontAvailable?:(font:string,text:string)=>boolean;
}
interface Positioned {token:ConversionToken;x:number;y:number;width:number}

async function paintBackground(ctx:CanvasRenderingContext2D,width:number,height:number,scale:number,background?:PngBackground){
  const base=background?.mode==='image'?background.baseColor:background?.color??'#ffffff';
  ctx.fillStyle=base;ctx.fillRect(0,0,width,height);
  if(background?.mode==='image'){
    if(!/^(blob:|data:image\/(?:png|jpeg|webp);base64,)/i.test(background.imageUrl))throw Error('请选择本机背景图片后再导出。');
    const image=new Image();image.src=background.imageUrl;
    try{await image.decode();}catch{throw Error('背景图片无法读取，请重新选择图片。');}
    const blur=Math.max(0,Math.min(40,background.blur))*scale,opacity=Math.max(0,Math.min(100,background.opacity))/100;
    ctx.save();
    if(typeof (ctx as {filter?:unknown}).filter==='string'){
      const rect=coverImage(width,height,image.naturalWidth,image.naturalHeight,blur);
      ctx.globalAlpha=opacity;ctx.filter=`blur(${blur}px)`;ctx.drawImage(image,rect.x,rect.y,rect.width,rect.height);
    }else{
      // Bound the fallback bitmap, premultiply transparent input over the base colour,
      // and preserve the Gaussian radius in output coordinates.
      const ratio=Math.min(1,1024/Math.max(width,height)),layer=document.createElement('canvas');
      layer.width=Math.max(1,Math.round(width*ratio));layer.height=Math.max(1,Math.round(height*ratio));
      const buffer=layer.getContext('2d');if(!buffer)throw Error('背景绘制环境不可用。');
      buffer.fillStyle=base;buffer.fillRect(0,0,layer.width,layer.height);buffer.globalAlpha=opacity;
      const rect=coverImage(layer.width,layer.height,image.naturalWidth,image.naturalHeight,blur*ratio);
      buffer.drawImage(image,rect.x,rect.y,rect.width,rect.height);
      const data=buffer.getImageData(0,0,layer.width,layer.height);data.data.set(blurPixels(data.data,layer.width,layer.height,blur*ratio));buffer.putImageData(data,0,0);
      ctx.drawImage(layer,0,0,width,height);layer.width=layer.height=1;
    }
    ctx.restore();
  }
  // Choose default ink from the exported background, retaining all explicit colours.
  const total=[0,0,0];for(const x of [.15,.5,.85])for(const y of [.2,.5,.8]){
    const pixel=ctx.getImageData(Math.min(width-1,Math.floor(width*x)),Math.min(height-1,Math.floor(height*y)),1,1).data;
    for(let channel=0;channel<3;channel++)total[channel]!+=pixel[channel]!;
  }
  const average=total.map(value=>Math.round(value/9)),color='#'+average.map(value=>value.toString(16).padStart(2,'0')).join('');
  return {ink:(.2126*average[0]!+.7152*average[1]!+.0722*average[2]!)<140?'#f8fafc':'#182332',color};
}

const clamp=(value:number,min:number,max:number)=>Math.min(max,Math.max(min,value));
function font(size:number,family:string,bold=false,italic=false){return `${italic?'italic ':''}${bold?'700 ':'400 '}${size}px ${family}`;}
function layout(ctx:CanvasRenderingContext2D,snapshot:MandarinIpaExportSnapshot){
  const {draft,tokens}=snapshot,ipaSize=effectiveIpaSize(draft),padding=28,titleHeight=38,contentWidth=clamp(snapshot.width,420,1200)-padding*2;
  const uiFont=snapshot.uiFont||'sans-serif',hanziFont=draft.hanziFont?quoteFamily(draft.hanziFont)+', '+uiFont:uiFont,ipaFont='"PTB-Doulos", "Doulos SIL", serif';
  const {ipaHeight,unitHeight,lineAdvance,columnGap}=resultMetrics(draft);
  const widths=tokens.map(token=>{
    if(token.kind==='literal'&&token.newline)return 0;
    let width:number;
    if(token.kind==='mapped'){
      ctx.font=font(ipaSize,ipaFont);const ipaWidth=ctx.measureText(token.value).width;
      ctx.font=font(draft.hanziSize,hanziFont,draft.bold,draft.italic);const hanziWidth=ctx.measureText(token.char).width;
      width=Math.max(ipaWidth,draft.display==='paired'?hanziWidth:0)+8;
    }else{
      ctx.font=font(draft.display==='paired'?draft.hanziSize:ipaSize,hanziFont,draft.bold&&draft.display==='paired',draft.italic&&draft.display==='paired');
      width=Math.max(ctx.measureText(token.char).width,token.char.trim()?0:draft.hanziSize*.5)+8;
    }
    return width;
  });
  const equalWidth=Math.max(Math.ceil(widths.reduce((max,width)=>Math.max(max,width),0)),snapshot.cellWidth??0);
  const placed:Positioned[]=[];let x=0,row=0;
  tokens.forEach((token,index)=>{
    if(token.kind==='literal'&&token.newline){x=0;row++;return;}
    const width=draft.equalWidth?equalWidth:widths[index],gap=x>0?columnGap:0;
    if(x>0&&x+gap+width>contentWidth){x=0;row++;}else x+=gap;
    placed.push({token,x:padding+x+width/2,y:padding+titleHeight+row*lineAdvance,width});x+=width;
  });
  const rows=Math.max(1,row+1),height=Math.ceil(padding*2+titleHeight+unitHeight+(rows-1)*lineAdvance);
  return {placed,width:contentWidth+padding*2,height,padding,titleHeight,lineAdvance,ipaSize,ipaHeight,ipaFont,uiFont,hanziFont};
}

export async function renderMandarinIpaPng(snapshot:MandarinIpaExportSnapshot,hooks:ExportHooks={}){
  if(!snapshot.draft.text.trim())throw Error('请输入汉字后再导出。');
  try{
    if(hooks.ready)await hooks.ready();
    else{
      // The application installs a loaded FontFace and a CSS face with the same
      // family. Loading the whole family can fetch the unused CSS face offline.
      const bundled=[...document.fonts].find(face=>face.family==='PTB-Doulos'&&face.status==='loaded');
      if(bundled)await bundled.loaded;else await fontsReady();
      await document.fonts.ready;
    }
  }catch{throw Error('Doulos SIL 字体加载失败，请恢复字体资源后重试。');}
  // Chromium may report `document.fonts.check()` as false for a loaded local
  // face with combining IPA text. Inspect the exact named FontFace instead.
  const check=hooks.fontAvailable??(()=>[...document.fonts].some(face=>face.family==='PTB-Doulos'&&face.status==='loaded'));
  if(!check(`${effectiveIpaSize(snapshot.draft)}px "PTB-Doulos"`,'a ɑ tʰ ã ˥˩'))throw Error('Doulos SIL 字体未就绪，请等待字体加载后重试。');
  if(snapshot.draft.hanziFont)try{await new FontFace('PTB-M13-check',`local(${quoteFamily(snapshot.draft.hanziFont)})`).load();}catch{throw Error(`汉字字体 ${snapshot.draft.hanziFont} 在当前设备不可用，请选择已安装的字体后重试。`);}
  const canvas=(hooks.createCanvas??(()=>document.createElement('canvas')))(),probe=canvas.getContext('2d');if(!probe)throw Error('图像绘制环境不可用。');
  const model=layout(probe,snapshot),scale=Math.min(3,16384/model.width,16384/model.height,Math.sqrt(24_000_000/((model.width+1)*(model.height+1))));
  if(scale<1)throw Error('输出内容过长，请分段导出。');
  canvas.width=Math.max(1,Math.ceil(model.width*scale));canvas.height=Math.max(1,Math.ceil(model.height*scale));
  const ctx=canvas.getContext('2d');if(!ctx)throw Error('图像绘制环境不可用。');
  try{
  const {ink,color}=await paintBackground(ctx,canvas.width,canvas.height,scale,snapshot.background);
  const candidate=backgroundColor(snapshot.ipaColor||'#215bd6');
  const ipaColor=snapshot.draft.ipaColor||(contrast(candidate,color)>=4.5?candidate:ink),hanziColor=snapshot.draft.hanziColor||ink;
  ctx.scale(scale,scale);
  ctx.textAlign='left';ctx.textBaseline='alphabetic';ctx.fillStyle=ink;ctx.font=font(18,model.uiFont,true);ctx.fillText('国际音标',model.padding,model.padding+20);
  ctx.font=font(11,model.uiFont);ctx.fillStyle=ink;ctx.fillText(snapshot.standardName??snapshot.draft.standard,model.padding+88,model.padding+20);
  for(const item of model.placed){
    const {token,x,y}=item;
    if(token.kind==='mapped'){
      ctx.textAlign='center';ctx.textBaseline='top';ctx.font=font(model.ipaSize,model.ipaFont);ctx.fillStyle=ipaColor;ctx.fillText(token.value,x,y+2+model.ipaSize*.12);
      if(snapshot.draft.display==='paired'){
        const hanziY=y+2+model.ipaHeight+snapshot.draft.gap;ctx.font=font(snapshot.draft.hanziSize,model.hanziFont,snapshot.draft.bold,snapshot.draft.italic);ctx.fillStyle=hanziColor;ctx.fillText(token.char,x,hanziY);
        if(snapshot.draft.underline){const measured=ctx.measureText(token.char).width;ctx.beginPath();ctx.moveTo(x-measured/2,hanziY+snapshot.draft.hanziSize*1.15);ctx.lineTo(x+measured/2,hanziY+snapshot.draft.hanziSize*1.15);ctx.strokeStyle=hanziColor;ctx.lineWidth=1;ctx.stroke();}
      }
    }else{
      ctx.textAlign='center';ctx.textBaseline='top';ctx.font=font(snapshot.draft.display==='paired'?snapshot.draft.hanziSize:model.ipaSize,model.hanziFont,snapshot.draft.bold&&snapshot.draft.display==='paired',snapshot.draft.italic&&snapshot.draft.display==='paired');ctx.fillStyle=hanziColor;
      const literalY=snapshot.draft.display==='paired'?y+2+model.ipaHeight+snapshot.draft.gap:y+2+model.ipaSize*.12;ctx.fillText(token.char,x,literalY);
      if(snapshot.draft.display==='paired'&&snapshot.draft.underline&&token.char.trim()){const measured=ctx.measureText(token.char).width;ctx.beginPath();ctx.moveTo(x-measured/2,literalY+snapshot.draft.hanziSize*1.15);ctx.lineTo(x+measured/2,literalY+snapshot.draft.hanziSize*1.15);ctx.strokeStyle=hanziColor;ctx.stroke();}
    }
  }
  return await new Promise<Blob>((resolve,reject)=>canvas.toBlob(blob=>blob?resolve(blob):reject(Error('PNG 编码失败。')),'image/png'));
  }
  finally{canvas.width=canvas.height=1;}
}

export function downloadMandarinIpa(blob:Blob,now=Date.now()){
  const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=`ipa_output_${now}.png`;link.click();setTimeout(()=>URL.revokeObjectURL(url),5000);
}
