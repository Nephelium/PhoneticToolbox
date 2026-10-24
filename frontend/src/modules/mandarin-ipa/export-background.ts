import {normalizeBackground} from '../../design/background.ts';

export interface PngBackground {mode:'image'|'color';color:string;baseColor:string;imageUrl:string;imageName:string;blur:number;opacity:number}
export function backgroundColor(value:string,fallback='#ffffff'){
  const hex=value.trim();if(/^#[\da-f]{6}$/i.test(hex))return hex.toLowerCase();
  if(/^#[\da-f]{3}$/i.test(hex))return '#'+[...hex.slice(1)].map(c=>c+c).join('').toLowerCase();
  const rgb=/^rgba?\(\s*(\d+)\s*[, ]\s*(\d+)\s*[, ]\s*(\d+)(?:\s*[,/]\s*[\d.]+)?\s*\)$/i.exec(hex);
  return rgb?'#'+rgb.slice(1,4).map(v=>Math.max(0,Math.min(255,Number(v))).toString(16).padStart(2,'0')).join(''):fallback;
}
export function pngBackgroundDefaults(value:{url:string;name:string;color:string;blur:number;opacity:number}):PngBackground{
  return {mode:value.url?'image':'color',color:backgroundColor(value.color),baseColor:backgroundColor(value.color),imageUrl:value.url,imageName:value.name,...normalizeBackground(value)};
}
/** Three separable box passes match the Gaussian variance on engines without Canvas filters. */
export function blurPixels(pixels:Uint8ClampedArray,width:number,height:number,sigma:number){
  const result=new Uint8ClampedArray(pixels);if(sigma<=0)return result;
  let low=Math.floor(Math.sqrt(4*sigma*sigma+1));if(low%2===0)low--;
  const high=low+2,count=Math.round((12*sigma*sigma-3*low*low-12*low-9)/(-4*low-4));
  const radii=Array.from({length:3},(_,i)=>Math.max(0,((i<count?low:high)-1)/2));
  let source=result,target=new Uint8ClampedArray(pixels.length);
  for(const radius of radii)for(const horizontal of [true,false]){
    const along=horizontal?width:height,across=horizontal?height:width;
    for(let line=0;line<across;line++)for(let channel=0;channel<4;channel++){
      const offset=(p:number)=>((horizontal?line*width+p:p*width+line)*4+channel);
      let sum=0;for(let p=-radius;p<=radius;p++)sum+=source[offset(Math.max(0,Math.min(along-1,p)))]!;
      for(let p=0;p<along;p++){
        target[offset(p)]=Math.round(sum/(2*radius+1));
        sum+=source[offset(Math.min(along-1,p+radius+1))]!-source[offset(Math.max(0,p-radius))]!;
      }
    }
    [source,target]=[target,source];
  }
  return source;
}
export function coverImage(width:number,height:number,imageWidth:number,imageHeight:number,blur:number){
  const inset=Math.ceil(blur*3),targetWidth=width+2*inset,targetHeight=height+2*inset,wide=targetWidth/imageWidth>=targetHeight/imageHeight;
  const w=wide?targetWidth:targetHeight*(imageWidth/imageHeight),h=wide?targetWidth*(imageHeight/imageWidth):targetHeight;
  return {x:(width-w)/2,y:(height-h)/2,width:w,height:h};
}
