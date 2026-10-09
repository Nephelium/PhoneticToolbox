import type {CSSProperties} from 'vue';
export const safeColor=(value:unknown):string|undefined=>typeof value==='string'&&(/^(?:#[0-9a-f]{3,8}|(?:rgb|hsl)a?\([\d\s.,%+-]+\)|[a-z]{1,24})$/i.test(value))?value:undefined;
const unit=(value:unknown,max:number):string|undefined=>{
  if(typeof value==='number'&&Number.isFinite(value)&&value>=0&&value<=max)return value+'px';
  if(typeof value==='string'&&/^(?:\d+\.?\d*|\.\d+)(?:px|em|rem|%)$/.test(value)&&parseFloat(value)<=max)return value;
  return undefined;
};
export function contentStyle(attrs:Record<string,unknown>={}):CSSProperties {
  const style:CSSProperties={};
  if(['left','center','right','justify'].includes(String(attrs.textAlign)))style.textAlign=attrs.textAlign as CSSProperties['textAlign'];
  const height=Number(attrs.lineHeight);if(Number.isFinite(height)&&height>=1&&height<=4)style.lineHeight=String(height);
  for(const prop of ['marginTop','marginBottom'] as const){const value=unit(attrs[prop],200);if(value)style[prop]=value;}
  const indent=unit(attrs.indent,200);if(indent)style.paddingInlineStart=indent;
  return style;
}
export function textStyle(attrs:Record<string,unknown>={}):CSSProperties {
  const style:CSSProperties={};
  for(const [key,css] of [['color','--manual-color-light'],['colorDark','--manual-color-dark'],['backgroundColor','--manual-bg-light'],['backgroundColorDark','--manual-bg-dark']] as const){const value=safeColor(attrs[key]);if(value)style[css]=value;}
  const size=unit(attrs.fontSize,96);if(size)style.fontSize=size;
  if(typeof attrs.fontFamily==='string'&&attrs.fontFamily.length<=100&&!/[\x00-\x1f"'\\/;{}<>]/.test(attrs.fontFamily))style.fontFamily='"'+attrs.fontFamily+'"';
  return style;
}
export function boundedSpan(value:unknown):number|undefined {const number=Number(value);return Number.isInteger(number)&&number>=1&&number<=100?number:undefined;}
export function mediaWidth(value:unknown):string|undefined {if(typeof value==='number'&&Number.isFinite(value)&&value>0&&value<=8000)return value+'px';if(typeof value==='string'&&/^(?:\d+\.?\d*|\.\d+)(?:px|%)$/.test(value)&&parseFloat(value)>0&&parseFloat(value)<=8000)return value;return undefined;}
