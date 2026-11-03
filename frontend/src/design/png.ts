import {messageRef} from '../i18n/index.ts';
import {uiMessage,uiError,uiErrorMessage,localized} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
// PNG pHYs layout and CRC: W3C PNG Third Edition, 11.3.4.3 and 5.5.
// Source registry: REF-PNG. Encoding itself remains the browser's canvas encoder.
const signature=[137,80,78,71,13,10,26,10];
export function exportSize(width:number,height:number){
  const w=Math.ceil(width*300/96),h=Math.ceil(height*300/96);
  if(!Number.isFinite(w)||!Number.isFinite(h)||w<1||h<1||w>16384||h>16384||w*h>32_000_000)
    throw uiError(uiMessage('state.the_full_image_exceeds_the_export_size_limit_reduce'));
  return {width:w,height:h};
}
export function png300dpi(bytes:Uint8Array):Uint8Array<ArrayBuffer>{
  if(bytes.length<33||!signature.every((v,i)=>bytes[i]===v))throw uiError(uiMessage('state.the_png_encoding_is_incomplete'));
  const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
  const physical=new Uint8Array(21),pv=new DataView(physical.buffer);
  pv.setUint32(0,9);physical.set([112,72,89,115],4);
  pv.setUint32(8,11811);pv.setUint32(12,11811);physical[16]=1;
  let crc=0xffffffff;
  for(const byte of physical.subarray(4,17)){crc^=byte;for(let bit=0;bit<8;bit++)crc=(crc>>>1)^((crc&1)?0xedb88320:0);}
  pv.setUint32(17,(crc^0xffffffff)>>>0);
  const parts:Uint8Array[]=[bytes.subarray(0,8)];let ended=false,hasHeader=false;
  for(let offset=8;offset<bytes.length;){
    if(offset+12>bytes.length)throw uiError(uiMessage('state.the_png_encoding_is_incomplete'));
    const n=view.getUint32(offset),end=offset+n+12;
    if(end>bytes.length)throw uiError(uiMessage('state.the_png_encoding_is_incomplete'));
    const type=String.fromCharCode(...bytes.subarray(offset+4,offset+8));
    if(!hasHeader){if(type!=='IHDR'||n!==13)throw uiError(uiMessage('state.invalid_png_header'));hasHeader=true;}
    if(type!=='pHYs')parts.push(bytes.subarray(offset,end));
    if(type==='IHDR')parts.push(physical);
    if(type==='IEND'){if(n!==0||end!==bytes.length)throw uiError(uiMessage('state.invalid_png_end_marker'));ended=true;}
    offset=end;
  }
  if(!ended)throw uiError(uiMessage('state.the_png_encoding_is_incomplete'));
  const result=new Uint8Array(parts.reduce((n,p)=>n+p.length,0));let offset=0;
  for(const part of parts){result.set(part,offset);offset+=part.length;}return result;
}
