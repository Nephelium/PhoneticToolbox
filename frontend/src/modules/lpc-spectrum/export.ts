import type {LpcResult} from './state.ts';
import {paintSvg} from '../../design/svg-fonts.ts';
import {png300dpi} from '../../design/png.ts';
import {spectrumSvg} from './export-scene.ts';

export async function spectrumPng(result:LpcResult,y:[number,number]){
  const scene=spectrumSvg(result,y),canvas=document.createElement('canvas');canvas.width=2400;canvas.height=1350;
  try{
    await paintSvg(canvas,scene.text,scene.width,scene.height);
    const blob=await new Promise<Blob>((resolve,reject)=>canvas.toBlob(b=>b?resolve(b):reject(Error('PNG 编码失败。')),'image/png'));
    return new Blob([png300dpi(new Uint8Array(await blob.arrayBuffer()))],{type:'image/png'});
  }finally{canvas.width=canvas.height=1;}
}
