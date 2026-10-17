import {parseWav} from './wav.ts';
self.onmessage=(event:MessageEvent<{buffer:ArrayBuffer;name:string;maxBytes?:number}>)=>{
 try{const asset=parseWav(event.data.buffer,event.data.name,event.data.maxBytes);const transfer=asset.channels.map(c=>c.buffer);for(const p of asset.peaks??[])transfer.push(p.min.buffer,p.max.buffer);self.postMessage({asset}, {transfer});}
 catch(e){self.postMessage({error:e instanceof Error?e.message:'音频解码失败。'});}
};
