import {parseWav} from './wav.ts';
import {errorDisplay,runtimeError} from '../i18n/runtime-errors.ts';
self.onmessage=(event:MessageEvent<{buffer:ArrayBuffer;name:string;maxBytes?:number}>)=>{
 try{const asset=parseWav(event.data.buffer,event.data.name,event.data.maxBytes);const transfer=asset.channels.map(c=>c.buffer);for(const p of asset.peaks??[])transfer.push(p.min.buffer,p.max.buffer);self.postMessage({asset}, {transfer});}
 catch(e){const display=errorDisplay(e)??runtimeError('platform.audio_decoding_failed').display;self.postMessage({error:display.key,display});}
};
