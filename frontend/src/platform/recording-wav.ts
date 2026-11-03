import {uiMessage,uiError,localized} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
/** The sample rate is the AudioContext rate, never an inferred hardware rate. */
export function floatWave(blocks:Float32Array[][],rate:number,channels:number):ArrayBuffer {
 if(!Number.isInteger(rate)||rate<8000||rate>96000||!Number.isInteger(channels)||channels<1||channels>8)throw uiError(uiMessage('platform.invalid_recording_format'));
 const frames=blocks.reduce((sum,block)=>{if(block.length!==channels||block.some(c=>c.length!==block[0].length))throw uiError(uiMessage('platform.recording_channel_lengths_do_not_match'));return sum+block[0].length;},0);
 if(!frames||frames*channels*4>99_999_956)throw uiError(uiMessage('platform.the_recording_is_empty_or_exceeds_the_100_mb'));
 const bytes=new ArrayBuffer(44+frames*channels*4),v=new DataView(bytes);
 const ascii=(offset:number,text:string)=>{for(let i=0;i<text.length;i++)v.setUint8(offset+i,text.charCodeAt(i));};
 ascii(0,'RIFF');v.setUint32(4,bytes.byteLength-8,true);ascii(8,'WAVEfmt ');v.setUint32(16,16,true);v.setUint16(20,3,true);v.setUint16(22,channels,true);v.setUint32(24,rate,true);v.setUint32(28,rate*channels*4,true);v.setUint16(32,channels*4,true);v.setUint16(34,32,true);ascii(36,'data');v.setUint32(40,bytes.byteLength-44,true);
 let offset=44;for(const block of blocks)for(let i=0;i<block[0].length;i++)for(let c=0;c<channels;c++){const sample=block[c][i];if(!Number.isFinite(sample))throw uiError(uiMessage('platform.the_recording_contains_invalid_samples'));v.setFloat32(offset,sample,true);offset+=4;}
 return bytes;
}
