import { reactive,shallowRef,toRaw } from 'vue';
import type { AudioAsset } from '../platform/types.ts';
export const playback=reactive({playing:false,position:0,volume:0.7,error:''});
const currentAudio=shallowRef<{asset:AudioAsset;channel:number}>();
export function isCurrentAudio(asset:AudioAsset|null,selectedChannel:number){return !!asset&&currentAudio.value?.asset===toRaw(asset)&&currentAudio.value?.channel===selectedChannel;}
let context:AudioContext|undefined, source:AudioBufferSourceNode|undefined,gain:GainNode|undefined;
let started=0,origin=0,limit=0,raf=0,generation=0;
const channel=typeof BroadcastChannel!=='undefined'?new BroadcastChannel('ptb-v3-audio'):null;
if(channel) channel.onmessage=()=>pause();
function tick(){ if(!context || !playback.playing)return;playback.position=Math.min(limit,origin+context.currentTime-started);raf=requestAnimationFrame(tick); }
export function pause(){generation++;playback.playing=false;cancelAnimationFrame(raf);if(source){source.onended=null;source.stop();source.disconnect();source=undefined;}gain?.disconnect();gain=undefined;}
export function stop(){pause();playback.position=0;currentAudio.value=undefined;}
export function volume(value:number){playback.volume=value;if(gain)gain.gain.value=value;}
export function seek(asset:AudioAsset,position:number,start:number,end:number,selectedChannel:number){
 const continuing=playback.playing&&isCurrentAudio(asset,selectedChannel);pause();
 currentAudio.value={asset:toRaw(asset),channel:selectedChannel};
 playback.position=Math.max(start,Math.min(position,end,asset.duration));
 if(continuing&&playback.position<end)void play(asset,playback.position,end,selectedChannel);
}
export async function play(asset:AudioAsset,start:number,end:number,selectedChannel:number){
  pause();const request=generation;playback.error='';playback.position=Math.max(0,start);currentAudio.value={asset:toRaw(asset),channel:selectedChannel};
  try {
    context??=new AudioContext();await context.resume();
    if(request!==generation)return;
    if(context.state!=='running')throw Error('音频设备未就绪，请重新点击播放。');
    channel?.postMessage('pause-other-windows');
    const buffer=context.createBuffer(1,asset.frames,asset.sampleRate);
    buffer.copyToChannel(new Float32Array(asset.channels[selectedChannel]),0);
    source=context.createBufferSource();gain=context.createGain();gain.gain.value=playback.volume;
    source.buffer=buffer;source.connect(gain).connect(context.destination);
    origin=Math.max(0,start);limit=Math.min(end,asset.duration);started=context.currentTime;
    source.onended=()=>{playback.playing=false;playback.position=limit;cancelAnimationFrame(raf);source?.disconnect();source=undefined;gain?.disconnect();gain=undefined;};
    source.start(0,origin,Math.max(0,limit-origin));playback.position=origin;playback.playing=true;tick();
  }catch(error){if(request!==generation)return;stop();playback.error=error instanceof Error?error.message:'播放失败，请检查输出设备。';}
}
