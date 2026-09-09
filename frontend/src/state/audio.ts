import { reactive } from 'vue';
import type { AudioAsset } from '../platform/types.ts';
export const playback=reactive({playing:false,position:0,volume:0.7,error:''});
let context:AudioContext|undefined, source:AudioBufferSourceNode|undefined,gain:GainNode|undefined;
let started=0,origin=0,limit=0,raf=0,generation=0;
const channel=typeof BroadcastChannel!=='undefined'?new BroadcastChannel('ptb-v3-audio'):null;
if(channel) channel.onmessage=()=>pause();
function tick(){ if(!context || !playback.playing)return;playback.position=Math.min(limit,origin+context.currentTime-started);raf=requestAnimationFrame(tick); }
export function pause(){generation++;playback.playing=false;cancelAnimationFrame(raf);if(source){source.onended=null;source.stop();source.disconnect();source=undefined;}gain?.disconnect();gain=undefined;}
export function stop(){pause();playback.position=0;}
export function volume(value:number){playback.volume=value;if(gain)gain.gain.value=value;}
export async function play(asset:AudioAsset,start:number,end:number,selectedChannel:number){
  pause();const request=generation;playback.error='';
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
  }catch(error){stop();playback.error=error instanceof Error?error.message:'播放失败，请检查输出设备。';}
}
