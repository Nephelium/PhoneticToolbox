import {messageRef} from '../i18n/index.ts';
import {uiMessage,uiError,uiErrorMessage,localized,messageProperty,setMessageProperty} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
import { reactive,shallowRef,toRaw,watch } from 'vue';
import {automaticAudioGain} from './audition.ts';
import {peakGain,samplePeak} from '../platform/playback-gain.ts';
import type { AudioAsset } from '../platform/types.ts';
import {assertPlaybackAllowed} from '../platform/capture-lease.ts';
export const playback=reactive(messageProperty({playing:false,position:0,volume:0.7,error:''},'error'));
const currentAudio=shallowRef<{asset:AudioAsset;channel:number}>();
export function isCurrentAudio(asset:AudioAsset|null,selectedChannel:number){return !!asset&&currentAudio.value?.asset===toRaw(asset)&&currentAudio.value?.channel===selectedChannel;}
let context:AudioContext|undefined, source:AudioBufferSourceNode|undefined,gain:GainNode|undefined;
let peak=0;const peaks=new WeakMap<Float32Array,number>();
function updateGain(){if(gain)gain.gain.value=playback.volume*peakGain(peak,automaticAudioGain.value);}
watch(automaticAudioGain,updateGain,{flush:'sync'});
let started=0,origin=0,limit=0,raf=0,generation=0;
const channel=typeof BroadcastChannel!=='undefined'?new BroadcastChannel('ptb-v3-audio'):null;
if(channel) channel.onmessage=()=>pause();
function tick(){ if(!context || !playback.playing)return;playback.position=Math.min(limit,origin+context.currentTime-started);raf=requestAnimationFrame(tick); }
export function pause(){generation++;playback.playing=false;cancelAnimationFrame(raf);if(source){source.onended=null;source.stop();source.disconnect();source=undefined;}gain?.disconnect();gain=undefined;}
export function stop(){pause();playback.position=0;currentAudio.value=undefined;}
export function volume(value:number){playback.volume=value;updateGain();}
export function seek(asset:AudioAsset,position:number,start:number,end:number,selectedChannel:number){
 const continuing=playback.playing&&isCurrentAudio(asset,selectedChannel);pause();
 currentAudio.value={asset:toRaw(asset),channel:selectedChannel};
 playback.position=Math.max(start,Math.min(position,end,asset.duration));
 if(continuing&&playback.position<end)void play(asset,playback.position,end,selectedChannel);
}
export async function play(asset:AudioAsset,start:number,end:number,selectedChannel:number){
  pause();const request=generation;playback.error='';playback.position=Math.max(0,start);currentAudio.value={asset:toRaw(asset),channel:selectedChannel};
  try {
    assertPlaybackAllowed();
    context??=new AudioContext();await context.resume();
    if(request!==generation)return;
    if(context.state!=='running')throw uiError(uiMessage('state.the_audio_device_is_not_ready_click_play_again'));
    channel?.postMessage('pause-other-windows');
    const buffer=context.createBuffer(1,asset.frames,asset.sampleRate);
    buffer.copyToChannel(new Float32Array(asset.channels[selectedChannel]),0);
    const pcm=asset.channels[selectedChannel]!;peak=peaks.get(pcm)??samplePeak([pcm]);peaks.set(pcm,peak);
    source=context.createBufferSource();gain=context.createGain();updateGain();
    source.buffer=buffer;source.connect(gain).connect(context.destination);
    const audioOrigin=asset.originSeconds??0;
    origin=Math.max(audioOrigin,start);limit=Math.min(end,asset.duration,audioOrigin+asset.frames/asset.sampleRate);started=context.currentTime;
    source.onended=()=>{playback.playing=false;playback.position=limit;cancelAnimationFrame(raf);source?.disconnect();source=undefined;gain?.disconnect();gain=undefined;};
    source.start(0,origin-audioOrigin,Math.max(0,limit-origin));playback.position=origin;playback.playing=true;tick();
  }catch(error){if(request!==generation)return;stop();setMessageProperty(playback,'error',error instanceof Error?uiErrorMessage(error):uiMessage('state.playback_failed_check_your_output_device'));}
}
