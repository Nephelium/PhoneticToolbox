import {onBeforeUnmount,ref,watch,type Ref} from 'vue';
import {automaticAudioGain} from './audition.ts';
import {bufferPeak,peakGain} from '../platform/playback-gain.ts';

/** Keep native media controls/timing and apply gain only to the output graph. */
export function useMediaAudition(element:Ref<HTMLMediaElement|undefined|null>){
 const message=ref('');let context:AudioContext|undefined,gain:GainNode|undefined,source:MediaElementAudioSourceNode|undefined,controller:AbortController|undefined,ticket=0,url='',peak=0;
 const update=()=>{if(gain)gain.gain.value=peakGain(peak,automaticAudioGain.value);};
 function cancel(){ticket++;controller?.abort();controller=undefined;}
 async function start(){
  const el=element.value;if(!el||el.paused||!automaticAudioGain.value||el.muted)return;
  const src=el.currentSrc||el.src;if(!src)return;
  const request=++ticket;controller?.abort();controller=new AbortController();const signal=controller.signal;
  try{
   context??=new AudioContext();await context.resume();if(request!==ticket)return;
   const locationURL=new URL(src,location.href);if(locationURL.protocol!=='blob:'&&locationURL.origin!==location.origin)throw Error('media_origin');
   if(src!==url){url='';peak=0;update();
    if(!Number.isFinite(el.duration)||el.duration>600)throw Error('media_budget');
    const response=await fetch(src,{signal});if(!response.ok)throw Error('media_read');
    const limit=64_000_000;if(Number(response.headers.get('content-length'))>limit)throw Error('media_budget');
    const reader=response.body?.getReader(),parts:Uint8Array[]=[];let size=0;
    if(reader){try{for(;;){const part=await reader.read();if(part.done)break;size+=part.value.length;if(size>limit)throw Error('media_budget');parts.push(part.value);}}finally{await reader.cancel().catch(()=>{});}}
    else{const bytes=new Uint8Array(await response.arrayBuffer());if(bytes.length>limit)throw Error('media_budget');parts.push(bytes);size=bytes.length;}
    const bytes=new Uint8Array(size);let offset=0;for(const part of parts){bytes.set(part,offset);offset+=part.length;}
    const decoded=await context.decodeAudioData(bytes.buffer);if(request!==ticket||src!==(el.currentSrc||el.src))return;
    peak=bufferPeak(decoded);url=src;
   }
   if(!source){source=context.createMediaElementSource(el);gain=context.createGain();source.connect(gain).connect(context.destination);}
   update();message.value='';
  }catch(error){if(request!==ticket||signal.aborted)return;peak=0;update();message.value='此音频暂时无法应用自动增益，正按原始音量播放。';}
 }
 const unwatch=watch(element,(el,old)=>{cancel();old?.removeEventListener('play',start);source?.disconnect();gain?.disconnect();void context?.close();context=undefined;source=undefined;gain=undefined;url='';peak=0;message.value='';el?.addEventListener('play',start);if(el&&!el.paused)void start();},{flush:'post',immediate:true});
 const unwatchGain=watch(automaticAudioGain,()=>{cancel();update();message.value='';if(automaticAudioGain.value&&!element.value?.paused)void start();},{flush:'sync'});
 onBeforeUnmount(()=>{unwatch();unwatchGain();cancel();element.value?.removeEventListener('play',start);source?.disconnect();gain?.disconnect();void context?.close();});
 return message;
}
