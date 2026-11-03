import {uiMessage,uiError,localized} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
import {receiveDisplayError} from '../i18n/index.ts';
import type {AudioAsset} from './types.ts';
export function decodeWav(buffer:ArrayBuffer,name:string,signal?:AbortSignal,maxBytes=64_000_000):Promise<AudioAsset>{
 return new Promise((resolve,reject)=>{if(signal?.aborted){reject(new DOMException('Preview cancelled','AbortError'));return;}const worker=new Worker(new URL('./wav.worker.ts',import.meta.url),{type:'module'});
  const cleanup=()=>{clearTimeout(timer);signal?.removeEventListener('abort',abort);worker.terminate();};
  const abort=()=>{cleanup();reject(new DOMException('Preview cancelled','AbortError'));};
  const timer=setTimeout(()=>{cleanup();reject(uiError(uiMessage('platform.audio_decoding_timed_out_choose_shorter_audio')));},20000);
  signal?.addEventListener('abort',abort,{once:true});
  worker.onmessage=event=>{cleanup();if(event.data.error)reject(receiveDisplayError(event.data.display,event.data.error));else resolve(event.data.asset);};
  worker.onerror=()=>{cleanup();reject(uiError(uiMessage('platform.the_audio_preview_worker_could_not_be_started')));};
  worker.postMessage({buffer,name,maxBytes},[buffer]);
 });
}
