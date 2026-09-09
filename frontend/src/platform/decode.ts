import type {AudioAsset} from './types.ts';
export function decodeWav(buffer:ArrayBuffer,name:string,signal?:AbortSignal):Promise<AudioAsset>{
 return new Promise((resolve,reject)=>{if(signal?.aborted){reject(new DOMException('Preview cancelled','AbortError'));return;}const worker=new Worker(new URL('./wav.worker.ts',import.meta.url),{type:'module'});
  const cleanup=()=>{clearTimeout(timer);signal?.removeEventListener('abort',abort);worker.terminate();};
  const abort=()=>{cleanup();reject(new DOMException('Preview cancelled','AbortError'));};
  const timer=setTimeout(()=>{cleanup();reject(Error('音频解码超时，请选择较短音频。'));},20000);
  signal?.addEventListener('abort',abort,{once:true});
  worker.onmessage=event=>{cleanup();if(event.data.error)reject(Error(event.data.error));else resolve(event.data.asset);};
  worker.onerror=()=>{cleanup();reject(Error('音频预览线程无法启动。'));};
  worker.postMessage({buffer,name},[buffer]);
 });
}
