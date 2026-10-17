/** One on-demand scientific Worker per active account/workspace adapter. */
export class BrowserScience {
 private worker:Worker|undefined;
 private next=0;
 private queue:{id:number;operation:string;raw:ArrayBuffer;args:unknown;resolve:(value:any)=>void;reject:(error:Error)=>void;signal?:AbortSignal;abort:()=>void}[]=[];
 private current:typeof this.queue[number]|undefined;
 private timer:ReturnType<typeof setTimeout>|undefined;
 private closed=false;
 private create:()=>Worker;
 constructor(create=()=>new Worker(new URL('./science.worker.ts',import.meta.url))){this.create=create;}
 run<T>(operation:string,raw:ArrayBuffer,args:unknown,signal?:AbortSignal):Promise<T>{
  if(this.closed)return Promise.reject(Error('Browser workspace closed'));
  if(signal?.aborted)return Promise.reject(new DOMException('Cancelled','AbortError'));
  if(raw.byteLength>100_000_000||this.queue.length>=8)return Promise.reject(Error('Browser compute budget exceeded'));
  return new Promise((resolve,reject)=>{
   // Vue settings may contain reactive arrays. Worker messages carry the same
   // immutable JSON snapshot used by the HTTP contract, never those proxies.
   const item={id:++this.next,operation,raw,args:JSON.parse(JSON.stringify(args)),resolve,reject,signal,abort:()=>{}};
   item.abort=()=>{if(this.current===item)this.reset(new DOMException('Cancelled','AbortError'));else{this.queue=this.queue.filter(v=>v!==item);item.signal?.removeEventListener('abort',item.abort);reject(new DOMException('Cancelled','AbortError'));}};
   signal?.addEventListener('abort',item.abort,{once:true});this.queue.push(item);this.pump();
  });
 }
 private pump(){
  if(this.closed||this.current||!this.queue.length)return;
  const item=this.current=this.queue.shift()!;
  try{
   if(!this.worker){this.worker=this.create();this.worker.onmessage=event=>{if(event.data.id!==this.current?.id)return;if(event.data.fatal){this.reset(Error(String(event.data.error)));return;}const task=this.current!;this.current=undefined;clearTimeout(this.timer);task.signal?.removeEventListener('abort',task.abort);if(event.data.error)task.reject(Error(String(event.data.error)));else task.resolve(event.data.result);this.pump();};this.worker.onerror=()=>this.reset(Error('Browser scientific worker unavailable'));}
   // Includes a first, on-demand runtime download; main thread stays responsive.
   this.timer=setTimeout(()=>this.reset(Error('Browser scientific computation timed out')),120000);
   this.worker.postMessage({id:item.id,operation:item.operation,raw:item.raw,args:item.args},[item.raw]);
  }catch(error){this.reset(error instanceof Error?error:Error(String(error)));}
 }
 private reset(error:Error){
  clearTimeout(this.timer);this.worker?.terminate();this.worker=undefined;
  const tasks=[...(this.current?[this.current]:[]),...this.queue];this.current=undefined;this.queue=[];
  for(const item of tasks){item.signal?.removeEventListener('abort',item.abort);item.reject(error);}
 }
 close(){this.closed=true;this.reset(new DOMException('Workspace closed','AbortError'));}
}
