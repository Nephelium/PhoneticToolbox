/** A session-owned Worker. No VTL state is shared with other projects/modules. */
export type LocalVocalExecutor={request:(op:string,body?:any)=>Promise<any>;close:()=>void};
type Task={id:number;op:string;body:any;resolve:(value:any)=>void;reject:(error:Error)=>void};
const stops=new Set(['animation/stop','deactivate']);
const renders=new Set(['animation/prepare','preview']);
export class BrowserVocal implements LocalVocalExecutor {
 private worker:Worker|undefined;
 private current:Task|undefined;
 private queue:Task[]=[];
 private checkpoint:any;
 private next=0;
 private closed=false;
 private timer:ReturnType<typeof setTimeout>|undefined;
 private create:()=>Worker;
 private generationBase=0;
 private lastGeneration=0;
 constructor(create=()=>new Worker(new URL('./vocal.worker.ts',import.meta.url))){this.create=create;}
 request(op:string,body:any={}):Promise<any>{
  if(this.closed)return Promise.reject(new DOMException('声道工作区已关闭。','AbortError'));
  if(stops.has(op)&&this.current&&renders.has(this.current.op)){
   // Synchronous native Wasm cannot process a message mid-call. Termination
   // cancels it immediately. Only acknowledged edits restore into a new Worker.
   this.reset(new DOMException('已取消生成','AbortError'));
   return Promise.resolve({active:false,mode:'idle',cancelled:true});
  }
  if(stops.has(op)){
   // Stopping a short PCM/pose operation can wait for its current block. Keep
   // the initialized engine and cached animation; only long renders need a
   // Worker reset. Never start an already queued render after deactivation.
   this.queue=this.queue.filter(task=>{if(renders.has(task.op)||task.op==='audio/pull'){task.reject(new DOMException('已取消生成','AbortError'));return false;}return true;});
  }
  if(this.queue.length>=8)return Promise.reject(Error('声道请求较多，请等待当前操作完成。'));
  return new Promise((resolve,reject)=>{
   const task={id:++this.next,op,body:JSON.parse(JSON.stringify(body)),resolve,reject};
   // PCM outranks queued display/monitor work. Writes remain in order, and a
   // running native call is never interrupted except by explicit cancellation.
   if(op==='audio/pull'||stops.has(op))this.queue.unshift(task);else this.queue.push(task);
   this.pump();
  });
 }
 private pump(){
  if(this.closed||this.current||!this.queue.length)return;
  const task=this.current=this.queue.shift()!;
  try{
   if(!this.worker){
    this.worker=this.create();
    this.worker.onmessage=event=>{
     if(event.data.id!==this.current?.id)return;
     if(event.data.fatal){this.reset(Error(String(event.data.error)));return;}
     const done=this.current!;this.current=undefined;clearTimeout(this.timer);
     if(event.data.error)done.reject(Error(String(event.data.error)));
     else{
      if(event.data.checkpoint)this.checkpoint={...this.checkpoint,...event.data.checkpoint};
      const value=event.data.value;
      if(typeof value?.playback_generation==='number'){value.playback_generation+=this.generationBase;this.lastGeneration=Math.max(this.lastGeneration,value.playback_generation);}
      done.resolve(value);
     }
     this.pump();
    };
    this.worker.onerror=()=>this.reset(Error('浏览器声道计算已中断，请重试当前操作。'));
   }
   this.timer=setTimeout(()=>this.reset(Error('声道计算超时，请缩短序列后重试。')),renders.has(task.op)?600000:120000);
   const body={...task.body};
   if(typeof body.playback_generation==='number')body.playback_generation-=this.generationBase;
   this.worker.postMessage({id:task.id,op:task.op,body,checkpoint:this.checkpoint});
  }catch(error){this.reset(error instanceof Error?error:Error(String(error)));}
 }
 private reset(error:Error){
  clearTimeout(this.timer);this.worker?.terminate();this.worker=undefined;
  this.generationBase=this.lastGeneration+1;
  const tasks=[...(this.current?[this.current]:[]),...this.queue];this.current=undefined;this.queue=[];
  for(const task of tasks)task.reject(error);
 }
 close(){this.closed=true;this.reset(new DOMException('声道工作区已关闭。','AbortError'));this.checkpoint=undefined;}
}
