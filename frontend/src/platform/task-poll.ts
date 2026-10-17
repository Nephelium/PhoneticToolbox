/** A queued task needs fewer reads; abort wakes the caller to send cancellation. */
export function taskPollDelay(state:string){return state==='queued'?1200:state==='cancel_requested'?350:500;}
export function waitForTaskPoll(state:string,signal?:AbortSignal):Promise<void>{
 if(signal?.aborted)return Promise.resolve();
 return new Promise(resolve=>{
  const done=()=>{clearTimeout(timer);signal?.removeEventListener('abort',done);resolve();};
  const timer=setTimeout(done,taskPollDelay(state));signal?.addEventListener('abort',done,{once:true});
 });
}

/** Per-owner/project, bounded, short-lived GET coalescing. Never caches errors. */
export function sharedJobReads<T>(load:(key:string)=>Promise<T>,ttl=500){
 const entries=new Map<string,{until:number;promise:Promise<T>}>();let generation=0;
 return {
  read(key:string):Promise<T>{
   const prior=entries.get(key);if(prior&&prior.until>Date.now())return prior.promise;
   if(prior)entries.delete(key);
   while(entries.size>=64)entries.delete(entries.keys().next().value!);
   const version=generation,entry:{until:number;promise:Promise<T>}={until:Infinity,promise:Promise.resolve(undefined as T)};
   entry.promise=load(key).then(value=>{if(version===generation&&entries.get(key)===entry)entry.until=Date.now()+ttl;return value;},error=>{if(entries.get(key)===entry)entries.delete(key);throw error;});
   entries.set(key,entry);return entry.promise;
  },
  clear(){generation++;entries.clear();}
 };
}
