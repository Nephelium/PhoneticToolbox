/** Share only pending reads. A mutation is a barrier for subsequent reads. */
export function coalescedRequests(send:<T>(body:unknown)=>Promise<T>,keyFor:(body:unknown)=>string|null) {
  const pending=new Map<string,{promise:Promise<unknown>;shared:boolean}>();
  return <T>(body:unknown):Promise<T>=>{
    const key=keyFor(body);
    if(key===null){pending.clear();return send<T>(body);}
    const existing=pending.get(key);
    if(existing){existing.shared=true;return existing.promise.then(value=>structuredClone(value) as T);}
    const result=send<T>(body),entry={promise:result,shared:false};pending.set(key,entry);
    const clear=()=>{if(pending.get(key)===entry)pending.delete(key);};
    void result.then(clear,clear);
    return result.then(value=>entry.shared?structuredClone(value):value);
  };
}

const reads=new Set(['parent','parameters','reconstruction_preview','reconstructions','egg_jobs','lpc_jobs',
  'list','get','job']);
export function desktopReadKey(body:unknown):string|null {
  const request=body as {op?:string};
  return request?.op&&reads.has(request.op)?JSON.stringify(body):null;
}
