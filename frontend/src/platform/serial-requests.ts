/** The owned Qt task bridge accepts one request at a time, across all modules. */
export function serialRequests(send:<T>(body:unknown)=>Promise<T>) {
  let tail:Promise<unknown>=Promise.resolve();let queued=0;
  return <T>(body:unknown):Promise<T>=>{
    if(queued>=128)return Promise.reject(Error('等待中的任务操作过多，请稍后再试。'));
    queued++;const result=tail.then(()=>send<T>(body));
    tail=result.then(()=>{queued--;},()=>{queued--;});
    return result;
  };
}
