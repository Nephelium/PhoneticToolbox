import {uiMessage,uiError,localized} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
/** The owned Qt task bridge accepts one request at a time, across all modules. */
export function serialRequests(send:<T>(body:unknown)=>Promise<T>) {
  let tail:Promise<unknown>=Promise.resolve();let queued=0;
  return <T>(body:unknown):Promise<T>=>{
    if(queued>=128)return Promise.reject(uiError(uiMessage('platform.too_many_task_operations_are_waiting_retry_later')));
    queued++;const result=tail.then(()=>send<T>(body));
    tail=result.then(()=>{queued--;},()=>{queued--;});
    return result;
  };
}
