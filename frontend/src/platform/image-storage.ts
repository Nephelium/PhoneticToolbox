/** Bounded database opening: a locked profile must not leave upload/startup waiting forever. */
export function openImageDatabase(name:string,store:string,timeout=12000):Promise<IDBDatabase>{
 return new Promise((resolve,reject)=>{
  const request=indexedDB.open(name,1);let settled=false;
  const fail=(error:unknown)=>{if(settled)return;settled=true;clearTimeout(timer);reject(error);};
  const timer=setTimeout(()=>fail(Error('本机图片存储暂未响应。请正常关闭其他 PhoneticToolbox 窗口后重试。')),timeout);
  request.onupgradeneeded=()=>request.result.createObjectStore(store);
  request.onblocked=()=>fail(Error('图片设置被另一个软件窗口占用，请关闭其他 PhoneticToolbox 窗口后重试。'));
  request.onerror=()=>fail(Error('无法打开本机图片存储，请正常关闭其他 PhoneticToolbox 窗口后重试。'));
  request.onsuccess=()=>{if(settled){request.result.close();return;}settled=true;clearTimeout(timer);resolve(request.result);};
 });
}
