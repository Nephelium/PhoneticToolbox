// M10 platform port. This module cannot open sockets or choose native paths.
import {applyFonts} from './fonts.js';
const pending=new Map();let sequence=0;
export function request(op,body={}){
  return new Promise((resolve,reject)=>{
    const id=String(++sequence),timer=setTimeout(()=>{pending.delete(id);reject(Error('声道引擎请求超时'));},160000);
    pending.set(id,{resolve,reject,timer});parent.postMessage({type:'m10-request',id,op,body},'*');
  });
}
window.addEventListener('message',event=>{
  if(event.source!==parent)return;
  const data=event.data;
  if(data?.type==='m10-fonts')void applyFonts(data.fonts);
  if(data?.type==='m10-response'){
    const item=pending.get(data.id);if(!item)return;pending.delete(data.id);clearTimeout(item.timer);
    data.ok?item.resolve(data.value):item.reject(Error(data.error));
  }
  if(data?.type==='m10-theme'){document.documentElement.dataset.theme=data.theme==='dark'?'dark':'light';document.dispatchEvent(new Event('m10-theme'));}
});
parent.postMessage({type:'m10-ready'},'*');
export const response=async(op,body)=>{const value=await request(op,body);return {ok:true,json:async()=>value};};
