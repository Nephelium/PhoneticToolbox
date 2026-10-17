import {installFeedbackChannel,type FeedbackChannel} from './feedback.ts';

/** Account Cookie/CSRF stay authoritative; feedback credentials never enter the browser. */
export function installWebFeedback(account:string,csrf:()=>string){
 const controller=new AbortController(),listeners=new Set<(id:string,payload:string)=>void>();
 const operations=new Set(['status','health','unread','messages','send','read','setMailNotifications']);
 const channel:FeedbackChannel={ready:{connect:fn=>listeners.add(fn),disconnect:fn=>listeners.delete(fn)},request(id,payload){
  void Promise.resolve().then(async()=>{
   const {operation,args}=JSON.parse(payload);if(!operations.has(operation))throw Error('此反馈操作不可用。');
   const response=await fetch('/api/v1/web-feedback',{method:'POST',credentials:'same-origin',cache:'no-store',redirect:'error',signal:controller.signal,
    headers:{'Content-Type':'application/json','X-PTB-Account':account,'X-CSRF-Token':csrf()},body:JSON.stringify({operation,args:args??{}})});
   const value=await response.json();
   if(!response.ok)throw Error(({authentication_required:'登录已过期，请重新登录。',account_changed:'账号已切换，请重新打开反馈。',verified_email_required:'当前账号没有已验证的邮箱，请联系维护者。',message_rate_limited:'消息发送过于频繁，请稍后重试。',invalid_image:'图片无法读取或超过大小限制。',invalid_message:'请输入文字或添加图片，并检查内容长度。'} as Record<string,string>)[value.detail]??'反馈服务暂不可用，请稍后重试。');
   return value;
  }).then(value=>{if(!controller.signal.aborted)for(const fn of listeners)fn(id,JSON.stringify({ok:true,value}));},error=>{if(!controller.signal.aborted)for(const fn of listeners)fn(id,JSON.stringify({ok:false,error:error instanceof Error?error.message:'反馈连接失败，请重试。'}));});
 }};
 installFeedbackChannel(channel);return ()=>{controller.abort();installFeedbackChannel(undefined);listeners.clear();};
}
