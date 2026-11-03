import type {MessageDescriptor} from '../i18n/index.ts';
import {uiMessage,uiError,localized} from '../modules/module-localization.ts';
import {tr} from '../i18n/index.ts';
import {installFeedbackChannel,type FeedbackChannel} from './feedback.ts';

/** Account Cookie/CSRF stay authoritative; feedback credentials never enter the browser. */
export function installWebFeedback(account:string,csrf:()=>string){
 const controller=new AbortController(),listeners=new Set<(id:string,payload:string)=>void>();
 const operations=new Set(['status','health','unread','messages','send','read','setMailNotifications']);
 const channel:FeedbackChannel={ready:{connect:fn=>listeners.add(fn),disconnect:fn=>listeners.delete(fn)},request(id,payload){
  void Promise.resolve().then(async()=>{
   const {operation,args}=JSON.parse(payload);if(!operations.has(operation))throw uiError(uiMessage('platform.this_feedback_operation_is_unavailable'));
   const response=await fetch('/api/v1/web-feedback',{method:'POST',credentials:'same-origin',cache:'no-store',redirect:'error',signal:controller.signal,
    headers:{'Content-Type':'application/json','X-PTB-Account':account,'X-CSRF-Token':csrf()},body:JSON.stringify({operation,args:args??{}})});
   const value=await response.json();
   if(!response.ok)throw uiError(({authentication_required:uiMessage('platform.your_login_has_expired_log_in_again'),account_changed:uiMessage('platform.the_account_has_changed_reopen_feedback'),verified_email_required:uiMessage('platform.this_account_has_no_verified_email_address_contact_the'),message_rate_limited:uiMessage('platform.messages_are_being_sent_too_frequently_try_again_later'),invalid_image:uiMessage('platform.the_image_could_not_be_read_or_exceeds_the'),invalid_message:uiMessage('platform.enter_text_or_attach_an_image_and_check_the')} as Record<string,MessageDescriptor>)[value.detail]??uiMessage('platform.the_feedback_service_is_temporarily_unavailable_retry_later'));
   return value;
  }).then(value=>{if(!controller.signal.aborted)for(const fn of listeners)fn(id,JSON.stringify({ok:true,value}));},error=>{if(!controller.signal.aborted)for(const fn of listeners)fn(id,JSON.stringify({ok:false,error:error instanceof Error?error.message:tr('platform.the_feedback_connection_failed_try_again')}));});
 }};
 installFeedbackChannel(channel);return ()=>{controller.abort();installFeedbackChannel(undefined);listeners.clear();};
}
