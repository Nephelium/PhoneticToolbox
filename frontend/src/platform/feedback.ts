import {uiMessage,uiError,responseError,localized} from '../modules/module-localization.ts';
import {tr,locale} from '../i18n/index.ts';
export interface FeedbackImage {id?:string;data:string;mime?:string;preview?:string}
export interface FeedbackClientInfo {source:'web'|'desktop';version?:string;packageKind?:'installer'|'portable'|'source'}
export interface FeedbackHealth {conversationTrash?:boolean;trashRetentionDays?:number}
export function supportsFeedbackTrash(health:FeedbackHealth){return health.conversationTrash===true&&health.trashRetentionDays===30;}
export interface FeedbackMessage {id:number;role:'user'|'admin';text:string;created:number;images:FeedbackImage[];mailState?:string;mailError?:string;client?:FeedbackClientInfo|null}
export interface FeedbackConversation {owner:string;email:string;emailVerified:boolean;mailConfigured:boolean;mailNotifications:boolean;status:'open'|'resolved';messages:FeedbackMessage[];cursor:number;hasMore:boolean;trashed?:boolean;deletedAt?:number|null;purgeAt?:number|null}
export interface FeedbackIdentity {connected:boolean;verified:boolean;owner:string;email:string;baseUrl:string;admin:boolean;platform?:string}
export interface FeedbackSummary {id:string;email:string;email_verified:number;status:string;lastId:number;updated:number;unread:number;client?:FeedbackClientInfo|null;trashed?:boolean;deletedAt?:number|null;purgeAt?:number|null}
export interface FeedbackEvent {id?:number;kind:string;count?:number;conversations?:number;day?:string;online?:boolean;owner?:string;messageId?:number}
export interface FeedbackChannel {request:(id:string,payload:string)=>void;ready:{connect:(fn:(id:string,payload:string)=>void)=>void;disconnect?:(fn:(id:string,payload:string)=>void)=>void};notification?:{connect:(fn:(payload:string)=>void)=>void;disconnect?:(fn:(payload:string)=>void)=>void}}
let channel:FeedbackChannel|undefined,sequence=0;
const jobs=new Map<string,{resolve:(value:any)=>void;reject:(error:Error)=>void;timer:ReturnType<typeof setTimeout>}>();
const listeners=new Set<(value:FeedbackEvent)=>void>();
function ready(id:string,payload:string){const job=jobs.get(id);if(!job)return;jobs.delete(id);clearTimeout(job.timer);try{const data=JSON.parse(payload);if(data.ok)job.resolve(data.value);else job.reject(responseError(data.error));}catch{job.reject(uiError(uiMessage('platform.the_feedback_response_could_not_be_parsed')));}}
function notification(payload:string){try{const data=JSON.parse(payload);for(const listener of listeners)listener(data);}catch{/* Malformed notifications are never interpreted as new feedback. */}}
export function installFeedbackChannel(next:FeedbackChannel|undefined){channel?.ready.disconnect?.(ready);channel?.notification?.disconnect?.(notification);for(const job of jobs.values()){clearTimeout(job.timer);job.reject(uiError(uiMessage('platform.the_feedback_service_disconnected')));}jobs.clear();channel=next;channel?.ready.connect(ready);channel?.notification?.connect(notification);}
export function onFeedbackNotification(fn:(value:FeedbackEvent)=>void){listeners.add(fn);return()=>listeners.delete(fn);}
export function feedbackCall<T>(operation:string,args:Record<string,unknown>={}):Promise<T>{const current=channel;if(!current)return Promise.reject(uiError(uiMessage('platform.use_feedback_in_the_desktop_application')));const id=`feedback_${++sequence}`;return new Promise((resolve,reject)=>{const timer=setTimeout(()=>{jobs.delete(id);reject(uiError(uiMessage('platform.the_feedback_request_timed_out_refresh_the_conversation_before')));},45000);jobs.set(id,{resolve,reject,timer});try{current.request(id,JSON.stringify({operation,args}));}catch{jobs.delete(id);clearTimeout(timer);reject(uiError(uiMessage('platform.the_feedback_request_was_not_submitted')));}});}
export async function initializeFeedbackManager(){const surface=window as unknown as {ptbFeedbackNative?:{request:(id:string,payload:string)=>void};ptbFeedbackAndroid?:{ready:typeof ready;notification:typeof notification};qt?:{webChannelTransport:unknown};QWebChannel?:new(transport:unknown,callback:(value:{objects:{feedback?:FeedbackChannel}})=>void)=>unknown};if(surface.ptbFeedbackNative){surface.ptbFeedbackAndroid={ready,notification};installFeedbackChannel({request:(id,payload)=>surface.ptbFeedbackNative!.request(id,payload),ready:{connect:()=>{}},notification:{connect:()=>{}}});return;}if(!surface.qt||!surface.QWebChannel)throw uiError(uiMessage('platform.open_this_page_using_the_separate_feedback_administration_tool'));await new Promise<void>((resolve,reject)=>{const timer=setTimeout(()=>reject(uiError(uiMessage('platform.the_administration_tool_connection_timed_out'))),15000);new surface.QWebChannel!(surface.qt!.webChannelTransport,value=>{clearTimeout(timer);if(!value.objects.feedback){reject(uiError(uiMessage('platform.the_feedback_administration_channel_is_unavailable')));return;}installFeedbackChannel(value.objects.feedback);resolve();});});}
export function feedbackTime(value:number){return new Date(value*1000).toLocaleString(locale.value,{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'});}
export function feedbackClientLabel(client?:FeedbackClientInfo|null){
  if(client?.source==='web')return tr('platform.web_version');
  if(client?.source!=='desktop')return tr('platform.source_not_recorded');
  const kind=({installer:tr('platform.installed_version'),portable:tr('platform.portable_version'),source:tr('platform.running_from_source')} as const)[client.packageKind!];
  return [client.version??tr('platform.version_not_recorded'),kind??tr('platform.installation_type_not_recorded')].join(' · ');
}

export async function prepareFeedbackImages(files:File[]):Promise<FeedbackImage[]>{
  if(files.length>4)throw uiError(uiMessage('platform.you_can_attach_up_to_4_images_per_message'));
  return Promise.all(files.map(async file=>{if(!['image/png','image/jpeg','image/webp'].includes(file.type)||file.size>2_000_000)throw uiError(uiMessage('platform.choose_png_jpeg_or_webp_images_no_larger_than'));const data=await new Promise<string>((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result));reader.onerror=()=>reject(uiError(uiMessage('platform.the_image_could_not_be_read')));reader.readAsDataURL(file);});return {data:data.split(',')[1]!,mime:file.type,preview:data};}));
}
