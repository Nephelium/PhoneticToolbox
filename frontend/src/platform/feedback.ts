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
function ready(id:string,payload:string){const job=jobs.get(id);if(!job)return;jobs.delete(id);clearTimeout(job.timer);try{const data=JSON.parse(payload);if(data.ok)job.resolve(data.value);else job.reject(Error(data.error));}catch{job.reject(Error('反馈响应无法解析。'));}}
function notification(payload:string){try{const data=JSON.parse(payload);for(const listener of listeners)listener(data);}catch{/* Malformed notifications are never interpreted as new feedback. */}}
export function installFeedbackChannel(next:FeedbackChannel|undefined){channel?.ready.disconnect?.(ready);channel?.notification?.disconnect?.(notification);for(const job of jobs.values()){clearTimeout(job.timer);job.reject(Error('反馈服务已断开。'));}jobs.clear();channel=next;channel?.ready.connect(ready);channel?.notification?.connect(notification);}
export function onFeedbackNotification(fn:(value:FeedbackEvent)=>void){listeners.add(fn);return()=>listeners.delete(fn);}
export function feedbackCall<T>(operation:string,args:Record<string,unknown>={}):Promise<T>{const current=channel;if(!current)return Promise.reject(Error('请在桌面应用中使用问题反馈。'));const id=`feedback_${++sequence}`;return new Promise((resolve,reject)=>{const timer=setTimeout(()=>{jobs.delete(id);reject(Error('反馈请求超时。请先刷新会话，再重试发送。'));},45000);jobs.set(id,{resolve,reject,timer});try{current.request(id,JSON.stringify({operation,args}));}catch{jobs.delete(id);clearTimeout(timer);reject(Error('反馈请求未提交。'));}});}
export async function initializeFeedbackManager(){const surface=window as unknown as {ptbFeedbackNative?:{request:(id:string,payload:string)=>void};ptbFeedbackAndroid?:{ready:typeof ready;notification:typeof notification};qt?:{webChannelTransport:unknown};QWebChannel?:new(transport:unknown,callback:(value:{objects:{feedback?:FeedbackChannel}})=>void)=>unknown};if(surface.ptbFeedbackNative){surface.ptbFeedbackAndroid={ready,notification};installFeedbackChannel({request:(id,payload)=>surface.ptbFeedbackNative!.request(id,payload),ready:{connect:()=>{}},notification:{connect:()=>{}}});return;}if(!surface.qt||!surface.QWebChannel)throw Error('请通过独立反馈管理工具打开此页面。');await new Promise<void>((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('管理工具连接超时。')),15000);new surface.QWebChannel!(surface.qt!.webChannelTransport,value=>{clearTimeout(timer);if(!value.objects.feedback){reject(Error('反馈管理通道不可用。'));return;}installFeedbackChannel(value.objects.feedback);resolve();});});}
export function feedbackTime(value:number){return new Date(value*1000).toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'});}
export function feedbackClientLabel(client?:FeedbackClientInfo|null){
  if(client?.source==='web')return '网页版';
  if(client?.source!=='desktop')return '来源未记录';
  const kind=({installer:'安装版',portable:'免安装版',source:'源码运行'} as const)[client.packageKind!];
  return [client.version??'版本未记录',kind??'安装形式未记录'].join(' · ');
}

export async function prepareFeedbackImages(files:File[]):Promise<FeedbackImage[]>{
  if(files.length>4)throw Error('每次最多添加 4 张图片。');
  return Promise.all(files.map(async file=>{if(!['image/png','image/jpeg','image/webp'].includes(file.type)||file.size>2_000_000)throw Error('请选择 PNG、JPEG 或 WebP 图片，每张不超过 2 MB。');const data=await new Promise<string>((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result));reader.onerror=()=>reject(Error('图片读取失败。'));reader.readAsDataURL(file);});return {data:data.split(',')[1]!,mime:file.type,preview:data};}));
}
