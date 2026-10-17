// M18: explicit native capability; paper bytes are never application assets.
export interface PaperAsset{path:string;size:number;sha256:string}
export interface Paper{id:string;title:string;titleZh:string;authors:string;version:string;sourceUrl:string;publishedAt:string;submittedAt:string;updatedAt?:string;acquired?:boolean;analysisDownloaded?:boolean;dismissed?:boolean;license:{id:string;url:string};translationNote:string;guide:string;abstract?:string;abstractZh?:string;analysis?:string;analysisPdf?:PaperAsset;original:PaperAsset;translation:PaperAsset;downloaded:boolean;local?:boolean;notice?:boolean}
export interface PaperStatus{firstLaunch:string;warning:string;papers:Paper[];notice?:Paper|null;autoReceive?:boolean;checkIntervalHours?:number;folders?:string[];collections?:Record<string,string>}
export interface LocalDocument{id:string;title:string;titleZh:string;relative:string;folder:string;size:number;publishedAt:string}
export interface LocalLibrary{root:string;folders:string[];documents:LocalDocument[];selected?:string;relocated?:Record<string,string>;imported?:string[];failures?:string[];cancelled?:boolean}
export type PaperLanguage='original'|'translation'|'analysis';
export interface PaperLine{text:string;x:number;y:number;w:number;h:number}
export interface PaperPage{pages:number;page:number;image:string;imageWidth?:number;imageHeight?:number;text:string;ratio:number;ratios:number[];lines:PaperLine[]}
export interface PaperAnnotation{id:string;page:number;kind:'highlight'|'note';color:'yellow'|'green'|'pink';text:string;quote:string;rects:number[][]}
export interface PaperAnnotations{revision:number;items:PaperAnnotation[]}
export interface PaperProgress{received:number;total:number}
interface Signal{connect(fn:(id:string,payload:string)=>void):void;disconnect?(fn:(id:string,payload:string)=>void):void}
export interface PapersChannel{ready:Signal;progress:Signal;statusChanged?:{connect(fn:(payload:string)=>void):void;disconnect?(fn:(payload:string)=>void):void};request(id:string,payload:string):void;cancel(id:string):void}
let channel:PapersChannel|undefined;let sequence=0;
const pending=new Map<string,{resolve:(value:any)=>void;reject:(error:Error)=>void;progress?:(value:PaperProgress)=>void;cleanup:()=>void}>();
const statusListeners=new Set<(value:PaperStatus)=>void>();
function statusChanged(payload:string){try{const value=JSON.parse(payload);for(const listener of statusListeners)listener(value);}catch{/* Ignore invalid unsolicited status. */}}
export function onPaperStatus(listener:(value:PaperStatus)=>void){statusListeners.add(listener);return ()=>statusListeners.delete(listener);}
function ready(id:string,payload:string){const job=pending.get(id);if(!job)return;pending.delete(id);job.cleanup();try{const r=JSON.parse(payload);if(r.ok)job.resolve(r.value);else job.reject(Error(r.error));}catch{job.reject(Error('论文响应无法解析。'));}}
function progress(id:string,payload:string){try{const p=JSON.parse(payload);if(Number.isFinite(p.received)&&Number.isFinite(p.total)&&p.received>=0&&p.total>=p.received)pending.get(id)?.progress?.(p);}catch{/* Ignore malformed progress; completion remains explicit. */}}
export function installPapersChannel(next:PapersChannel|undefined){
  channel?.ready.disconnect?.(ready);channel?.progress.disconnect?.(progress);
  channel?.statusChanged?.disconnect?.(statusChanged);
  for(const [id,p] of pending){channel?.cancel(id);p.cleanup();p.reject(Error('论文服务已断开。'));}pending.clear();
  channel=next;channel?.ready.connect(ready);channel?.progress.connect(progress);
  channel?.statusChanged?.connect(statusChanged);
}
export function papersAvailable(){return !!channel;}
function invoke<T>(operation:string,args:Record<string,unknown>={},signal?:AbortSignal,onProgress?:(p:PaperProgress)=>void):Promise<T>{
  const current=channel;if(!current)return Promise.reject(Error('请在桌面版打开论文精读，获取服务器论文并离线阅读。'));
  if(signal?.aborted)return Promise.reject(Error('操作已取消。'));
  const id=`paper_${++sequence}`;
  return new Promise((resolve,reject)=>{
    const stop=(message:string)=>{if(!pending.delete(id))return;cleanup();current.cancel(id);reject(Error(message));};
    const abort=()=>stop('操作已取消，已完成的下载保留。');
    const timer=setTimeout(()=>stop('论文操作超时，请重试。'),['download','downloadSelected','export','importLocal','chooseLibrary'].includes(operation)?30*60*1000:60000);
    const cleanup=()=>{clearTimeout(timer);signal?.removeEventListener('abort',abort);};
    pending.set(id,{resolve,reject,progress:onProgress,cleanup});signal?.addEventListener('abort',abort,{once:true});
    try{current.request(id,JSON.stringify({operation,args}));}catch{stop('无法提交论文请求。');}
  });
}
export function createPapers(invoke:<T>(operation:string,args?:Record<string,unknown>,signal?:AbortSignal,onProgress?:(p:PaperProgress)=>void)=>Promise<T>){return {
  status:()=>invoke<PaperStatus>('status'),
  refresh:(signal?:AbortSignal)=>invoke<PaperStatus>('refresh',{},signal),
  download:(since:string,signal?:AbortSignal,onProgress?:(p:PaperProgress)=>void)=>invoke<PaperStatus>('download',{since},signal,onProgress),
  downloadSelected:(ids:string[],signal?:AbortSignal,onProgress?:(p:PaperProgress)=>void)=>invoke<PaperStatus>('downloadSelected',{ids},signal,onProgress),
  setAutoReceive:(enabled:boolean)=>invoke<PaperStatus>('autoReceive',{enabled}),
  setInterval:(hours:number)=>invoke<PaperStatus>('interval',{hours}),
  createRecommendationFolder:(name:string)=>invoke<PaperStatus>('createRecommendationFolder',{name}),
  deleteRecommendationFolder:(folder:string)=>invoke<PaperStatus>('deleteRecommendationFolder',{folder}),
  collectRecommendation:(id:string,folder:string)=>invoke<PaperStatus>('collectRecommendation',{id,folder}),
  removeDownload:(id:string)=>invoke<PaperStatus>('removeDownload',{id}),
  library:()=>invoke<LocalLibrary>('library'),
  chooseLibrary:()=>invoke<LocalLibrary>('chooseLibrary'),
  importLocal:(folder:string)=>invoke<LocalLibrary>('importLocal',{folder}),
  createFolder:(name:string)=>invoke<LocalLibrary>('createFolder',{name}),
  deleteFolder:(folder:string)=>invoke<LocalLibrary>('deleteFolder',{folder}),
  moveLocal:(id:string,folder:string)=>invoke<LocalLibrary>('moveLocal',{id,folder}),
  removeLocal:(id:string)=>invoke<LocalLibrary>('removeLocal',{id}),
  render:(id:string,language:PaperLanguage,page:number,width:number,signal?:AbortSignal)=>invoke<PaperPage>('render',{id,language,page,width},signal),
  annotations:(id:string,language:PaperLanguage,signal?:AbortSignal)=>invoke<PaperAnnotations>('annotations',{id,language},signal),
  saveAnnotation:(id:string,language:PaperLanguage,revision:number,item:PaperAnnotation,remove=false)=>invoke<PaperAnnotations>('saveAnnotation',{id,language,revision,item,remove}),
  export:(id:string,language:PaperLanguage,annotated:boolean)=>invoke<{cancelled:boolean;name?:string}>('export',{id,language,annotated}),
};}
export type PapersClient=ReturnType<typeof createPapers>&{web?:boolean;onStatus?:(listener:(value:PaperStatus)=>void)=>()=>void;maintain?:()=>()=>void;trash?:()=>Promise<{id:string;title:string}[]>;restoreLocal?:(id:string)=>Promise<LocalLibrary>};
export const papers=createPapers(invoke);
export function eligible(papers:Paper[],since:string,today:string){return papers.filter(p=>p.publishedAt<=today&&(!p.updatedAt||p.updatedAt<=today)&&(p.publishedAt>=since||(p.updatedAt??'')>=since||p.acquired));}
export function missing(papers:Paper[],since:string,today:string){return eligible(papers,since,today).filter(p=>!p.downloaded&&!p.dismissed);}
export function bytesLabel(n:number){return n<1_000_000?`${(n/1000).toFixed(0)} KB`:`${(n/1_000_000).toFixed(1)} MB`;}
