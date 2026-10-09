import {CHAPTER_SCHEMA,MANUAL_SCHEMA,type ManualChapter,type ManualChapterDescriptor,type ManualNode,type ManualProject,type ManualSearchEntry,type ManualSearchHit,type ManualSection,type ManualAsset} from './types.ts';

const record=(v:unknown):v is Record<string,unknown>=>!!v&&typeof v==='object'&&!Array.isArray(v);
export const validStableId=(v:unknown):v is string=>typeof v==='string'&&/^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$/.test(v);
export function safeRelativePath(value:unknown):value is string {
  if(typeof value!=='string'||!value||value.length>2048||/[\\\x00-\x1f?#:]/.test(value)||value.startsWith('/'))return false;
  let decoded=value;
  try{for(let n=0;n<3&&decoded.includes('%');n++){const next=decodeURIComponent(decoded);if(next===decoded)break;decoded=next;}}catch{return false;}
  return !/[\\\x00-\x1f?#:%]/.test(decoded)&&decoded.split('/').every(part=>!!part&&part!=='.'&&part!=='..');
}
export function safeLink(value:unknown):string|null {
  if(typeof value!=='string'||value.length>2048||/[\x00-\x20\\]/.test(value))return null;
  if(/^https?:\/\//i.test(value)){try{const url=new URL(value);return !url.username&&!url.password?url.href:null;}catch{return null;}}
  if(/^mailto:[^@\s]+@[^@\s]+$/i.test(value))return value;
  if(value.startsWith('#')&&validStableId(value.slice(1)))return value;
  return safeRelativePath(value)?value:null;
}
export function assetUrl(base:string,path:string):string|null {
  if(!safeRelativePath(path)||/[\x00-\x1f\\?#]/.test(base))return null;
  if(/^[a-z][a-z0-9+.-]*:/i.test(base)&&!/^https?:\/\//i.test(base))return null;
  if(base.startsWith('//'))return null;
  if(/^https?:\/\//i.test(base)){try{const url=new URL(base);if(url.username||url.password)return null;}catch{return null;}}
  else if(!/^(?:\/|\.\/|\.\.\/)?[^:]*$/.test(base))return null;
  return base.replace(/\/?$/,'/')+path.split('/').map(part=>encodeURIComponent(decodeURIComponent(part))).join('/');
}
/** Host-provided preview resolvers may use a session URL, never an active scheme. */
export function safeMediaSource(value:unknown):string|null {
  if(typeof value!=='string'||!value||/[\x00-\x20\\]/.test(value)||value.startsWith('//'))return null;
  if(/^[a-z][a-z0-9+.-]*:/i.test(value)&&!/^https?:\/\//i.test(value)&&!/^blob:https?:\/\//i.test(value))return null;
  if(/^https?:\/\//i.test(value)){try{const url=new URL(value);if(url.username||url.password)return null;}catch{return null;}}
  return value;
}
export function manualAnchor(chapterId:string,targetId?:string):string {
  return 'manual-'+encodeURIComponent(chapterId)+(targetId?'--'+encodeURIComponent(targetId):'');
}
function requireString(value:unknown,label:string):asserts value is string {if(typeof value!=='string'||!value.trim())throw Error(label+'缺少有效文字。');}
function requireId(value:unknown,label:string):asserts value is string {if(!validStableId(value))throw Error(label+'的稳定标识无效。');}
function uniqueId(seen:Set<string>,id:string,label:string){if(seen.has(id))throw Error(label+'存在重复标识：'+id);seen.add(id);}

export function parseManualProject(value:unknown):ManualProject {
  if(!record(value)||value.schemaVersion!==MANUAL_SCHEMA)throw Error('说明书格式版本不兼容。');
  requireId(value.id,'说明书');requireString(value.title,'说明书标题');
  if(!Array.isArray(value.chapters)||!Array.isArray(value.assets))throw Error('说明书缺少章节或素材清单。');
  const chapterIds=new Set<string>(),assetIds=new Set<string>();
  for(const chapter of value.chapters){
    if(!record(chapter))throw Error('章节清单格式无效。');requireId(chapter.id,'章节');uniqueId(chapterIds,chapter.id,'章节');requireString(chapter.title,'章节标题');
    if(!safeRelativePath(chapter.path))throw Error('章节路径必须位于说明书资源目录。');
    if(chapter.sections!==undefined){if(!Array.isArray(chapter.sections))throw Error('小节目录格式无效。');const ids=new Set<string>();for(const s of chapter.sections){if(!record(s))throw Error('小节目录格式无效。');requireId(s.id,'小节');uniqueId(ids,s.id,'小节');requireString(s.title,'小节标题');if(!Number.isInteger(s.level)||Number(s.level)<1||Number(s.level)>6)throw Error('小节标题层级无效。');}}
  }
  for(const asset of value.assets){
    if(!record(asset))throw Error('素材清单格式无效。');requireId(asset.id,'素材');uniqueId(assetIds,asset.id,'素材');
    if(!safeRelativePath(asset.path))throw Error('素材路径必须位于说明书资源目录。');
    if(!['image','audio','video','example'].includes(String(asset.kind))||!['public','software-only'].includes(String(asset.distribution)))throw Error('素材类型或分发范围无效。');
    for(const field of ['width','height','duration','sampleRate','channels'])if(asset[field]!==undefined&&(typeof asset[field]!=='number'||!Number.isFinite(asset[field])||Number(asset[field])<(field==='duration'?0:1)))throw Error('素材元数据无效。');
  }
  if(value.searchIndex!==undefined){if(!Array.isArray(value.searchIndex))throw Error('搜索索引格式无效。');for(const item of value.searchIndex){if(!record(item)||typeof item.text!=='string'||!chapterIds.has(String(item.chapterId))||(item.targetId!==undefined&&!validStableId(item.targetId)))throw Error('搜索索引引用无效。');}}
  if(value.references!==undefined){if(!Array.isArray(value.references))throw Error('参考文献格式无效。');const ids=new Set<string>();for(const item of value.references){if(!record(item))throw Error('参考文献格式无效。');requireId(item.id,'参考文献');uniqueId(ids,item.id,'参考文献');requireString(item.label,'参考文献');}}
  return value as unknown as ManualProject;
}
export function parseManualChapter(value:unknown,expectedId?:string):ManualChapter {
  if(!record(value)||value.schemaVersion!==CHAPTER_SCHEMA)throw Error('章节格式版本不兼容。');
  requireId(value.id,'章节');requireString(value.title,'章节标题');if(expectedId&&value.id!==expectedId)throw Error('章节内容与目录标识不一致。');
  if(!record(value.body)||value.body.type!=='doc')throw Error('章节缺少正文文档。');
  const ids=new Set<string>();let count=0;
  const visit=(node:unknown,depth:number)=>{
    if(depth>64||++count>30000)throw Error('章节内容超过安全结构上限。');
    if(!record(node)||typeof node.type!=='string'||!node.type)throw Error('正文节点格式无效。');
    if(node.attrs!==undefined&&!record(node.attrs))throw Error('正文属性格式无效。');
    if(record(node.attrs)&&node.attrs.id!==undefined){requireId(node.attrs.id,'正文节点');uniqueId(ids,node.attrs.id,'正文节点');}
    if(node.text!==undefined&&typeof node.text!=='string')throw Error('正文文字格式无效。');
    if(node.marks!==undefined){if(!Array.isArray(node.marks)||node.marks.some(m=>!record(m)||typeof m.type!=='string'||(m.attrs!==undefined&&!record(m.attrs))))throw Error('文字格式标记无效。');}
    if(node.content!==undefined){if(!Array.isArray(node.content))throw Error('正文子节点格式无效。');for(const child of node.content)visit(child,depth+1);}
  };
  visit(value.body,0);return value as unknown as ManualChapter;
}
export function nodeText(node:ManualNode):string {
  const attrs=node.attrs??{};
  const extra=['caption','note','label','latex','alt','title'].flatMap(key=>typeof attrs[key]==='string'?[attrs[key] as string]:[]);
  return [node.text??'',...extra,...(node.content??[]).map(nodeText)].filter(Boolean).join(['paragraph','text','heading','codeBlock'].includes(node.type)?'':' ');
}
export function chapterSections(chapter:ManualChapter):ManualSection[] {
  const sections:ManualSection[]=[];
  const visit=(node:ManualNode)=>{if(node.type==='heading'&&validStableId(node.attrs?.id))sections.push({id:node.attrs.id,title:nodeText(node),level:Math.max(1,Math.min(6,Number(node.attrs.level)||2))});for(const child of node.content??[])visit(child);};
  visit(chapter.body);return sections;
}
/** Display numbers follow the current book order; stable IDs remain the navigation identity. */
export function sectionNumbers(sections:ManualSection[],chapterNumber:number):Record<string,string> {
  const numbers:Record<string,string>={};
  const counters=[0,0,0,0,0,0,0];
  for(const section of sections){
    const level=Math.max(2,Math.min(6,Math.trunc(section.level)||2));
    counters[level]++;
    for(let next=level+1;next<=6;next++)counters[next]=0;
    for(let parent=2;parent<level;parent++)if(!counters[parent])counters[parent]=1;
    numbers[section.id]=[chapterNumber,...counters.slice(2,level+1)].join('.');
  }
  return numbers;
}
export function chapterDisplayNumbers(chapter:ManualChapter,chapterNumber:number):Record<string,string> {
  const numbers=sectionNumbers(chapterSections(chapter),chapterNumber);
  const counts={image:0,audio:0,video:0,table:0};
  const visit=(node:ManualNode)=>{
    const type=node.type as keyof typeof counts;
    if(type in counts){counts[type]++;if(validStableId(node.attrs?.id))numbers[node.attrs.id]=chapterNumber+'-'+counts[type];}
    for(const child of node.content??[])visit(child);
  };
  visit(chapter.body);
  return numbers;
}
export function unnumberedCaption(value:string):string {
  return value.replace(/^\s*(?:图|表|例音|音频|视频)\s*\d+(?:\s*[-.－—]\s*\d+)*\s*[:：.]\s*/u,'').trim();
}
export function chapterSearchEntries(chapter:ManualChapter):ManualSearchEntry[] {
  let targetId:string|undefined;
  return (chapter.body.content??[]).map(node=>{if(validStableId(node.attrs?.id))targetId=node.attrs.id;return {chapterId:chapter.id,targetId,text:nodeText(node),title:chapter.title};}).filter(entry=>entry.text.trim());
}
export function searchManual(entries:ManualSearchEntry[],query:string,limit=80):ManualSearchHit[] {
  const terms=query.trim().normalize('NFKC').toLocaleLowerCase().split(/\s+/).filter(Boolean);
  if(!terms.length)return [];
  const hits:ManualSearchHit[]=[];
  for(const entry of entries){const normalized=entry.text.normalize('NFKC').toLocaleLowerCase();if(!terms.every(term=>normalized.includes(term)))continue;
    const index=normalized.indexOf(terms[0]),start=Math.max(0,index-35),end=Math.min(entry.text.length,index+115);
    hits.push({...entry,excerpt:(start?'…':'')+entry.text.slice(start,end)+(end<entry.text.length?'…':'')});if(hits.length>=limit)break;
  }return hits;
}
/** Software-only overlays add assets; a different asset cannot replace a public ID. */
export function mergeManualAssets(publicAssets:ManualAsset[],localAssets:ManualAsset[]):ManualAsset[] {
  const result=[...publicAssets],byId=new Map(publicAssets.map(asset=>[asset.id,asset]));
  for(const asset of localAssets){if(asset.distribution!=='software-only'||!validStableId(asset.id)||!safeRelativePath(asset.path))throw Error('软件素材清单无效。');const prior=byId.get(asset.id);if(prior){if(JSON.stringify(prior)!==JSON.stringify(asset))throw Error('软件素材标识与公共清单冲突：'+asset.id);continue;}byId.set(asset.id,asset);result.push(asset);}return result;
}
export async function fetchChapter(base:string,descriptor:ManualChapterDescriptor,signal:AbortSignal):Promise<ManualChapter> {
  const url=assetUrl(base,descriptor.path);if(!url)throw Error('章节路径无效。');
  const response=await fetch(url,{signal,credentials:'same-origin'});if(!response.ok)throw Error('章节加载失败，请检查本地说明书资源。');
  return parseManualChapter(await response.json(),descriptor.id);
}
