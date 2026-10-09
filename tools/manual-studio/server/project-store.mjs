import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import { gzipSync, gunzipSync } from 'node:zlib';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

export const PROJECT_VERSION = 'ptb-manual/1';
export const CHAPTER_VERSION = 'ptb-manual-chapter/1';
const sharedSchema=JSON.parse(readFileSync(path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../../../frontend/src/manual/schema-chapter.json'),'utf8'));
export const NODE_TYPES = new Set(sharedSchema['x-supportedNodes']);
export const MARK_TYPES = new Set(sharedSchema['x-supportedMarks']);
export const sha = data => createHash('sha256').update(data).digest('hex');
const json = value => JSON.stringify(value, null, 2) + '\n';
export class StoreError extends Error { constructor(message, status = 400, extra = {}) { super(message); this.status = status; this.extra = extra; } }

export function relativePath(value, prefix) {
  if (typeof value !== 'string' || !value || /[\\:\x00-\x1f?#%]/.test(value) || value.startsWith('/') || value.split('/').some(s => !s || s === '.' || s === '..') || (prefix && !value.startsWith(prefix + '/'))) throw new StoreError(`非法工程相对路径：${String(value).slice(0,100)}`);
  return value;
}
export async function safePath(root, relative) {
  relativePath(relative);
  let current = root;
  for (const segment of relative.split('/')) {
    current = path.join(current, segment);
    try { if ((await fs.lstat(current)).isSymbolicLink()) throw new StoreError('工程内的符号链接不可读写'); } catch (e) { if (e.code !== 'ENOENT') throw e; }
  }
  return current;
}
const stableId = (id, where) => { if (typeof id !== 'string' || !/^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,159}$/.test(id)) throw new StoreError(`${where} 的稳定 ID 无效`); };
const safeLink = href => typeof href === 'string' && (/^https?:\/\//i.test(href) || /^mailto:[^\s]+$/i.test(href) || /^#[A-Za-z0-9_.:-]+$/.test(href) || /^manual:[A-Za-z0-9_.:/#-]+$/.test(href));
export function validateProject(project) {
  if (!project || project.schemaVersion !== PROJECT_VERSION || !Array.isArray(project.chapters) || !Array.isArray(project.assets) || !Array.isArray(project.references ?? [])) throw new StoreError('工程格式须为 ptb-manual/1');
  stableId(project.id, '工程');
  if (typeof project.title !== 'string') throw new StoreError('工程标题无效');
  const ids = new Set(), paths = new Set();
  for (const c of project.chapters) { stableId(c.id,'章节'); relativePath(c.path,'chapters'); if (!c.path.endsWith('.json') || ids.has(c.id) || paths.has(c.path) || typeof c.title !== 'string') throw new StoreError('章节 ID、路径重复或标题无效'); ids.add(c.id); paths.add(c.path); }
  ids.clear();
  for (const a of project.assets) { stableId(a.id,'素材'); relativePath(a.path,'assets'); if(ids.has(a.id) || !['image','audio','video','example'].includes(a.kind)) throw new StoreError('素材 ID 重复或种类无效'); ids.add(a.id); if(!['public','software-only'].includes(a.distribution ?? 'public')) throw new StoreError('素材分发范围无效'); if(a.distribution === 'software-only' && a.git !== false) throw new StoreError('software-only 素材必须 git=false'); if(/(?:[A-Za-z]:[\\/]|\\\\[^\\]|file:\/\/)/.test(JSON.stringify(a.source ?? ''))) throw new StoreError('公开工程不可保存素材的私有绝对来源路径'); }
  return project;
}
export function validateChapter(chapter, expectedId) {
  if (!chapter || chapter.schemaVersion !== CHAPTER_VERSION || chapter.id !== expectedId || typeof chapter.title !== 'string' || chapter.body?.type !== 'doc') throw new StoreError('章节格式或稳定 ID 无效');
  let count = 0; const unsupported = [], anchors = new Set();
  const visit = (node, depth = 0) => {
    if (++count > 100000 || depth > 45 || !node || typeof node.type !== 'string') throw new StoreError('章节节点过多、过深或格式错误');
    if(!NODE_TYPES.has(node.type)) unsupported.push(node.type);
    if(node.attrs?.id) { stableId(node.attrs.id,'正文锚点'); if(anchors.has(node.attrs.id)) throw new StoreError(`正文锚点重复：${node.attrs.id}`); anchors.add(node.attrs.id); }
    if(['image','audio','video'].includes(node.type)) stableId(node.attrs?.assetId, '媒体引用');
    if(node.type === 'text' && typeof node.text !== 'string') throw new StoreError('文字节点无效');
    if(Object.keys(node.attrs??{}).some(k=>/^on/i.test(k)||['src','innerHTML','script'].includes(k))) throw new StoreError('正文不允许脚本或直接媒体地址，请用 assetId');
    for(const mark of node.marks ?? []) { if(!MARK_TYPES.has(mark.type)) unsupported.push(`mark:${mark.type}`); if(mark.type === 'link' && !safeLink(mark.attrs?.href)) throw new StoreError('链接仅允许 HTTP(S)、mailto 或工程内定位'); }
    for(const child of node.content ?? []) visit(child,depth+1);
  };
  visit(chapter.body); return [...new Set(unsupported)];
}
async function atomicWrite(file, data) {
  await fs.mkdir(path.dirname(file),{recursive:true});
  const temp = file + '.tmp-' + randomUUID();
  const handle = await fs.open(temp,'wx');
  try {
    try { await handle.writeFile(data); await handle.sync(); } finally { await handle.close(); }
    await fs.rename(temp,file);
  } finally {
    try { await fs.unlink(temp); } catch(e) { if(e.code!=='ENOENT') console.warn('Manual temporary write retained:',temp,e.message); }
  }
}
const stamp = () => new Date().toISOString().replace(/[:.]/g,'-') + '-' + randomUUID().slice(0,8);
const processAlive = pid => { try { process.kill(pid,0); return true; } catch(e) { return e.code === 'EPERM'; } };
const snapshotName = /^\d{4}-\d{2}-\d{2}T[\d-]+Z-[a-f0-9]{8}$/;
async function removeSnapshot(root, relative) {
  if (!relative.startsWith('.studio/')) throw new StoreError('仅可清理作者工具的自动保存副本');
  const target = await safePath(root, relative);
  if (!target.startsWith(root + path.sep)) throw new StoreError('自动保存路径越界');
  const inspect = async file => {
    const stat = await fs.lstat(file);
    if (stat.isSymbolicLink()) throw new StoreError('自动保存副本含符号链接，须单独检查');
    if (stat.isDirectory()) for (const name of await fs.readdir(file)) await inspect(path.join(file,name));
  };
  await inspect(target);
  await fs.rm(target, { recursive: true });
}

export class ProjectStore {
  constructor() { this.root = null; this.lockId = randomUUID(); this.queue = Promise.resolve(); }
  serial(action) { const result = this.queue.then(action); this.queue = result.catch(() => {}); return result; }
  async release() {
    if(!this.root) return;
    const file = path.join(this.root,'.studio','lock.json');
    try { const lock=JSON.parse(await fs.readFile(file,'utf8')); if(lock.id === this.lockId) await atomicWrite(file,json({...lock,released:true})); } catch { /* A failed release never deletes another owner's lock. */ }
    this.root = null;
  }
  async open(directory) {
    if(typeof directory !== 'string' || !path.isAbsolute(directory)) throw new StoreError('请输入工程的绝对目录');
    const target = await fs.realpath(directory);
    if(target === this.root) return this.load();
    validateProject(JSON.parse(await fs.readFile(await safePath(target,'project.json'),'utf8')));
    const lockFile = await safePath(target,'.studio/lock.json'); await fs.mkdir(path.dirname(lockFile),{recursive:true});
    try {
      const old = JSON.parse(await fs.readFile(lockFile,'utf8'));
      if(!old.released && processAlive(old.pid) && old.id !== this.lockId) throw new StoreError('此工程已由另一个作者工具实例打开，请先退出该实例。',423);
    } catch(e) { if(e.code !== 'ENOENT') throw e; }
    // The mkdir reservation closes the check/write race. It is retained and released by rename, never recursively removed.
    const reservation = path.join(target,'.studio','reservation');
    try { await fs.mkdir(reservation); } catch(e) {
      if(e.code !== 'EEXIST') throw e;
      let stale=false;try{const owner=JSON.parse(await fs.readFile(path.join(reservation,'owner.json'),'utf8'));stale=!processAlive(owner.pid);}catch{stale=Date.now()-(await fs.stat(reservation)).mtimeMs>30000;}
      if(!stale)throw new StoreError('工程正在被另一个实例打开，请稍后重试。',423);
      await fs.rename(reservation,reservation+'-interrupted-'+stamp());await fs.mkdir(reservation);
    }
    await atomicWrite(path.join(reservation,'owner.json'),json({pid:process.pid,id:this.lockId}));
    try {
      try {const old=JSON.parse(await fs.readFile(lockFile,'utf8'));if(!old.released&&processAlive(old.pid)&&old.id!==this.lockId)throw new StoreError('此工程已由另一个作者工具实例打开。',423);}catch(e){if(e.code!=='ENOENT')throw e;}
      await atomicWrite(lockFile,json({id:this.lockId,pid:process.pid,time:new Date().toISOString(),released:false}));
    } finally { await fs.rename(reservation, reservation+'-released-'+stamp()); }
    await this.release(); this.root=target; return this.load();
  }
  async create(directory, title = 'PhoneticToolbox 使用说明书') {
    if(typeof directory !== 'string' || !path.isAbsolute(directory)) throw new StoreError('新工程目录须为绝对路径');
    try { const entries = await fs.readdir(directory); if(entries.length) throw new StoreError('新工程目录须为空，已有工程不会被覆盖'); } catch(e) { if(e.code !== 'ENOENT') throw e; }
    await fs.mkdir(directory,{recursive:true});
    const root = await fs.realpath(directory); await fs.mkdir(path.join(root,'chapters')); await fs.mkdir(path.join(root,'assets'));
    await atomicWrite(path.join(root,'project.json'),json({schemaVersion:PROJECT_VERSION,id:'manual-'+randomUUID(),title,language:'zh-CN',chapters:[],assets:[],references:[]}));
    return this.open(root);
  }
  requireOpen() { if(!this.root) throw new StoreError('请先打开说明书工程'); return this.root; }
  async load() {
    const root = this.requireOpen(), projectText = await fs.readFile(await safePath(root,'project.json'),'utf8'), project=validateProject(JSON.parse(projectText));
    const chapters = [], unsupported = []; let signature=projectText;
    for(const descriptor of project.chapters) { const source=await fs.readFile(await safePath(root,descriptor.path),'utf8'), chapter=JSON.parse(source); unsupported.push(...validateChapter(chapter,descriptor.id).map(type=>({chapterId:chapter.id,type}))); chapters.push(chapter); signature+=descriptor.path+'\0'+source; }
    return {path:root,project,chapters,revision:sha(signature),unsupported};
  }
  async recovery(payload) { return this.serial(() => this.writeRecovery(payload)); }
  async writeRecovery(payload) {
    const root=this.requireOpen();
    validateProject(payload.project); for(const chapter of payload.chapters ?? []) validateChapter(chapter,chapter.id);
    const id='draft-'+stamp(); await atomicWrite(await safePath(root,`.studio/recovery/${id}.json`),json({id,time:new Date().toISOString(),baseRevision:payload.baseRevision,project:payload.project,chapters:payload.chapters}));
    // Publish the new durable draft before retiring the previous one.
    await this.pruneSnapshots({ recoveryId:id }); return id;
  }
  async pruneSnapshots({recoveryId, committed=false}={}) {
    const root=this.requireOpen(), deleted=[], retained=[];
    const entries=async folder=>{try{return await fs.readdir(await safePath(root,folder));}catch(e){if(e.code==='ENOENT')return [];throw e;}};
    const retire=async relative=>{try{await removeSnapshot(root,relative);deleted.push(relative);}catch(e){retained.push({path:relative,reason:e.message});}};
    if(recoveryId) for(const name of await entries('.studio/recovery')) {
      if(name!==recoveryId+'.json' && name.startsWith('draft-') && name.endsWith('.json') && snapshotName.test(name.slice(6,-5))) await retire('.studio/recovery/'+name);
    }
    if(committed) {
      let pending=false;
      for(const name of await entries('.studio/transactions')) {
        if(!name.endsWith('.json') || !snapshotName.test(name.slice(0,-5))) continue;
        const relative='.studio/transactions/'+name;
        try {
          const value=JSON.parse(await fs.readFile(await safePath(root,relative),'utf8'));
          if(value.state==='committed') await retire(relative);
          else {pending=true;retained.push({path:relative,reason:'Unfinished save; recovery evidence retained'});}
        } catch(e) {pending=true;retained.push({path:relative,reason:e.message});}
      }
      // Keep rollback material for interrupted writes. Normal completed saves
      // retain only the current project and one recovery draft.
      if(!pending) for(const name of await entries('.studio/history')) if(snapshotName.test(name)) await retire('.studio/history/'+name);
    }
    const report={schema:'ptb-manual-retention/1',recoveryLimit:1,deleted,retained,time:new Date().toISOString()};
    await atomicWrite(await safePath(root,'.studio/retention-report.json'),json(report));
    if(retained.length) console.warn('Manual autosave cleanup retained entries; see .studio/retention-report.json');
    return report;
  }
  async save(payload) { return this.serial(async()=>{
    const root=this.requireOpen(); validateProject(payload.project);
    for(const chapter of payload.chapters ?? []) validateChapter(chapter,chapter.id);
    const recoveryId=await this.writeRecovery(payload), current=await this.load();
    if(payload.baseRevision !== current.revision) throw new StoreError('磁盘内容已经变化。已保留当前恢复稿，请重新读取并比较后继续。',409,{recoveryId,currentRevision:current.revision});
    const incoming = new Map((payload.chapters ?? []).map(c=>[c.id,c]));
    for(const descriptor of payload.project.chapters) {
      const chapter=incoming.get(descriptor.id) ?? current.chapters.find(c=>c.id===descriptor.id);
      if(!chapter) throw new StoreError(`缺少章节正文：${descriptor.id}`);
      validateChapter(chapter,descriptor.id);
      if(chapter.title !== descriptor.title) throw new StoreError(`章节标题与目录不一致：${descriptor.id}`);
    }
    const history='.studio/history/'+stamp();
    await atomicWrite(await safePath(root,history+'/project.json'),json(current.project));
    for(const d of current.project.chapters) { const old=current.chapters.find(c=>c.id===d.id); await atomicWrite(await safePath(root,history+'/'+d.path),json(old)); }
    const transaction={state:'prepared',project:payload.project,chapters:[...incoming.values()],previousRevision:current.revision,recoveryId};
    const journal=await safePath(root,'.studio/transactions/'+stamp()+'.json'); await atomicWrite(journal,json(transaction));
    for(const descriptor of payload.project.chapters) if(incoming.has(descriptor.id)) await atomicWrite(await safePath(root,descriptor.path),json(incoming.get(descriptor.id)));
    await atomicWrite(await safePath(root,'project.json'),json(payload.project));
    await atomicWrite(journal,json({...transaction,state:'committed'}));
    const saved=await this.load();
    await this.pruneSnapshots({committed:true});
    return saved;
  }); }
  async recoveries() {
    const root=this.requireOpen(), folder=await safePath(root,'.studio/recovery'); let names=[];
    try { names=await fs.readdir(folder); } catch(e) { if(e.code!=='ENOENT') throw e; }
    const values=[];
    for(const name of names.filter(n=>n.endsWith('.json')).sort().reverse().slice(0,100)) { const data=JSON.parse(await fs.readFile(await safePath(root,'.studio/recovery/'+name),'utf8')); values.push({id:data.id,time:data.time,chapterCount:data.chapters?.length ?? 0}); }
    return values;
  }
  async readRecovery(id) { if(!/^draft-[a-zA-Z0-9-]+$/.test(id)) throw new StoreError('恢复稿 ID 无效'); return JSON.parse(await fs.readFile(await safePath(this.requireOpen(),`.studio/recovery/${id}.json`),'utf8')); }
  async addAsset({name,data,kind,distribution='software-only',sourceType='未分类',source='',baseRevision}) { return this.serial(async()=>{
    const root=this.requireOpen(), snapshot=await this.load();
    if(baseRevision!==undefined && baseRevision!==snapshot.revision)throw new StoreError('磁盘工程已变化，素材尚未导入。请重新读取并比较后再导入。',409);
    if(typeof data !== 'string' || data.length > 128*1024*1024 || !/^[A-Za-z0-9+/]*={0,2}$/.test(data)) throw new StoreError('素材数据无效或超过 96 MiB 单文件限制');
    const extension=path.extname(String(name)).toLowerCase(), allow={image:['.png','.jpg','.jpeg','.webp','.gif'],audio:['.wav','.mp3','.ogg','.flac','.m4a'],video:['.mp4','.webm','.mov'],example:['.txt','.csv','.json','.pdf']};
    if(!allow[kind]?.includes(extension)) throw new StoreError('素材类型或文件后缀不支持');
    const buffer=Buffer.from(data,'base64'), hash=sha(buffer), id='asset-'+hash.slice(0,20), relative=`assets/${id}${extension}`;
    const mime={'.png':'image/png','.jpg':'image/jpeg','.jpeg':'image/jpeg','.webp':'image/webp','.gif':'image/gif','.wav':'audio/wav','.mp3':'audio/mpeg','.ogg':'audio/ogg','.flac':'audio/flac','.m4a':'audio/mp4','.mp4':'video/mp4','.webm':'video/webm','.mov':'video/quicktime'}[extension] ?? 'application/octet-stream';
    const existing=snapshot.project.assets.find(a=>a.id===id); if(existing) return {asset:existing,snapshot};
    const asset={id,path:relative,kind,mime,sha256:hash,title:path.basename(String(name)),sourceType,source,distribution,git:distribution==='public'};
    validateProject({...snapshot.project,assets:[...snapshot.project.assets,asset]});
    const file=await safePath(root,relative); try { const old=await fs.readFile(file); if(sha(old)!==hash) throw new StoreError('素材哈希路径冲突'); } catch(e) { if(e.code!=='ENOENT') throw e; await atomicWrite(file,buffer); }
    await atomicWrite(await safePath(root,'.studio/history/'+stamp()+'/project.json'),json(snapshot.project));snapshot.project.assets.push(asset); await atomicWrite(await safePath(root,'project.json'),json(snapshot.project));
    const saved=await this.load();
    await this.writeRecovery({...saved,baseRevision:saved.revision});
    await this.pruneSnapshots({committed:true});
    return {asset,snapshot:saved};
  }); }
  async exportBundle(mode='software') {
    const snapshot=await this.load(), root=this.requireOpen(), files=[], skipped=[];
    const project=structuredClone(snapshot.project);
    if(mode==='public') { project.assets=project.assets.filter(a=>{const include=a.distribution==='public' && a.git!==false; if(!include) skipped.push(a.id); return include;}); }
    const add=async relative=>{const bytes=await fs.readFile(await safePath(root,relative));files.push({path:relative,sha256:sha(bytes),data:bytes.toString('base64')});};
    const projectBytes=Buffer.from(json(project)); files.push({path:'project.json',sha256:sha(projectBytes),data:projectBytes.toString('base64')});
    for(const chapter of project.chapters) await add(chapter.path);
    for(const asset of project.assets) await add(asset.path);
    if(mode==='software') {
      const walk=async directory=>{for(const entry of await fs.readdir(await safePath(root,directory),{withFileTypes:true})) { const relative=directory+'/'+entry.name; if(entry.isSymbolicLink()) throw new StoreError('导出不允许工程符号链接'); if(entry.isDirectory()) await walk(relative); else if(!files.some(f=>f.path===relative)) await add(relative); }};
      for(const directory of ['chapters','assets']) {try {await walk(directory);}catch(e){if(e.code!=='ENOENT')throw e;}}
    }
    return {bytes:gzipSync(Buffer.from(json({format:'ptb-manual-bundle/1',distribution:mode,files}))),skipped};
  }
  async importBundle(directory, bytes) {
    let bundle; try { bundle=JSON.parse(gunzipSync(bytes,{maxOutputLength:512*1024*1024}).toString('utf8')); } catch { throw new StoreError('工程包损坏或展开大小超过 512 MiB'); }
    if(bundle.format!=='ptb-manual-bundle/1' || !Array.isArray(bundle.files)) throw new StoreError('工程包格式无效');
    const decoded=[], seen=new Set();
    for(const file of bundle.files) { relativePath(file.path); if(file.path!=='project.json' && !file.path.startsWith('chapters/') && !file.path.startsWith('assets/')) throw new StoreError('工程包包含非内容文件'); if(seen.has(file.path)) throw new StoreError('工程包存在重复路径'); seen.add(file.path); const data=Buffer.from(file.data,'base64');if(sha(data)!==file.sha256)throw new StoreError(`工程包哈希不符：${file.path}`);decoded.push({...file,data}); }
    const project=validateProject(JSON.parse(decoded.find(f=>f.path==='project.json')?.data.toString('utf8') ?? 'null'));
    for(const c of project.chapters) {const file=decoded.find(f=>f.path===c.path);if(!file)throw new StoreError('工程包缺章节正文');validateChapter(JSON.parse(file.data.toString('utf8')),c.id);}
    if(typeof directory!=='string' || !path.isAbsolute(directory))throw new StoreError('导入目标须为绝对目录');
    try {if((await fs.readdir(directory)).length)throw new StoreError('导入须使用全新空目录，已有工程不会被覆盖');}catch(e){if(e.code!=='ENOENT')throw e;}
    await fs.mkdir(directory,{recursive:true});const root=await fs.realpath(directory);
    for(const file of decoded)await atomicWrite(await safePath(root,file.path),file.data);
    return this.open(root);
  }
}
