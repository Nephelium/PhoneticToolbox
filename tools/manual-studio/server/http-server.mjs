import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { randomBytes } from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { ProjectStore, StoreError, safePath } from './project-store.mjs';
import { pageLifetime } from './page-lifetime.mjs';
const execute=promisify(execFile);

const mime = name => ({'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.json':'application/json; charset=utf-8','.png':'image/png','.jpg':'image/jpeg','.jpeg':'image/jpeg','.webp':'image/webp','.gif':'image/gif','.wav':'audio/wav','.mp3':'audio/mpeg','.ogg':'audio/ogg','.flac':'audio/flac','.mp4':'video/mp4','.webm':'video/webm','.woff2':'font/woff2','.ttf':'font/ttf','.otf':'font/otf'}[path.extname(name)] ?? 'application/octet-stream');
async function requestData(req, limit = 144*1024*1024) { let length=0;const chunks=[];for await(const chunk of req){length+=chunk.length;if(length>limit)throw new StoreError('请求超过限制',413);chunks.push(chunk);}return Buffer.concat(chunks); }
export async function startServer({studioRoot,projectRoot,port=0,dev=false,pageCloseGraceMs=3000}={}) {
  const store=new ProjectStore(),capability=randomBytes(32).toString('hex'); let origin='',vite;
  if(projectRoot)await store.open(projectRoot);
  if(dev){const {createServer}=await import('vite');vite=await createServer({root:studioRoot,configFile:path.join(studioRoot,'vite.config.mjs'),server:{middlewareMode:true},appType:'spa'});}
  const respond=(res,status,value)=>{res.writeHead(status,{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store'});res.end(JSON.stringify(value));};
  const pages=pageLifetime(()=>close(),pageCloseGraceMs);
  const pageId=id=>{if(typeof id!=='string'||!/^[a-f0-9-]{36}$/.test(id))throw new StoreError('页面会话无效',400);return id;};
  const server=http.createServer(async(req,res)=>{
    try {
      if(req.headers.host!==new URL(origin).host)throw new StoreError('主机地址无效',403);
      const url=new URL(req.url,origin),route=url.pathname;
      if(route.startsWith('/api/')) {
        // sendBeacon cannot set Authorization. Its small close notification must
        // still prove the capability and exact same origin; no token in a URL.
        if(req.method==='POST' && route==='/api/page-close') {
          if(req.headers.origin!==origin)throw new StoreError('仅允许当前作者页面同源操作。',403);
          const body=JSON.parse((await requestData(req,2048)).toString('utf8')||'{}');
          if(body.capability!==capability)throw new StoreError('作者会话已失效',401);
          pages.leave(pageId(body.id));return respond(res,200,{closed:true});
        }
        if(req.headers.authorization!==`Bearer ${capability}`)throw new StoreError('作者会话已失效，请从启动器重新打开。',401);
        if(req.method!=='GET' && req.headers.origin!==origin)throw new StoreError('仅允许当前作者页面同源操作。',403);
        if(req.method==='GET' && route==='/api/session')return respond(res,200,{open:!!store.root,snapshot:store.root?await store.load():null});
        if(req.method==='GET' && route==='/api/launcher')return respond(res,200,{pid:process.pid,studioRoot:path.resolve(studioRoot),projectRoot:store.root,lockId:store.lockId,pages:pages.count});
        if(req.method==='GET' && route==='/api/page-live')return pages.attach(pageId(url.searchParams.get('id')),res);
        if(req.method==='GET' && route==='/api/recovery')return respond(res,200,await store.recoveries());
        if(req.method==='GET' && route.startsWith('/api/recovery/'))return respond(res,200,await store.readRecovery(decodeURIComponent(route.split('/').pop())));
        if(req.method==='GET' && route==='/api/export') {const {bytes,skipped}=await store.exportBundle(url.searchParams.get('distribution')==='public'?'public':'software');res.writeHead(200,{'Content-Type':'application/gzip','Content-Disposition':'attachment; filename="manual.ptbmanual.gz"','Cache-Control':'no-store','X-Skipped-Assets':String(skipped.length)});return res.end(bytes);}
        if(req.method==='POST') {
          const body=JSON.parse((await requestData(req)).toString('utf8')||'{}');
          if(route==='/api/launch'){pages.reserve();return respond(res,200,{ready:true});}
          if(route==='/api/open')return respond(res,200,await store.open(body.path));
          if(route==='/api/create')return respond(res,200,await store.create(body.path,body.title));
          if(route==='/api/save')return respond(res,200,await store.save(body));
          if(route==='/api/draft')return respond(res,200,{id:await store.recovery(body)});
          if(route==='/api/asset'){if(typeof body.baseRevision!=='string')throw new StoreError('素材导入必须带工程版本');return respond(res,200,await store.addAsset(body));}
          if(route==='/api/import')return respond(res,200,await store.importBundle(body.path,Buffer.from(body.data,'base64')));
          if(route==='/api/build') {
            // A reading snapshot belongs to the opened project. It cannot choose an arbitrary output directory.
            const distribution=body.distribution==='public'?'public':'software',dir=await safePath(store.requireOpen(),distribution==='public'?'.studio/reading-public':'.studio/reading');
            try{await execute('python',[path.resolve(studioRoot,'../../scripts/manual/build.py'),'--project',store.root,'--output',dir,'--distribution',distribution],{maxBuffer:16*1024*1024,windowsHide:true});}catch(e){throw new StoreError(String(e.stderr||e.message).slice(0,4000));}
            const project=JSON.parse(await fs.readFile(path.join(dir,'project.json'),'utf8')),chapters=[];for(const descriptor of project.chapters)chapters.push(JSON.parse(await fs.readFile(await safePath(dir,descriptor.path),'utf8')));
            return respond(res,200,{path:dir,project,chapters,distribution});
          }
          if(route==='/api/close'){respond(res,200,{closed:true});setTimeout(()=>void close(),100);return;}
        }
        throw new StoreError('作者接口不存在',404);
      }
      if(req.method!=='GET' && req.method!=='HEAD')throw new StoreError('方法不允许',405);
      if(req.method==='GET' && route==='/')pages.reserve();
      if(route.startsWith('/media/')||route.startsWith('/reading/')||route.startsWith('/reading-public/')) {
        // Media access requires the capability in the query because HTML media cannot add Authorization.
        if(url.searchParams.get('session')!==capability)throw new StoreError('媒体会话无效',403);
        const relative=decodeURIComponent(route.slice(route.startsWith('/media/')?7:route.startsWith('/reading-public/')?16:9)),root=route.startsWith('/media/')?store.requireOpen():await safePath(store.requireOpen(),route.startsWith('/reading-public/')?'.studio/reading-public':'.studio/reading');
        if(!relative.startsWith('assets/') && !relative.startsWith('chapters/') && relative!=='project.json')throw new StoreError('只允许读取工程内容',403);
        const file=await safePath(root,relative),stat=await fs.stat(file),range=req.headers.range;
        if(range){const match=/^bytes=(\d+)-(\d*)$/.exec(range);if(!match)throw new StoreError('媒体范围无效',416);const start=Number(match[1]),end=match[2]?Math.min(Number(match[2]),stat.size-1):stat.size-1;if(start>end||start>=stat.size)throw new StoreError('媒体范围超出文件',416);const handle=await fs.open(file,'r');try{const bytes=Buffer.alloc(end-start+1);await handle.read(bytes,0,bytes.length,start);res.writeHead(206,{'Content-Type':mime(file),'Content-Length':bytes.length,'Content-Range':`bytes ${start}-${end}/${stat.size}`,'Accept-Ranges':'bytes','Cache-Control':'no-store'});res.end(bytes);}finally{await handle.close();}return;}
        res.writeHead(200,{'Content-Type':mime(file),'Content-Length':stat.size,'Accept-Ranges':'bytes','Cache-Control':'no-store'});return res.end(req.method==='HEAD'?undefined:await fs.readFile(file));
      }
      if(vite)return vite.middlewares(req,res,()=>respond(res,404,{error:'页面不存在'}));
      const root=path.join(studioRoot,'dist'),relative=decodeURIComponent(route)==='/'?'index.html':decodeURIComponent(route).replace(/^\//,'');
      let file;try{file=await safePath(root,relative);await fs.access(file);}catch(e){if(e.code!=='ENOENT')throw e;file=path.join(root,'index.html');}
      res.writeHead(200,{'Content-Type':mime(file),'Cache-Control':'no-store','Content-Security-Policy':"default-src 'self'; img-src 'self' data:; media-src 'self' blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"});return res.end(await fs.readFile(file));
    }catch(e){respond(res,e.status??500,{error:e.code==='ENOENT'?'文件不存在，请检查工程路径或素材。':e.message,...e.extra});}
  });
  await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(port,'127.0.0.1',resolve);});
  origin=`http://127.0.0.1:${server.address().port}`;
  let closing;
  const close=()=>closing??=(async()=>{pages.dispose();if(vite)await vite.close();await new Promise(resolve=>server.close(resolve));await store.queue;await store.release();})();
  return {server,store,capability,origin,url:origin+'/#'+capability,close,reservePage:()=>pages.reserve()};
}
