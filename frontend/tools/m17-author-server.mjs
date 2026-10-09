import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash,randomBytes,randomUUID} from 'node:crypto';
import {fileURLToPath} from 'node:url';

const here=path.dirname(fileURLToPath(import.meta.url));
const base=path.resolve(here,'../src/modules/ipa-plus/data');
const hash=text=>createHash('sha256').update(text).digest('hex');
const textKeys=['nameZh','nameEn','descriptionZh','usageZh','contrastZh','notesZh'];
const plain=value=>!!value&&typeof value==='object'&&!Array.isArray(value);
const localMedia=value=>typeof value==='string'&&/^(?:[\p{L}\p{N}_-]+\/)*[\p{L}\p{N}_ .-]+\.(?:mp3|wav|ogg|m4a|mp4|webm)$/u.test(value)&&!value.split('/').some(p=>p==='.'||p==='..');

export function validateContent(value){
 if(!plain(value)||Object.keys(value).some(k=>![...textKeys,'media'].includes(k)))throw Error('内容字段无效。');
 for(const key of textKeys)if(key in value&&(typeof value[key]!=='string'||value[key].length>20000))throw Error('文字须为不超过20000字符的文本。');
 if('media' in value){
  const media=value.media;if(!plain(media)||Object.keys(media).some(k=>!['audio','video','animation'].includes(k)))throw Error('演示配置无效。');
  for(const key of ['audio','video'])if(key in media&&(!localMedia(media[key])||!(key==='audio'?/\.(mp3|wav|ogg|m4a)$/:/\.(mp4|webm)$/).test(media[key])))throw Error('请填写项目内的音频或视频相对路径。');
  if('animation' in media){const a=media.animation;if(!plain(a)||Object.keys(a).some(k=>!['renderer','version','config'].includes(k))||!/^[-a-z0-9.]+$/.test(a.renderer)||!Number.isInteger(a.version)||a.version<1||!plain(a.config)||JSON.stringify(a.config).length>100000)throw Error('实时动画须包含接口名、正整数版本及JSON对象配置。');}
 }
 return value;
}

// Never imported by the product or vite.config. Enabled by the owner launcher only.
export function m17AuthorTool({contentPath=path.join(base,'symbol-content.json')}={}){
 const capability=randomBytes(32).toString('hex');let serial=Promise.resolve();
 const plugin={name:'m17-author-only',apply:'serve',configureServer(server){
  server.middlewares.use(async(req,res,next)=>{
   const url=new URL(req.url??'/', 'http://127.0.0.1');
   if(!url.pathname.startsWith('/__m17_author'))return next();
   const respond=(status,value)=>{res.statusCode=status;res.setHeader('Content-Type','application/json; charset=utf-8');res.end(JSON.stringify(value));};
   const host=req.headers.host??'',origin=req.headers.origin;
   if(!/^127\.0\.0\.1:\d+$/.test(host)||!['127.0.0.1','::ffff:127.0.0.1'].includes(req.socket.remoteAddress)||origin&&origin!=='http://'+host)return respond(403,{message:'维护工具只接受本机会话。'});
   res.setHeader('Cache-Control','no-store');res.setHeader('X-Content-Type-Options','nosniff');res.setHeader('Referrer-Policy','no-referrer');res.setHeader('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; object-src 'none'");
   if(url.pathname==='/__m17_author'&&req.method==='GET'){res.setHeader('Content-Type','text/html; charset=utf-8');res.end(await fs.readFile(path.join(here,'m17-author.html'),'utf8'));return;}
   if(url.pathname==='/__m17_author/editor.js'&&req.method==='GET'){res.setHeader('Content-Type','text/javascript; charset=utf-8');res.end(await fs.readFile(path.join(here,'m17-author.js'),'utf8'));return;}
   if(url.pathname==='/__m17_author/editor.css'&&req.method==='GET'){res.setHeader('Content-Type','text/css; charset=utf-8');res.end(await fs.readFile(path.join(here,'m17-author.css'),'utf8'));return;}
   if(req.headers['x-m17-session']!==capability)return respond(403,{message:'请从源码维护入口重新打开。'});
   if(url.pathname==='/__m17_author/stop'&&req.method==='POST'){respond(200,{message:'维护工具已关闭。'});setTimeout(()=>void server.close(),50);return;}
   if(url.pathname!=='/__m17_author/content')return respond(404,{message:'入口不存在。'});
   try{
    if(req.method==='GET'){
     await serial;const text=await fs.readFile(contentPath,'utf8'),data=JSON.parse(text);
     if(data.version!==1)throw Error('内容文件版本无法识别。');
     return respond(200,{...data,revision:hash(text),catalog:JSON.parse(await fs.readFile(path.join(base,'catalog.json'),'utf8')).entries});
    }
    if(req.method!=='PUT')return respond(405,{message:'操作不支持。'});
    if(!req.headers['content-type']?.startsWith('application/json'))return respond(415,{message:'须使用JSON内容。'});
    const chunks=[];let size=0;for await(const chunk of req){size+=chunk.length;if(size>200000)return respond(413,{message:'内容过长。'});chunks.push(chunk);}
    const request=JSON.parse(Buffer.concat(chunks).toString('utf8')),content=validateContent(request.content);
    const catalog=JSON.parse(await fs.readFile(path.join(base,'catalog.json'),'utf8'));
    if(!catalog.entries.some(e=>e.id===request.id))return respond(400,{message:'音标ID无效。'});
    const operation=serial.then(async()=>{
     const text=await fs.readFile(contentPath,'utf8'),data=JSON.parse(text);
     if(request.revision!==hash(text))return respond(409,{message:'内容已在另一窗口改变。请重新打开后合并，当前编辑保留。'});
     if(data.version!==1||!plain(data.entries))throw Error('内容文件版本无法识别。');
     data.entries[request.id]=content;
     const nextText=JSON.stringify(data,null,2)+'\n',temporary=contentPath+'.'+randomUUID()+'.pending';
     await fs.writeFile(temporary,nextText,{encoding:'utf8',flag:'wx'});await fs.rename(temporary,contentPath);
     respond(200,{revision:hash(nextText),message:'已保存到源码内容文件。'});
    });serial=operation.catch(()=>{});await operation;
   }catch(e){respond(400,{message:e instanceof SyntaxError?'JSON格式无效。':e.code?'内容文件读取或保存失败，当前编辑保留。':e.message});}
  });
 }};
 return {plugin,capability};
}
