import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {spawn} from 'node:child_process';
import {m17AuthorTool} from '../frontend/tools/m17-author-server.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const {createServer}=await import(pathToFileURL(path.join(root,'frontend/node_modules/vite/dist/node/index.js')));
const {plugin,capability}=m17AuthorTool();
const server=await createServer({root:path.join(root,'frontend'),plugins:[plugin],optimizeDeps:{entries:['index.html']},server:{host:'127.0.0.1',port:0},logLevel:'error'});await server.listen();
const url=server.resolvedUrls.local[0]+'__m17_author#'+capability;
// The capability stays in this process and the local browser, never in logs.
spawn('powershell.exe',['-NoProfile','-Command',`Start-Process -FilePath '${url}'`],{windowsHide:true,stdio:'ignore'});
console.log('音标内容维护工具已启动。保存后点击页面中的关闭维护工具。');
process.once('SIGINT',()=>void server.close());process.once('SIGTERM',()=>void server.close());
