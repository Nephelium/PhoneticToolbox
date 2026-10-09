import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),lock=JSON.parse(await fs.readFile(path.join(root,'package-lock.json'),'utf8')),records=[];
for(const [relative,entry]of Object.entries(lock.packages)){
  if(!relative)continue;
  const directory=path.join(root,relative);let pkg;try{pkg=JSON.parse(await fs.readFile(path.join(directory,'package.json'),'utf8'));}catch{continue;}
  const record={name:pkg.name,version:pkg.version,license:pkg.license??'unknown',repository:typeof pkg.repository==='string'?pkg.repository:pkg.repository?.url??null,homepage:pkg.homepage??null,scope:entry.dev?'author-development':'author-runtime',resolved:entry.resolved,integrity:entry.integrity,licenseFiles:[]};
  for(const file of await fs.readdir(directory)){if(!/^(license|copying|notice)(\..*)?$/i.test(file))continue;const source=path.join(directory,file);if(!(await fs.stat(source)).isFile())continue;const bytes=await fs.readFile(source),target='licenses/'+pkg.name.replace(/[@/]/g,'_')+'/'+file;await fs.mkdir(path.dirname(path.join(root,target)),{recursive:true});await fs.writeFile(path.join(root,target),bytes);record.licenseFiles.push({path:target,sha256:createHash('sha256').update(bytes).digest('hex')});}
  records.push(record);
}
records.sort((a,b)=>a.name.localeCompare(b.name));
const fonts=[];for(const [name,fontFile,license]of [['Doulos SIL','DoulosSIL-Regular.ttf','Doulos-OFL.txt'],['JetBrains Mono','JetBrainsMono-Regular.woff2','JetBrainsMono-OFL.txt']]){const source=path.resolve(root,'../../frontend/src/assets'),bytes=await fs.readFile(path.join(source,fontFile)),licenseBytes=await fs.readFile(path.join(source,license)),target='licenses/fonts/'+license;await fs.mkdir(path.dirname(path.join(root,target)),{recursive:true});await fs.writeFile(path.join(root,target),licenseBytes);fonts.push({name,source:'Existing verified product font asset',fontFile,sha256:createHash('sha256').update(bytes).digest('hex'),license:'OFL-1.1',licenseFile:target,licenseSha256:createHash('sha256').update(licenseBytes).digest('hex')});}
await fs.writeFile(path.join(root,'THIRD_PARTY.json'),JSON.stringify({verifiedAt:'2026-10-05',scope:'Standalone author tool only; excluded from product build',method:'Official npm lock integrity and installed package metadata/license files; Tiptap official docs/repository checked independently',packages:records,fonts},null,2)+'\n','utf8');
console.log(JSON.stringify({packages:records.length,licenses:records.reduce((n,r)=>n+r.licenseFiles.length,0),unknown:records.filter(r=>r.license==='unknown').map(r=>r.name)}));
