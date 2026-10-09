// Use exactly the production reader's parser before freezing a distribution.
import {readFile} from 'node:fs/promises';
import path from 'node:path';
import {parseManualProject,parseManualChapter} from '../src/manual/content.ts';
const root=path.resolve(process.argv[2]??'dist/manual');
const project=parseManualProject(JSON.parse(await readFile(path.join(root,'project.json'),'utf8')));
let nodes=0;
for(const descriptor of project.chapters){
  const chapter=parseManualChapter(JSON.parse(await readFile(path.join(root,descriptor.path),'utf8')),descriptor.id);
  const visit=n=>{nodes++;for(const child of n.content??[])visit(child);};visit(chapter.body);
}
console.log(JSON.stringify({success:true,chapters:project.chapters.length,assets:project.assets.length,nodes}));
