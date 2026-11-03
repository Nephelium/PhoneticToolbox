// Consume complete generated readers through the same parser and search used by
// the app. The caller owns and removes the temporary content after verification.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {parseManualProject,parseManualChapter,chapterSearchEntries,searchManual} from '../../frontend/src/manual/content.ts';
const scratch=process.argv[2],result={};
for(const locale of ['en','ja']){
  const folder=path.join(scratch,locale),project=parseManualProject(JSON.parse(fs.readFileSync(path.join(folder,'project.json'),'utf8')));
  assert.equal(project.language,locale);
  const entries=[];
  for(const descriptor of project.chapters){
    const chapter=parseManualChapter(JSON.parse(fs.readFileSync(path.join(folder,descriptor.path),'utf8')),descriptor.id);
    const current=chapterSearchEntries(chapter);
    assert(current.length>0,locale+'/'+descriptor.id+' has no search content');
    assert(current.some(item=>item.chapterId===chapter.id));
    entries.push(...current);
  }
  const query=locale==='en'?'fundamental frequency':'基本周波数';
  assert(searchManual(entries,query).length>0,locale+' full-text scientific search is empty');
  result[locale]={chapters:project.chapters.length,searchEntries:entries.length,scientificSearch:query};
}
process.stdout.write(JSON.stringify(result));
