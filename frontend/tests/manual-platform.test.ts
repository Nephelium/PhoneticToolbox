import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {manualForHost,manualIntroduction,webIntroductionChapter} from '../src/app/manual-project.ts';
import {moduleHelpTarget} from '../src/manual/help-navigation.ts';
import {parseManualProject,chapterSearchEntries,searchManual} from '../src/manual/content.ts';

const project=parseManualProject(JSON.parse(readFileSync(new URL('../../manual/project.json',import.meta.url),'utf8')));
// Author chapters can contain optional null anchors removed by the reader build.
project.searchIndex=project.chapters.flatMap(chapter=>chapterSearchEntries(
  JSON.parse(readFileSync(new URL('../../manual/'+chapter.path,import.meta.url),'utf8'))));

test('web introduction precedes the full shared manual; desktop keeps its existing order and searches',()=>{
  const web=manualForHost(project,'browser'),desktop=manualForHost(project,'desktop');
  assert.equal(web.chapters[0].id,webIntroductionChapter);
  assert.equal(desktop.chapters[0].id,'getting-started');
  assert.deepEqual(web.chapters.slice(1),desktop.chapters);
  for(const chapter of desktop.chapters)assert.equal(web.chapters.find(item=>item.id===chapter.id),chapter);
  assert.deepEqual(web.chapters.filter(item=>/^m\d\d$/.test(item.id)).map(item=>item.id).sort(),
    Array.from({length:19},(_,i)=>'m'+String(i+1).padStart(2,'0')));
  assert.ok(searchManual(web.searchIndex!,'扩容').some(hit=>hit.chapterId===webIntroductionChapter));
  assert.ok(desktop.searchIndex!.every(hit=>hit.chapterId!==webIntroductionChapter));
  assert.deepEqual(desktop.searchIndex,web.searchIndex!.filter(hit=>hit.chapterId!==webIntroductionChapter));
  assert.equal(project.chapters[0].id,webIntroductionChapter);
});

test('general help opens the host introduction and every module keeps its stable chapter target',()=>{
  for(const kind of ['browser','desktop'] as const){
    assert.deepEqual(moduleHelpTarget(undefined,undefined,manualIntroduction(kind)),{chapterId:manualIntroduction(kind)});
    for(let i=1;i<=19;i++){
      const id='M'+String(i).padStart(2,'0');
      assert.deepEqual(moduleHelpTarget(id,'section',manualIntroduction(kind)),{chapterId:id.toLowerCase(),targetId:'section'});
    }
  }
  assert.deepEqual(moduleHelpTarget(),{chapterId:'getting-started'});
});
