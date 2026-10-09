import test from 'node:test';
import assert from 'node:assert/strict';
import {createSSRApp,h} from 'vue';
import {renderToString} from 'vue/server-renderer';
import {CHAPTER_SCHEMA,MANUAL_SCHEMA,type ManualChapter,type ManualChapterDescriptor,type ManualAsset} from '../src/manual/types.ts';
import {assetUrl,chapterSections,chapterSearchEntries,manualAnchor,mergeManualAssets,nodeText,parseManualChapter,parseManualProject,safeLink,safeMediaSource,safeRelativePath,searchManual} from '../src/manual/content.ts';
import {ManualChapterRepository} from '../src/manual/loader.ts';
import {ManualMediaCoordinator} from '../src/manual/media.ts';
import {boundedSpan,contentStyle,textStyle} from '../src/manual/styles.ts';
import TextNode from '../src/manual/TextNode.ts';
import {manualTocSections,manualReadingPath,headingAtReadingLine} from '../src/manual/navigation.ts';

// Synthetic stress content verifies format/behaviour. It is never shipped as a chapter.
const chapter=(id='M08'):ManualChapter=>({schemaVersion:CHAPTER_SCHEMA,id,title:'测试用样章',body:{type:'doc',content:[
  {type:'heading',attrs:{id:'input',level:2},content:[{type:'text',text:'输入与 IPA a̤'}]},
  {type:'paragraph',attrs:{id:'parameters'},content:[{type:'text',text:'基频设置 100 Hz，步骤只改变选中的音频。',marks:[{type:'bold'}]}]},
  {type:'image',attrs:{id:'image1',assetId:'sample-image',caption:'实际截图裁切说明'}},
  {type:'audio',attrs:{id:'audio1',assetId:'sample-audio',caption:'原音与合成结果试听'}},
  {type:'formula',attrs:{id:'formula1',latex:'f_0 = 1/T',display:true}},
  {type:'table',attrs:{id:'table1',caption:'参数范围',note:'频率单位为 Hz'},content:[{type:'tableRow',content:[{type:'tableHeader',attrs:{colspan:2},content:[{type:'paragraph',content:[{type:'text',text:'宽表头'}]}]}]}]},
  {type:'codeBlock',attrs:{id:'code1',language:'python'},content:[{type:'text',text:'print("中文 <script>")'}]},
  {type:'newEditorNode',attrs:{id:'future'},content:[{type:'text',text:'未知节点保留的内容'}]}
]}});
const descriptor=(id:string):ManualChapterDescriptor=>({id,title:id,path:'chapters/'+id+'.json'});

test('reading outline keeps third-level siblings under their second-level section and preserves book numbering',()=>{
  const sections=[{id:'a',title:'输入',level:2},{id:'a1',title:'文件',level:3},{id:'a11',title:'编码',level:4},{id:'a2',title:'参数',level:3},{id:'b',title:'保存',level:2},{id:'b1',title:'格式',level:3}];
  assert.deepEqual(manualTocSections(sections,3).map(s=>[s.id,s.number,s.children.map(c=>[c.id,c.number])]),[
    ['a','3.1',[['a1','3.1.1'],['a2','3.1.2']]],['b','3.2',[['b1','3.2.1']]]
  ]);
  assert.deepEqual(manualTocSections([],3),[]);
});

test('heading, deep-heading and search-body targets resolve to the same visible reading ancestry',()=>{
  const heading=(id:string,level:number)=>({type:'heading',attrs:{id,level},content:[{type:'text',text:id}]});
  const sample:ManualChapter={...chapter(),body:{type:'doc',content:[heading('a',2),heading('a1',3),heading('detail',4),{type:'paragraph',attrs:{id:'search-hit'},content:[{type:'text',text:'关键词'}]},heading('a2',3),heading('b',2),{type:'paragraph',attrs:{id:'b-body'}},heading('b1',3)]}};
  assert.deepEqual(manualReadingPath(sample,'a'),{sectionId:'a'});
  for(const id of ['a1','detail','search-hit'])assert.deepEqual(manualReadingPath(sample,id),{sectionId:'a',subsectionId:'a1'});
  assert.deepEqual(manualReadingPath(sample,'a2'),{sectionId:'a',subsectionId:'a2'});
  assert.deepEqual(manualReadingPath(sample,'b-body'),{sectionId:'b'});
  assert.deepEqual(manualReadingPath(sample,'b1'),{sectionId:'b',subsectionId:'b1'});
  assert.deepEqual(manualReadingPath(sample,'missing'),{});
  assert.deepEqual(manualReadingPath(sample),{});
});

test('reading line tracks heading crossings in both directions and reaches the final section at the scroll limit',()=>{
  const state={viewportHeight:600,scrollTop:500,scrollHeight:2000};
  assert.equal(headingAtReadingLine([{id:'a',top:-300},{id:'a1',top:40},{id:'b',top:160}],state),'a1');
  assert.equal(headingAtReadingLine([{id:'a',top:-200},{id:'a1',top:120},{id:'b',top:260}],state),'a');
  assert.equal(headingAtReadingLine([{id:'a',top:140}],{...state,scrollTop:0}),undefined);
  assert.equal(headingAtReadingLine([{id:'a',top:-300},{id:'b',top:200}],{...state,scrollTop:1400}),'b');
  assert.equal(headingAtReadingLine([{id:'a',top:140}],{viewportHeight:600,scrollTop:0,scrollHeight:600}),undefined);
  assert.equal(headingAtReadingLine([],state),undefined);
});

test('versioned project preserves optional software media while rejecting unsafe publication paths',()=>{
  const value={schemaVersion:MANUAL_SCHEMA,id:'ptb-manual',title:'使用说明',chapters:[descriptor('M08')],assets:[{id:'sample-audio',path:'assets/audio/中文 空格.wav',kind:'audio',distribution:'software-only',git:false,duration:1.5}],searchIndex:[{chapterId:'M08',targetId:'audio1',text:'试听原音'}]};
  assert.equal(parseManualProject(value).assets[0].distribution,'software-only');
  for(const path of ['../source.wav','/absolute.wav','C:/private.wav','assets\\a.wav','assets/%2e%2e/a.wav','assets/%252e%252e/a.wav','assets/a.wav?key=secret','https://example.com/a.wav']){
    assert.equal(safeRelativePath(path),false,path);
    assert.throws(()=>parseManualProject({...value,assets:[{...value.assets[0],path}]}));
  }
  assert.throws(()=>parseManualProject({...value,chapters:[descriptor('M08'),descriptor('M08')]}),/重复标识/);
  assert.throws(()=>parseManualProject({...value,searchIndex:[{chapterId:'missing',text:'不应索引'}]}),/引用无效/);
  assert.throws(()=>parseManualProject({...value,assets:[{...value.assets[0],duration:'1s'}]}),/元数据/);
});

test('chapter IDs remain independent of titles and unknown nodes remain intact',()=>{
  const value=chapter();value.title='已改名的样章';
  const parsed=parseManualChapter(value,'M08');assert.equal(parsed.body.content?.at(-1)?.type,'newEditorNode');
  assert.deepEqual(chapterSections(parsed),[{id:'input',title:'输入与 IPA a̤',level:2}]);
  assert.equal(manualAnchor('M08','input'),'manual-M08--input');
  assert.throws(()=>parseManualChapter(value,'M09'),/标识不一致/);
  value.body.content!.push({type:'paragraph',attrs:{id:'input'},content:[]});assert.throws(()=>parseManualChapter(value),/重复标识/);
  assert.throws(()=>parseManualChapter({...chapter(),schemaVersion:'ptb-manual-chapter/2'}),/版本不兼容/);
});

test('search covers captions, table notes, IPA, code and formula without fetching chapter media',()=>{
  const entries=chapterSearchEntries(chapter());
  for(const [query,targetId] of [['截图','image1'],['原音 合成','audio1'],['频率单位','table1'],['f_0','formula1'],['print','code1'],['未知节点','future']])assert.equal(searchManual(entries,query)[0].targetId,targetId,query);
  assert.equal(searchManual(entries,'').length,0);assert.equal(searchManual(entries,'不在样章里的词').length,0);
  assert.equal(nodeText(chapter().body).includes('宽表头'),true);
});

test('asset resolution supports relocatable Unicode paths and rejects active schemes',()=>{
  assert.equal(assetUrl('./manual/','assets/中文 空格.wav'),'./manual/assets/%E4%B8%AD%E6%96%87%20%E7%A9%BA%E6%A0%BC.wav');
  assert.equal(assetUrl('http://127.0.0.1:3000/manual','assets/x.wav'),'http://127.0.0.1:3000/manual/assets/x.wav');
  for(const base of ['javascript:alert(1)','file:///C:/private','data:text/html,evil','//remote/'])assert.equal(assetUrl(base,'assets/x.wav'),null);
  for(const href of ['javascript:alert(1)','data:text/html,<script>','file:///C:/private','https://name:secret@example.com/','https://example.com/a\n'])assert.equal(safeLink(href),null);
  assert.equal(safeLink('https://example.org/paper'),'https://example.org/paper');assert.equal(safeLink('#input'),'#input');
  assert.equal(safeMediaSource('/media/assets/a.wav?session=local'),'/media/assets/a.wav?session=local');
  assert.equal(safeMediaSource('javascript:alert(1)'),null);assert.equal(safeMediaSource('data:text/html,evil'),null);
});

test('software-only asset overlay does not overwrite a public ID',()=>{
  const image:ManualAsset={id:'image',path:'assets/image.png',kind:'image',distribution:'public'};
  const audio:ManualAsset={id:'audio',path:'assets/audio.wav',kind:'audio',distribution:'software-only',git:false};
  assert.deepEqual(mergeManualAssets([image],[audio]),[image,audio]);
  assert.throws(()=>mergeManualAssets([image],[{...audio,id:'image'}]),/冲突/);
  assert.throws(()=>mergeManualAssets([],[{...audio,path:'../private.wav'}]),/无效/);
});

test('chapter repository loads only requested JSON and bounds its cache',async()=>{
  const requested:string[]=[];const repo=new ManualChapterRepository(async d=>{requested.push(d.id);return chapter(d.id);},2);
  const signal=new AbortController().signal;
  await repo.load(descriptor('M08'),signal);await repo.load(descriptor('M08'),signal);assert.deepEqual(requested,['M08']);
  await repo.load(descriptor('M09'),signal);await repo.load(descriptor('M11'),signal);assert.equal(repo.get('M08'),undefined);assert.equal(repo.values().length,2);
  await repo.load(descriptor('M08'),signal);assert.deepEqual(requested,['M08','M09','M11','M08']);
});

test('late aborted chapter does not enter cache or replace the latest chapter',async()=>{
  let finish!:(value:ManualChapter)=>void;const repo=new ManualChapterRepository(d=>d.id==='M08'?new Promise(resolve=>finish=resolve):Promise.resolve(chapter(d.id)));
  const old=new AbortController();const delayed=repo.load(descriptor('M08'),old.signal);old.abort();
  const latest=await repo.load(descriptor('M09'),new AbortController().signal);finish(chapter('M08'));
  await assert.rejects(delayed,{name:'AbortError'});assert.equal(latest.id,'M09');assert.equal(repo.get('M08'),undefined);assert.equal(repo.get('M09')?.id,'M09');
});

test('media ownership pauses the previous example and workbench, and rejects protected playback',()=>{
  let workbenchPauses=0,announcements=0,allowed=true;
  const coordinator=new ManualMediaCoordinator({assertAllowed(){if(!allowed)throw Error('录音正在进行');},pauseWorkbench(){workbenchPauses++;},announce(){announcements++;}});
  const a={pauses:0,pause(){this.pauses++;}},b={pauses:0,pause(){this.pauses++;}};
  coordinator.activate(a);coordinator.activate(b);assert.equal(a.pauses,1);assert.equal(workbenchPauses,2);assert.equal(announcements,2);
  coordinator.release(a);coordinator.pause();assert.equal(b.pauses,1);
  allowed=false;assert.throws(()=>coordinator.activate(a),/录音/);assert.equal(a.pauses,2);
  allowed=true;assert.throws(()=>coordinator.activate(a,false),/当前任务/);assert.equal(workbenchPauses,2);
});

test('formatted text is escaped and dangerous links/inline CSS cannot execute',async()=>{
  const app=createSSRApp({render:()=>h(TextNode,{chapterId:'M08',node:{type:'text',text:'<script>alert(1)</script>',marks:[{type:'bold'},{type:'link',attrs:{href:'javascript:alert(1)'}}]}})});
  const html=await renderToString(app);assert(html.includes('&lt;script&gt;'));assert(!html.includes('href="javascript:'));assert(html.includes('<strong>'));
  assert.deepEqual(textStyle({color:'url(javascript:evil)',backgroundColor:'red;position:fixed',fontFamily:'bad";background:url(x)',fontSize:'999px'}),{});
  assert.deepEqual(contentStyle({textAlign:'left;position:fixed',lineHeight:99,marginTop:'calc(1px)',indent:-2}),{});
  assert.equal(boundedSpan(0),undefined);assert.equal(boundedSpan(2),2);
  assert.equal(textStyle({color:'#123456',colorDark:'#abcdef',fontSize:'18px'})['--manual-color-dark'],'#abcdef');
});
