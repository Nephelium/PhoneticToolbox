import test from 'node:test';
import assert from 'node:assert/strict';
import {eligible,missing,installPapersChannel,papers,type Paper} from '../src/platform/papers.ts';
import {rasterWidth,wheelZoom} from '../src/modules/paper-reading/pdf-view.ts';

test('M18 raster density follows monitor/UI scale and bounds decoded requests',()=>{
 assert.equal(rasterWidth(816,1.5),1224);
 assert.equal(rasterWidth(816,2,150),2448);
 assert.equal(rasterWidth(1632,1.5),2448);
 assert.equal(rasterWidth(816,1),1020);
 assert.equal(rasterWidth(2448,3),4096);
});
test('M18 wheel zoom handles pixels/lines/pages, reverses direction and stays within limits',()=>{
 assert(wheelZoom(100,-120,0,800)>100);
 assert(wheelZoom(100,120,0,800)<100);
 assert.equal(wheelZoom(100,-3,1,800),wheelZoom(100,-48,0,800));
 assert.equal(wheelZoom(100,-1,2,800),wheelZoom(100,-120,0,800));
 assert.equal(wheelZoom(300,-120,0,800),300);
 assert.equal(wheelZoom(25,120,0,800),25);
});
test('M18 date boundary includes selected day and excludes future, history skips complete papers',()=>{
 const entries=[{publishedAt:'2026-10-06',downloaded:false},{publishedAt:'2026-10-07',downloaded:true},{publishedAt:'2026-10-07',downloaded:false},{publishedAt:'2026-10-08',downloaded:false}] as Paper[];
 assert.equal(eligible(entries,'2026-10-07','2026-10-07').length,2);
 assert.equal(missing(entries,'2026-10-06','2026-10-07').length,2);
});
test('M18 disconnected browser does not silently fetch remote documents',async()=>{
 installPapersChannel(undefined);await assert.rejects(papers.status(),/桌面版/);
});
test('M18 existing acquired papers and new supplemental analysis remain eligible',()=>{
 const entries=[{publishedAt:'2026-10-01',updatedAt:'2026-10-10',acquired:true,downloaded:false},
 {publishedAt:'2026-10-01',acquired:true,downloaded:true},
 {publishedAt:'2026-10-01',updatedAt:'2026-10-11',acquired:true,downloaded:false}] as Paper[];
 assert.equal(eligible(entries,'2026-10-10','2026-10-10').length,2);
 assert.equal(missing(entries,'2026-10-10','2026-10-10').length,1);
});
test('M18 cancellation removes pending request and ignores late native results',async()=>{
 let ready:(id:string,payload:string)=>void=()=>{};let id='';const cancelled:string[]=[];
 installPapersChannel({ready:{connect(fn){ready=fn;}},progress:{connect(){}},request(key){id=key;},cancel(key){cancelled.push(key);}});
 const controller=new AbortController();const promise=papers.download('2026-10-07',controller.signal);controller.abort();
 await assert.rejects(promise,/取消/);assert.deepEqual(cancelled,[id]);ready(id,JSON.stringify({ok:true,value:{}}));installPapersChannel(undefined);
});
test('M18 abandoning an annotation read cancels native work and ignores its late response',async()=>{
 let ready:(id:string,payload:string)=>void=()=>{};let id='',operation='';const cancelled:string[]=[];
 installPapersChannel({ready:{connect(fn){ready=fn;}},progress:{connect(){}},request(key,payload){id=key;operation=JSON.parse(payload).operation;},cancel(key){cancelled.push(key);}});
 const controller=new AbortController();const result=papers.annotations('local_fixture','original',controller.signal);controller.abort();
 await assert.rejects(result,/取消/);assert.equal(operation,'annotations');assert.deepEqual(cancelled,[id]);
 ready(id,JSON.stringify({ok:true,value:{revision:0,items:[]}}));installPapersChannel(undefined);
});

test('paper folder collision keeps the raw diagnostic and literal filename while a single failure relocalizes',async()=>{
 const {setLocale,messageRef}=await import('../src/i18n/index.ts');
 const {uiErrorMessage}=await import('../src/modules/module-localization.ts');
 const filename='说明-{page}-é.txt';
 const raw='上一级已有同名文件或文件夹：'+filename+'。未移动文件，也未删除收藏夹。';
 let ready:(id:string,payload:string)=>void=()=>{};
 installPapersChannel({ready:{connect(fn){ready=fn;}},progress:{connect(){}},request(id){ready(id,JSON.stringify({ok:false,error:raw,display:{key:'native.errors.papers.folderCollision',values:{filename}}}));},cancel(){}});
 const notice=messageRef('');
 try{
  await setLocale('zh-CN');
  try{await papers.deleteFolder('owned collection');assert.fail('Expected actual refusal envelope');}catch(error){notice.value=uiErrorMessage(error);}
  assert.equal(notice.value,raw);
  for(const code of ['en','ja'] as const){await setLocale(code);assert.notEqual(notice.value,raw);assert(notice.value.includes(filename));assert(!notice.value.includes('{filename}'));}
  await setLocale('zh-CN');assert.equal(notice.value,raw);
 }finally{installPapersChannel(undefined);await setLocale('zh-CN');}
});

test('one native background failure translates its product prefix and known reason without rewriting unknown diagnostics',async()=>{
 const {paperStatusWarning}=await import('../src/platform/papers.ts');
 const {setLocale,messageRef}=await import('../src/i18n/index.ts');
 const known={warning:'自动检查未完成：操作已取消。',warningError:{ok:false,error:'操作已取消。'},firstLaunch:'',papers:[]};
 const raw='未登记诊断-{reason} / private.pdf',unknown={...known,warning:'自动检查未完成：'+raw,warningError:{error:raw}};
 const before=JSON.stringify([known,unknown]);
 const notice=messageRef(''),diagnostic=messageRef('');
 notice.value=paperStatusWarning(known);diagnostic.value=paperStatusWarning(unknown);
 try{
  await setLocale('zh-CN');assert.equal(notice.value,known.warning);
  for(const code of ['en','ja'] as const){await setLocale(code);assert(!notice.value.includes('自动检查'));assert(!notice.value.includes('操作已取消'));assert(diagnostic.value.endsWith(raw));assert(!diagnostic.value.startsWith('自动检查'));}
  assert.equal(JSON.stringify([known,unknown]),before);
 }finally{await setLocale('zh-CN');}
});
