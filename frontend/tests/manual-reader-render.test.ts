import test from 'node:test';
import assert from 'node:assert/strict';
import {createSSRApp,h} from 'vue';
import {renderToString} from 'vue/server-renderer';
import {createServer} from 'vite';
import vue from '@vitejs/plugin-vue';
import type {ManualChapter,ManualProject} from '../src/manual/types.ts';

test('shared Vue document renders safe stress content and graceful missing-media feedback',async()=>{
  // Node exposes BroadcastChannel too. This test has no browser windows to coordinate.
  const broadcast=globalThis.BroadcastChannel;
  Object.defineProperty(globalThis,'BroadcastChannel',{configurable:true,writable:true,value:undefined});
  const server=await createServer({configFile:false,root:new URL('..',import.meta.url).pathname.replace(/^\/(?=[A-Z]:)/i,''),plugins:[vue()],server:{middlewareMode:true},appType:'custom',logLevel:'error'});
  try{
    const {default:ManualDocument}=await server.ssrLoadModule('/src/manual/ManualDocument.vue');
    const chapter:ManualChapter={schemaVersion:'ptb-manual-chapter/1',id:'stress',title:'仅用于测试的样章',body:{type:'doc',content:[
      {type:'heading',attrs:{id:'stable-section',level:2},content:[{type:'text',text:'中文与 ã̤'}]},
      {type:'paragraph',content:[{type:'text',text:'<script>alert(1)</script>',marks:[{type:'bold'}]},{type:'formula',attrs:{latex:'f_0=1/T',display:false}}]},
      {type:'table',attrs:{id:'merged-table',caption:'参数表',note:'表注'},content:[{type:'tableRow',content:[{type:'tableHeader',attrs:{colspan:2},content:[{type:'paragraph',content:[{type:'text',text:'合并表头'}]}]}]}]},
      {type:'image',attrs:{id:'missing-figure',assetId:'not-distributed',caption:'缺少软件素材'}},
      {type:'audio',attrs:{id:'audio-figure',assetId:'audio',caption:'测试播放器'}},
      {type:'codeBlock',attrs:{language:'python'},content:[{type:'text',text:'x = '},{type:'text',text:'"<script>"'}]},
      {type:'unknownExtension',attrs:{id:'original-extension'},content:[{type:'text',text:'原始内容仍被保留'}]}
    ]}};
    const html=await renderToString(createSSRApp({render:()=>h(ManualDocument,{chapter,chapterNumber:4,assets:[{id:'audio',path:'assets/测试.wav',kind:'audio',distribution:'software-only',duration:1,sampleRate:16000,channels:1}],assetBaseUrl:'/manual/'})}));
    assert(html.includes('id="manual-stress--stable-section"'));assert(html.includes('colspan="2"'));assert(html.includes('合并表头'));
    assert(/manual-section-number[^>]*>4\.1/.test(html));assert(html.includes('表 4-1：'));assert(html.includes('参数表'));assert(html.includes('图 4-1：缺少软件素材'));assert(html.includes('例音 4-1：测试播放器'));
    assert(html.includes('&lt;script&gt;alert(1)&lt;/script&gt;'));assert(!html.includes('<script>'));
    assert(html.includes('图片素材暂不可用'));assert(html.includes('not-distributed'));assert(html.includes('unknownExtension'));assert(html.includes('原始内容仍被保留'));
    assert(html.includes('preload="none"'));assert(html.includes('assets/%E6%B5%8B%E8%AF%95.wav'));assert(!html.includes('autoplay'));
    assert(html.includes('class="katex"'));assert(html.includes('<math'));assert(html.includes('x = &quot;&lt;script&gt;&quot;'));
    const empty=await renderToString(createSSRApp({render:()=>h(ManualDocument,{chapter:{...chapter,id:'m10',title:'生理参数合成',body:{type:'doc',content:[]}},assets:[]})}));
    assert(empty.includes('本章正文尚未提供'));assert(empty.includes('生理参数合成'));

    const {default:ManualReader}=await server.ssrLoadModule('/src/manual/ManualReader.vue');let loads=0;
    const project:ManualProject={schemaVersion:'ptb-manual/1',id:'test-book',title:'测试目录',chapters:[{id:'stress',title:'样章',path:'chapters/stress.json',sections:[{id:'parent',title:'默认可见小节',level:2},{id:'child',title:'尚未展开的小节',level:3}]}],assets:[]};
    const shell=await renderToString(createSSRApp({render:()=>h(ManualReader,{project,chapterLoader:async()=>{loads++;return chapter;}})}));
    assert(shell.includes('搜索说明书'));assert(shell.includes('样章'));assert.equal(loads,0,'no chapter or media fetch during pre-render');
    assert(shell.includes('默认可见小节'));assert(!shell.includes('尚未展开的小节'));assert(shell.includes('aria-expanded="false"'));
    const lease=await server.ssrLoadModule('/src/platform/capture-lease.ts');
    const {manualPlayback}=await server.ssrLoadModule('/src/manual/playback.ts');
    const example={pauses:0,pause(){this.pauses++;}};
    for(const owner of ['M05','M16']){lease.claimCapture(owner);assert.throws(()=>manualPlayback.activate(example),/采集|录音/);lease.releaseCapture(owner);}
    assert.equal(example.pauses,2,'real shared capture leases block native examples before workbench ownership changes');
  }finally{await server.close();Object.defineProperty(globalThis,'BroadcastChannel',{configurable:true,writable:true,value:broadcast});}
});
