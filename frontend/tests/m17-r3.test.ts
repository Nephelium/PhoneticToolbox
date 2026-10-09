import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {placeHover} from '../src/modules/ipa-plus/hover-position.ts';
import {registerSymbolAnimation,symbolAnimation} from '../src/modules/ipa-plus/playback.ts';
import {validateContent} from '../tools/m17-author-server.mjs';

test('M17 R3 hover avoids symbol and pointer at corners, center and 150% scale',()=>{
 for(const scale of [1,1.5])for(const [width,height] of [[1200,650],[720,450],[480,360]])for(const [x,y] of [[10,10],[width-40,10],[10,height-40],[width-40,height-40],[width/2,height/2]]){
  const bounds={left:0,top:0,right:width,bottom:height},anchor={left:x,top:y,right:x+24,bottom:y+24},pointer={x:x+12,y:y+12};
  const p=placeHover(anchor,pointer,bounds,scale),rect={left:p.left,top:p.top,right:p.left+p.width,bottom:p.top+p.maxHeight};
  assert(rect.left>=0&&rect.top>=0&&rect.right<=width&&rect.bottom<=height);
  assert(rect.right<=anchor.left||rect.left>=anchor.right||rect.bottom<=anchor.top||rect.top>=anchor.bottom,'symbol stays visible');
  assert(!(pointer.x>=rect.left&&pointer.x<=rect.right&&pointer.y>=rect.top&&pointer.y<=rect.bottom),'pointer stays clear');
 }
});
test('M17 R3 curator content rejects external assets, traversal, unknown keys and invalid animation versions',()=>{
 assert.deepEqual(validateContent({notesZh:'自写内容\n第二行',media:{audio:'m17-media/音频.wav',video:'m17-media/example.mp4',animation:{renderer:'vocal-tract',version:1,config:{frames:[]}}}}).notesZh,'自写内容\n第二行');
 for(const audio of ['https://x/a.wav','../a.wav','/a.wav','C:/a.wav','m17-media/a.wav?x','m17-media/a.mp4'])assert.throws(()=>validateContent({media:{audio}}));
 for(const value of [{html:'<script>'},{notesZh:9},{media:{animation:{renderer:'x',version:0,config:{}}}},{media:{animation:{renderer:'x',version:1,config:[]}}},{media:{unknown:'x'}}])assert.throws(()=>validateContent(value));
});
test('M17 R3 animation registration has explicit ownership and cannot replace another renderer',()=>{
 const renderer=()=>{},release=registerSymbolAnimation('test',renderer);assert.equal(symbolAnimation('test'),renderer);assert.throws(()=>registerSymbolAnimation('test',()=>{}));release();assert.equal(symbolAnimation('test'),undefined);
});
test('M17 R3 all product descriptions omit chat image labels and retain original IDs and references',()=>{
 const data=JSON.parse(fs.readFileSync(new URL('../src/modules/ipa-plus/data/catalog.json',import.meta.url),'utf8'));
 assert.equal(data.entries.length,625);
 for(const e of data.entries){for(const key of ['nameZh','descriptionZh','usageZh','contrastZh'])assert(!/图4|用户|构音和转写原则|自主概述/.test(e[key]),e.id+':'+key);for(const r of e.sourceRefs)assert(!/图4|用户|构音和转写原则|自主概述/.test(r.locator),e.id);}
});
