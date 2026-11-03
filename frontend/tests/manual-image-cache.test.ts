import test from 'node:test';
import assert from 'node:assert/strict';
import {ManualImageCache} from '../src/manual/image-cache.ts';
import {sharedAssetUrl} from '../src/manual/content.ts';
import type {ManualAsset} from '../src/manual/types.ts';

test('duplicate image IDs share one payload read and URL, chapter return reuses bytes, versions stay isolated',async()=>{
 let reads=0,created=0;const revoked:string[]=[];
 const cache=new ManualImageCache(async()=>{reads++;return new Blob(['same PNG bytes']);},{createObjectURL:()=>`blob:owned-${++created}`,revokeObjectURL:url=>revoked.push(url)});
 const payload='assets/sha256.png',assetOverrides={first:payload,second:payload};
 cache.configure('en:1:source-a');cache.retain(Object.values(assetOverrides));
 assert.equal(cache.resolve(assetOverrides.first),null);assert.equal(cache.resolve(assetOverrides.second),null);
 await new Promise(resolve=>setTimeout(resolve,0));
 assert.equal(reads,1);assert.equal(created,1);
 assert.equal(cache.resolve(assetOverrides.first),cache.resolve(assetOverrides.second));
 cache.retain([]);assert.deepEqual(revoked,['blob:owned-1']);
 cache.retain([payload]);assert.equal(cache.resolve(payload),'blob:owned-2');assert.equal(reads,1);
 cache.configure('en:2:source-a');cache.retain([payload]);cache.resolve(payload);await new Promise(resolve=>setTimeout(resolve,0));assert.equal(reads,2);
 cache.configure('ja:2:source-b');cache.retain([payload]);cache.resolve(payload);await new Promise(resolve=>setTimeout(resolve,0));assert.equal(reads,3);
 cache.dispose();assert.equal(cache.urls.size,0);assert.equal(created,revoked.length);
});

test('late image reads cannot create URLs after language change or teardown',async()=>{
 let finish!:(blob:Blob)=>void,created=0;
 const cache=new ManualImageCache(()=>new Promise(resolve=>{finish=resolve;}),{createObjectURL:()=>{created++;return 'blob:unexpected';},revokeObjectURL:()=>{}});
 cache.configure('en:1:a');cache.retain(['assets/a.png']);cache.resolve('assets/a.png');cache.configure('ja:1:a');
 finish(new Blob(['old bytes']));await new Promise(resolve=>setTimeout(resolve,0));
 assert.equal(created,0);assert.equal(cache.urls.size,0);assert.equal(cache.pending.size,0);cache.dispose();
});

test('foreign shared assets resolve by stable identity after host packaging renames PNG/WAV',async()=>{
 const audio:ManualAsset={id:'original-audio',kind:'audio',path:'assets/software-only/original.wav',distribution:'software-only',caption:'Translated audio caption'};
 const image:ManualAsset={id:'scientific-figure',kind:'image',path:'assets/public/figure.png',distribution:'public',alt:'Translated scientific description'};
 const host=new Map<string,ManualAsset>([
  [audio.id,{...audio,path:'assets/packed/audio-hash.mp3',mime:'audio/mpeg'}],
  [image.id,{...image,path:'assets/packed/image-hash.webp',mime:'image/webp'}],
 ]);
 const before=JSON.stringify([audio,image]);
 assert.equal(sharedAssetUrl('/manual/',audio,host),'/manual/assets/packed/audio-hash.mp3');
 assert.equal(sharedAssetUrl('/manual/',image,host),'/manual/assets/packed/image-hash.webp');
 assert.equal(JSON.stringify([audio,image]),before,'translated captions and original pack paths are not rewritten');
});
