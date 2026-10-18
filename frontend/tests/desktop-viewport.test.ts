import test from 'node:test';
import assert from 'node:assert/strict';
import {workbenchViewportWidth,type ViewportDevice} from '../src/state/desktop-viewport.ts';
const computer:ViewportDevice={userAgent:'Mozilla Windows Chrome',platform:'Win32',touchPoints:0,coarsePointer:false,finePointer:true,screenWidth:1920};
test('desktop viewport leaves computers and touch laptops unchanged',()=>{
 assert.equal(workbenchViewportWidth(computer),null);
 assert.equal(workbenchViewportWidth({...computer,touchPoints:10,coarsePointer:true}),null);
 assert.equal(workbenchViewportWidth({...computer,userAgent:'Mozilla Macintosh Safari',platform:'MacIntel'}),null);
});
test('phones and Android tablets use a 1440 CSS pixel desktop canvas',()=>{
 assert.equal(workbenchViewportWidth({...computer,userAgent:'Mozilla iPhone Safari',screenWidth:390}),1440);
 assert.equal(workbenchViewportWidth({...computer,userAgent:'Mozilla Android Chrome',screenWidth:800}),1440);
});
test('iPad desktop Safari user agents still get a desktop canvas with or without a trackpad',()=>{
 for(const finePointer of [false,true])assert.equal(workbenchViewportWidth({...computer,userAgent:'Mozilla Macintosh Safari',platform:'MacIntel',touchPoints:5,finePointer,screenWidth:1024}),1440);
});
test('a touch-only tablet with an unrecognized UA is covered, and wide displays retain their width',()=>{
 assert.equal(workbenchViewportWidth({...computer,touchPoints:5,coarsePointer:true,finePointer:false,screenWidth:1280}),1440);
 assert.equal(workbenchViewportWidth({...computer,userAgent:'Mozilla iPad Safari',screenWidth:1920}),1920);
});
