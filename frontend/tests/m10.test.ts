import test from 'node:test';
import assert from 'node:assert/strict';
import {frameIntervals,silentAt} from '../public/vocal-tract/keyframes.js';
import {moveControl} from '../public/vocal-tract/controls.mjs';
import {readFileSync} from 'node:fs';
import {registerNose} from '../public/vocal-tract/anatomy.mjs';
import {planeSegments,stitch,sectionPaths} from '../public/vocal-tract/geometry.mjs';
test('M10 interval names and exact consecutive boundaries',()=>{
  const frames=[{name:'/n/',duration:.6},{name:'/a/',duration:.3},{preset:'i',duration:1}];
  const out=frameIntervals(frames);assert.equal(out[0].name,'/n/');assert.equal(out[1].start,.6);assert.equal(out[2].end,1.9);
  frames[0].duration=1;assert.equal(frameIntervals(frames)[1].start,1);
});
test('M10 side drag reaches native lateral bracing range',()=>{
  const meta={parameters:[{name:'TS3',min:-1,max:1}]};
  assert.ok(moveControl(meta,[0],'side3',0,-1).params[0]<-.8);
  assert.equal(moveControl(meta,[0],'side3',0,.5).params[0],.15);
  assert.equal(moveControl(meta,[0],'side3',0,10).params[0],1);
});
test('M10 silent intervals exclude F0 at exact start and include next pose at end',()=>{
  const frames=[{duration:.2},{duration:.05,silent:true},{duration:.25}];
  assert.equal(silentAt(frames,.3999),false);
  assert.equal(silentAt(frames,.4),true);
  assert.equal(silentAt(frames,.4999),true);
  assert.equal(silentAt(frames,.5),false);
  assert.equal(frameIntervals(frames)[1].name,'静音');
  assert.equal(silentAt([{duration:.05,silent:true}],1),true);
});
test('M10 tongue root and hyoid have two-axis control',()=>{
  const meta={parameters:['TRX','TRY','HX','HY'].map(name=>({name,min:-10,max:10}))};
  assert.deepEqual(moveControl(meta,[0,0,0,0],'root',.2,-.3).params,[.2,-.3,0,0]);
  assert.deepEqual(moveControl(meta,[0,0,0,0],'hyoid',.1,-.1).params,[0,0,.1,-.1]);
});
test('M10 nasal sections stay within the reference head without artificial floor chords',()=>{
  const load=(name:string)=>JSON.parse(readFileSync(new URL('../public/vocal-tract/assets/'+name+'.json',import.meta.url),'utf8'));
  const original=load('nasal'),nose=registerNose(original),head=load('head');
  assert.equal(nose.triangles,original.triangles);
  assert.equal(nose.vertices.length,original.vertices.length);
  for(let i=0;i<head.vertices.length;i+=3){const [x,y,z]=head.vertices.slice(i,i+3);head.vertices.splice(i,3,(z-.127)*100+6.5,(y-1.85)*85-1,x*80);}
  const envelope=stitch(planeSegments(head.vertices,head.triangles))[0];
  const inside=(p:number[])=>{let value=false;for(let i=0,j=envelope.length-1;i<envelope.length;j=i++){const a=envelope[i],b=envelope[j];if((a[1]>p[1])!==(b[1]>p[1])&&p[0]<(b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0])value=!value;}return value;};
  for(const z of [-.5,.5]){
    const sections=sectionPaths(nose,z);
    for(const path of sections){
      assert.ok(Math.hypot(...path[0].map((v:number,k:number)=>v-path.at(-1)[k]))<.002,'slice already closed before SVG rendering');
      assert.ok(path.every(inside),'nasal reference protrudes beyond the head');
      for(let i=1;i<path.length;i++)assert.ok(Math.hypot(path[i][0]-path[i-1][0],path[i][1]-path[i-1][1])<2,'spurious long closing chord');
    }
  }
});
