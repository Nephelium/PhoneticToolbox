import test from 'node:test';
import assert from 'node:assert/strict';
import {dentalSolids,subtractTeeth} from '../public/vocal-tract/rigid-contact.mjs';

const cube={name:'lower_teeth',ribs:2,points:5,vertices:[0,0,-1,1,0,-1,1,1,-1,0,1,-1,0,0,-1,0,0,1,1,0,1,1,1,1,0,1,1,0,0,1]};
const at=(m:any,i:number)=>m.vertices.slice(i*3,i*3+3);
test('M10-R12 dental contact cuts a crossing face even when all vertices are outside',()=>{
  const solids=dentalSolids([cube]),mesh={vertices:[-2,.5,0,3,.5,0,.5,3,0],triangles:[0,1,2]};
  const original=structuredClone(mesh),teeth=structuredClone(cube),cut=subtractTeeth(mesh,solids);
  assert.deepEqual(mesh,original);assert.deepEqual(cube,teeth);assert.ok(cut.triangles.length>3);
  let area=0;
  for(let i=0;i<cut.triangles.length;i+=3){
    const pts=cut.triangles.slice(i,i+3).map(j=>at(cut,j));
    area+=Math.abs((pts[1][0]-pts[0][0])*(pts[2][1]-pts[0][1])-(pts[2][0]-pts[0][0])*(pts[1][1]-pts[0][1]))/2;
    for(let a=0;a<=10;a++)for(let b=0;b<=10-a;b++){
      const p=[0,1,2].map(k=>(pts[0][k]*a+pts[1][k]*b+pts[2][k]*(10-a-b))/10);
      assert.ok(!(p[0]>1e-7&&p[0]<1-1e-7&&p[1]>1e-7&&p[1]<1-1e-7));
    }
  }
  assert.ok(Math.abs(area-(6.25-.5))<1e-8);
});
test('M10-R12 preserves native protected contact triangles and does not move teeth',()=>{
  const mesh={vertices:[-.1,.5,0,1.1,.5,0,.5,1.5,0],triangles:[0,1,2]};
  assert.deepEqual(subtractTeeth(mesh,dentalSolids([cube]),3),mesh);
  const moved={...cube,vertices:cube.vertices.map((v,i)=>i%3===0?v+20:v)};
  assert.deepEqual(subtractTeeth(mesh,dentalSolids([moved])),mesh);
  assert.notDeepEqual(subtractTeeth(mesh,dentalSolids([cube])),mesh);
});
