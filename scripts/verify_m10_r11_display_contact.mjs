import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {velumModel} from '../frontend/public/vocal-tract/anatomy.mjs';
import {displayLips} from '../frontend/public/vocal-tract/geometry.mjs';
const root=new URL('../output/validation/m10-r11/',import.meta.url);
const cases=JSON.parse(readFileSync(new URL('contact-cases.json',root),'utf8'));
const display=[],shapes=[];let reference;
for(const {name,state} of cases){
  const shown=displayLips(state),mesh=velumModel(shown),origin=state.contours.uvula[0],points=[];
  const native=state.meshes.find(m=>m.name==='tongue'),tongue=shown.meshes.find(m=>m.name==='tongue');
  assert.deepEqual(tongue.vertices.slice(0,34*tongue.points*3),native.vertices.slice(0,34*native.points*3),name+' dorsal vertices');
  const dorsal=mesh=>{const out=[];for(let i=0;i<mesh.triangles.length;i+=3){const t=mesh.triangles.slice(i,i+3);if(t.every(j=>j<34*mesh.points))out.push(...t);}return out;};
  assert.deepEqual(dorsal(tongue),dorsal(native),name+' dorsal contact triangles');
  for(let q=0;q<=12;q++)for(let j=4;j<=10;j++){
    points.push(mesh.vertices.slice((q*mesh.profile.length+j)*3,(q*mesh.profile.length+j)*3+3).map((v,k)=>v-(origin[k]??0)));
  }
  reference??=points;
  const deviation=Math.max(...points.map((p,i)=>Math.hypot(...p.map((v,k)=>v-reference[i][k]))));
  assert.ok(deviation<.00004,name+' display changed rigid uvula size/shape');
  shapes.push({name,maximum_deviation_cm:deviation});
  display.push({name,mesh,tongue});
}
writeFileSync(new URL('contact-display.json',root),JSON.stringify(display));
writeFileSync(new URL('uvula-shape-report.json',root),JSON.stringify(shapes,null,2));
console.log(JSON.stringify({cases:shapes.length,maxUvulaShapeDeviationCm:Math.max(...shapes.map(x=>x.maximum_deviation_cm))}));
