import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {connectedAirway,nasalOutlet,sectionPaths,airwayGuide,nearestSection,displayLips} from '../../gui/resources/vocal_tract/geometry.mjs';
import {registerNose,velumModel} from '../../gui/resources/vocal_tract/anatomy.mjs';
import '../../gui/resources/vocal_tract/keyframes.js';
import {PoseHistory,moveControl} from '../../gui/resources/vocal_tract/controls.mjs';
const s=JSON.parse(await readFile(new URL('./fixtures/airway-a.json',import.meta.url)));
const nose=registerNose(JSON.parse(await readFile(new URL('../../gui/resources/vocal_tract/assets/nasal.json',import.meta.url))));
const outlet=nasalOutlet(nose);
function topology(mesh){
  const n=mesh.vertices.length/3,parents=Array.from({length:n},(_,i)=>i),used=new Set(),edges=new Map(),directions=new Map();
  function root(i){while(parents[i]!==i){parents[i]=parents[parents[i]];i=parents[i];}return i;}
  for(let i=0;i<mesh.triangles.length;i+=3)for(let j=0;j<3;j++){const a=mesh.triangles[i+j],b=mesh.triangles[i+(j+1)%3];assert.ok(a>=0&&a<n);assert.notEqual(a,b);used.add(a);used.add(b);parents[root(a)]=root(b);const k=Math.min(a,b)*n+Math.max(a,b);edges.set(k,(edges.get(k)||0)+1);directions.set(k,(directions.get(k)||0)+(a<b?1:-1));}
  for(const [key,count] of edges)if(count===2)assert.equal(directions.get(key),0,'adjacent triangles must have consistent winding');
  return {components:new Set([...used].map(root)).size,boundaries:[...edges.values()].filter(n=>n===1).length,nonmanifold:[...edges.values()].filter(n=>n>2).length};
}
test('open port joins oral and nasal surfaces through shared seam indices',()=>{
  const mesh=connectedAirway({...s,nasal:{...s.nasal,port_area:.7}},nose,outlet),t=topology(mesh);
  assert.equal(t.components,1);assert.equal(t.nonmanifold,0);assert.equal(t.boundaries,257); // two nostrils + lips + glottis
  let area=0;const ring=mesh.junction.throat;for(let i=0;i<ring.length;i++){const a=ring[i],b=ring[(i+1)%ring.length];area+=a[0]*b[2]-b[0]*a[2];}
  assert.ok(Math.abs(Math.abs(area/2)-.7)<1e-9);assert.ok(mesh.vertices.every(Number.isFinite));
});
test('closed port separates air spaces with capped seams, not a dangling tube',()=>{
  const mesh=connectedAirway({...s,nasal:{...s.nasal,port_area:0}},nose,outlet),t=topology(mesh);
  assert.equal(t.components,2);assert.equal(t.nonmanifold,0);assert.equal(t.boundaries,257);assert.equal(mesh.junction.open,false);
});
test('port throat follows continuous requested areas including a small opening',()=>{
  for(const area of [.001,.1,1.5]){const m=connectedAirway({...s,nasal:{...s.nasal,port_area:area}},nose,outlet);assert.ok(m.junction.open);assert.ok(m.vertices.every(Number.isFinite));const t=topology(m);assert.equal(t.components,1);assert.equal(t.nonmanifold,0);}
});
test('one drag is one undo step; redo and a new edit preserve expected history',()=>{
  const h=new PoseHistory(),a={params:[0,1],preset:'a'},b={params:[.2,1],preset:''},c={params:[.7,1],preset:''};
  h.begin(a);h.begin(b);h.commit(c);assert.equal(h.past.length,1);assert.deepEqual(h.undo(c),a);assert.deepEqual(h.redo(a),c);
  assert.deepEqual(h.undo(c),a);h.begin(a);h.commit(b);assert.equal(h.redo(b),null);assert.deepEqual(h.undo(b),a);
});
test('blade and side handles affect their native controls and respect limits',()=>{
  const names=['TCX','TCY','TBX','TBY','TTX','TTY','TS1','TS2','TS3'];const meta={parameters:names.map(name=>({name,min:-1,max:1}))},p=names.map(()=>0);
  const blade=moveControl(meta,p,'blade',.3,.5);assert.deepEqual(blade.params,[0,0,.3,.5,0,0,0,0,0]);
  for(let i=1;i<=3;i++){const side=moveControl(meta,p,'side'+i,0,5);assert.equal(side.params[names.indexOf('TS'+i)],.3);assert.ok(side.limited);}
  const braced=p.slice();braced[names.indexOf('TS1')]=1;
  assert.equal(moveControl(meta,braced,'side1',0,-.1).params[names.indexOf('TS1')],.27);
  assert.deepEqual(moveControl(meta,braced,'side1',.1,0).params,braced);
});
test('lip width and pitch are restored by undo and redo',()=>{
  const h=new PoseHistory(),a={params:[0,1],preset:'',lip_width:.8,f0:120},b={...a,lip_width:1.3,f0:180};
  h.begin(a);h.commit(b);assert.deepEqual(h.undo(b),a);assert.deepEqual(h.redo(a),b);
});
test('soft-palate sagittal contour is cut from the displayed 3D volume',()=>{
  const model=velumModel(s),paths=sectionPaths(model);
  assert.equal(paths.length,1);
  for(const p of paths[0])assert.ok(model.profile.some(q=>Math.hypot(p[0]-q[0],p[1]-q[1])<1e-5));
  const t=topology(model);assert.equal(t.components,1);assert.equal(t.boundaries,0);assert.equal(t.nonmanifold,0);
});
test('registered nasal floor clears the hard-palate reference',()=>{
  for(let i=0;i<nose.vertices.length;i+=3){const [x,y,z]=nose.vertices.slice(i,i+3);if(x>=0&&x<=4&&Math.abs(z)<1.2){
    const nearest=s.contours.upper_cover.slice(13,20).reduce((a,b)=>Math.abs(a[0]-x)<Math.abs(b[0]-x)?a:b);
    assert.ok(y>nearest[1]+.15,'nasal air must not overlap the palatal roof');
  }}
});
test('all eight vowel velums lower and open posteriorly without broken tissue meshes',async()=>{
  const cases=JSON.parse(await readFile(new URL('./fixtures/velum-presets.json',import.meta.url)));
  for(const name of new Set(cases.map(s=>s.name))){const poses=cases.filter(s=>s.name===name).map(velumModel);
    assert.ok(poses[2].tip[1]<poses[0].tip[1]-.5);
    assert.equal(poses[0].gap,0);assert.ok(poses[2].gap>poses[1].gap&&poses[1].gap>0);
    for(const p of poses){const t=topology(p);assert.equal(t.boundaries,0);assert.equal(t.nonmanifold,0);assert.equal(sectionPaths(p).length,1);assert.ok(p.profile.every(v=>v[0]>=p.wall+p.gap-1e-9));}
  }
});

test('air-space guides follow edited tongue boundaries and do not bridge midline closure',()=>{
  const moved=structuredClone(s),i=80,base=airwayGuide(s)[i];
  moved.airway_sections[i].lower[48]+=.4;
  const changed=airwayGuide(moved)[i],normal=s.centerline[i].slice(3);
  assert.ok(Math.abs(changed.point[1]-base.point[1]-.2*normal[1])<1e-9);
  assert.equal(changed.position,base.position);assert.equal(nearestSection(s,base.position),i);
  assert.deepEqual(changed.upper,base.upper);
  moved.airway_sections[i].upper[48]=null;assert.equal(airwayGuide(moved)[i],null);
});

test('positive nasal openings clear the inclined posterior wall, including line thickness',async()=>{
  const cases=JSON.parse(await readFile(new URL('./fixtures/velum-presets.json',import.meta.url)));
  for(const state of cases)for(const area of [.01,.05,.2,.5]){
    const model=velumModel({...state,nasal:{...state.nasal,port_area:area}});
    for(const p of model.profile)assert.ok(p[0]-model.wallAt(p[1])>=model.gap+model.relief-1e-7);
    assert.equal(sectionPaths(model).length,1);
  }
});

test('lip relief retains the true midline, is shared by both views, and leaves native state intact',async()=>{
  const state=JSON.parse(await readFile(new URL('./fixtures/lips-teeth.json',import.meta.url))),before=JSON.stringify(state),shown=displayLips(state);
  assert.equal(JSON.stringify(state),before);
  for(const name of ['upper_lip','lower_lip']){
    const mesh=shown.meshes.find(m=>m.name===name),rib=(mesh.ribs-1)/2;
    for(let i=0;i<mesh.points;i++){
      const p=mesh.vertices.slice((rib*mesh.points+i)*3,(rib*mesh.points+i)*3+3);
      assert.ok(Math.abs(p[2])<1e-5);assert.deepEqual(shown.contours[name][i],p.slice(0,2));
    }
    assert.ok(shown.contours[name][4][0]>5,'midline lip must not be replaced by the lateral commissure');
  }
});

test('source pressure and voicing changes are independent undoable pose edits',()=>{
  const h=new PoseHistory(),a={params:[1],source:{mode:'voiced',pressure_pa:800,vibration:1}},b={params:[1],source:{mode:'whisper',pressure_pa:1200,vibration:0}};
  h.begin(a);h.commit(b);assert.deepEqual(h.undo(b),a);assert.deepEqual(h.redo(a),b);
  b.source.pressure_pa=10;assert.equal(h.undo(b).source.pressure_pa,800);
});
