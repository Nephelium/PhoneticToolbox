import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {connectedAirway,point,sectionPaths,planeSegments,stitch,airwayReferencePaths} from '../frontend/public/vocal-tract/geometry.mjs';
import {registerNose,velumModel} from '../frontend/public/vocal-tract/anatomy.mjs';
const root=new URL('../',import.meta.url),read=p=>JSON.parse(readFileSync(new URL(p,root),'utf8'));
const cases=read('output/validation/m10-r11/sweep.json'),nose=registerNose(read('frontend/public/vocal-tract/assets/nasal.json'));
const head=read('frontend/public/vocal-tract/assets/head.json');
for(let i=0;i<head.vertices.length;i+=3){const [x,y,z]=head.vertices.slice(i,i+3);head.vertices.splice(i,3,(z-.127)*100+6.5,(y-1.85)*85-1,x*80);}
const cross=(a,b,c)=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
function simple(p){for(let i=0;i<p.length;i++)for(let j=i+2;j<p.length;j++){
  if(i===0&&j===p.length-1)continue;const a=p[i],b=p[(i+1)%p.length],c=p[j],d=p[(j+1)%p.length];
  if(cross(a,b,c)*cross(a,b,d)<-1e-10&&cross(c,d,a)*cross(c,d,b)<-1e-10)return false;
}return true;}
function contains(p,poly){let yes=false;for(let i=0,j=poly.length-1;i<poly.length;j=i++){
  const a=poly[i],b=poly[j];if((a[1]>p[1])!==(b[1]>p[1])&&p[0]<(b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0])yes=!yes;
}return yes;}
const heads=new Map([0,.5,1,1.5,2].map(z=>[z,stitch(planeSegments(head.vertices,head.triangles,z))[0]]));
const results=[],bounds=[];let previous=null,maxDelta=0,maxBranchDelta=0,maxGateDelta=0,uvulaReference=null,maxUvulaDelta=0;
for(const {name,state:s} of cases){
  const p=velumModel(s),air=connectedAirway(s,nose);
  assert.ok(simple(p.profile),name+' self-intersecting palate');
  assert.ok(p.vertices.every(Number.isFinite)&&air.vertices.every(Number.isFinite),name+' nonfinite mesh');
  const nativeUvula=s.contours.uvula[0],uvula=[];
  for(let q=0;q<=12;q++)for(let j=4;j<=10;j++)uvula.push(p.vertices.slice((q*p.profile.length+j)*3,(q*p.profile.length+j)*3+3).map((v,k)=>v-(nativeUvula[k]??0)));
  uvulaReference??=uvula;
  const shapeDelta=Math.max(...uvula.map((v,i)=>Math.hypot(...v.map((x,k)=>x-uvulaReference[i][k]))));
  maxUvulaDelta=Math.max(maxUvulaDelta,shapeDelta);
  assert.ok(shapeDelta<.00004,name+' changed rigid uvula shape');
  const ring=air.junction.throat,area=Math.abs(ring.reduce((v,p,i)=>{const q=ring[(i+1)%ring.length];return v+p[0]*q[2]-q[0]*p[2];},0)/2);
  if(s.nasal.port_area>0)assert.ok(Math.abs(area-s.nasal.port_area)<1e-7,name+' gate area');
  assert.ok(sectionPaths(air.nasalReference,.5).length>0,name+' unified nasal slice');
  const reference=airwayReferencePaths(air);
  if(s.nasal.port_area>0){
    assert.ok(reference.some(p=>Math.min(...p.map(q=>q[1]))<-6&&Math.max(...p.map(q=>q[1]))>4),name+' disconnected oral/nasal section');
    for(const path of reference){
      const edges=path.slice(1).map((p,i)=>[path[i],p]).filter(e=>e.every(p=>p[0]<0&&p[1]>-3&&p[1]<1.5));
      for(let i=0;i<edges.length;i++)for(let j=i+1;j<edges.length;j++){
        const [a,b]=edges[i],[c,d]=edges[j];
        assert.ok(!(cross(a,b,c)*cross(a,b,d)<-1e-10&&cross(c,d,a)*cross(c,d,b)<-1e-10),name+' crossed posterior attachment');
      }
    }
  }
  if(name.startsWith('a-')&&previous){
    assert.equal(p.profile.length,previous.p.profile.length,name+' topology');
    const delta=Math.max(...p.profile.map((v,i)=>Math.hypot(v[0]-previous.p.profile[i][0],v[1]-previous.p.profile[i][1])));
    const branch=Math.hypot(...air.junction.start.map((v,i)=>v-previous.air.junction.start[i]));
    maxDelta=Math.max(delta,maxDelta);maxBranchDelta=Math.max(branch,maxBranchDelta);
    const gateDelta=Math.max(...ring.map((v,i)=>Math.hypot(...v.map((x,k)=>x-previous.air.junction.throat[i][k]))));
    maxGateDelta=Math.max(maxGateDelta,gateDelta);assert.ok(gateDelta<.15,name+' transverse port jump '+gateDelta);
    assert.ok(delta<.15,name+' palate jump '+delta);assert.ok(branch<.15,name+' branch jump '+branch);
  }
  if(name.startsWith('a-'))previous={p,air};
  if(['a-0.000','a-0.750','a-1.500','i-1.500','u-1.500','n-1.500'].includes(name)){
    for(const [z,h] of heads){const nasal=planeSegments(air.nasalReference.vertices,air.nasalReference.triangles,z).flat(),palate=planeSegments(p.vertices,p.triangles,z).flat();
      const outside=[...nasal,...palate].filter(v=>!contains(v,h));
      bounds.push({name,z,outside:outside.length,total:nasal.length+palate.length,examples:outside.slice(0,3)});
    }
  }
  results.push({name,profileVertices:p.profile.length,open:s.nasal.port_area>0,area});
}
const report={passed:true,cases:results.length,maxPalateStepCm:maxDelta,maxBranchStepCm:maxBranchDelta,maxGateStepCm:maxGateDelta,maxUvulaShapeDeviationCm:maxUvulaDelta,bounds};
writeFileSync(new URL(process.argv[2]??'output/validation/m10-r11/geometry-report.json',root),JSON.stringify(report,null,2));
console.log(JSON.stringify(report));
assert.ok(bounds.every(x=>x.outside===0),'Nasal/velum reference outside head envelope');
